"""The Contextwaise API, seen from the SDK.

Two calls: relevance scoring (Jev, with the API's key, never shipped in the SDK) and
usage reports for the dashboard. Neither may ever slow down or break the agent:

- scoring has a short timeout; after a failure the API is skipped for a while and the
  model gets the cleaned output whole instead (nothing cut);
- reports are batched, sent from a background thread, and dropped if they can't be
  sent. They carry sizes, tool names and the model id only, never tool output.

This module owns the API contract (paths, payloads, request limits): the rest of the
SDK never builds a request.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from ._types import ShrinkResult
from ._version import __version__

logger = logging.getLogger("contextwaise")

TIMEOUT_SECONDS = 4.0
PAUSE_AFTER_FAILURE_SECONDS = 30.0
# The free tokens are used up: only an upgrade brings scoring back, so retrying every 30s is pointless.
PAUSE_AFTER_QUOTA_SECONDS = 3600.0
REPORT_BATCH = 50
REPORT_DELAY_SECONDS = 2.0
MAX_QUEUED_REPORTS = 1000

# The API's request limits (LIMITS and CALL_ID in the app's src/lib/tool-results.ts).
# Lengths are JavaScript string lengths: UTF-16 code units.
_MAX_SECTION_CHARS = 20_000
_MAX_INTENT_CHARS = 4000
_CALL_ID = re.compile(r"[\w.:-]{1,128}", re.ASCII)


@dataclass(frozen=True, slots=True)
class ScoreRequest:
    """Sections of one tool result to score for what the agent was after."""

    call_id: str
    tool: str
    intent: str
    sections: Sequence[str]
    agent: str | None = None
    """Overrides the client's agent, for apps hosting many agents on one client."""


class Api:
    """A connection to a Contextwaise app. Thread-safe."""

    def __init__(self, *, api_key: str, base_url: str, agent: str | None) -> None:
        self._base_url = base_url.rstrip("/")
        self._agent = agent
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": f"contextwaise-python/{__version__}",
        }
        self._http = httpx.Client(headers=self._headers, timeout=TIMEOUT_SECONDS)
        self._async_http: httpx.AsyncClient | None = None
        self._paused_until = 0.0
        self._reports = _ReportQueue(self._send_reports)

    # ---- scoring ----
    # Everything but the send is shared, so the sync and async paths can't drift.

    def score(self, request: ScoreRequest) -> list[float] | None:
        """Each section's relevance, 0 to 1, or None when the API is unavailable."""
        http_request = self._score_request(request)
        if http_request is None:
            return None
        try:
            return self._read_scores(self._http.send(http_request), len(request.sections))
        except _RejectedError as error:
            logger.warning("relevance API rejected this result, passed whole: %s", error)
            return None
        except _QuotaError as error:
            self._pause(error, PAUSE_AFTER_QUOTA_SECONDS)
            return None
        except Exception as error:  # the API must never break the agent
            self._pause(error)
            return None

    async def ascore(self, request: ScoreRequest) -> list[float] | None:
        """``score``, without blocking the event loop."""
        http_request = self._score_request(request)
        if http_request is None:
            return None
        if self._async_http is None:
            self._async_http = httpx.AsyncClient(headers=self._headers, timeout=TIMEOUT_SECONDS)
        try:
            return self._read_scores(await self._async_http.send(http_request), len(request.sections))
        except _RejectedError as error:
            logger.warning("relevance API rejected this result, passed whole: %s", error)
            return None
        except _QuotaError as error:
            self._pause(error, PAUSE_AFTER_QUOTA_SECONDS)
            return None
        except Exception as error:  # the API must never break the agent
            self._pause(error)
            return None

    def _score_request(self, request: ScoreRequest) -> httpx.Request | None:
        if time.monotonic() < self._paused_until:
            return None
        body = {
            "callId": _api_call_id(request.call_id),
            "tool": request.tool,
            "intent": _clip(request.intent, _MAX_INTENT_CHARS),
            # One oversized section would make the API reject the whole call: a huge
            # unit (one minified blob) is scored on its start instead.
            "sections": [_clip(section, _MAX_SECTION_CHARS) for section in request.sections],
            **self._agent_field(request.agent),
        }
        return self._http.build_request("POST", f"{self._base_url}/api/v1/tool-results", json=body)

    @staticmethod
    def _read_scores(response: httpx.Response, expected: int) -> list[float]:
        # A request the API rejects (too big, malformed) is this call's problem: the next one may score fine.
        if response.status_code in (400, 413, 422):
            raise _RejectedError(f"HTTP {response.status_code}: {response.text[:160]}")
        if response.status_code == 429 and "quota_exceeded" in response.text:
            raise _QuotaError(f"free Contextwaise tokens used up (upgrade in Billing): {response.text[:160]}")
        response.raise_for_status()
        scores = response.json().get("scores")
        if not isinstance(scores, list) or len(scores) != expected:
            raise ValueError(f"expected {expected} scores, got {scores!r:.80}")
        if not all(isinstance(s, (int, float)) and not isinstance(s, bool) for s in scores):
            raise ValueError("scores must be numbers")
        return [float(s) for s in scores]

    def _pause(self, error: Exception, seconds: float = PAUSE_AFTER_FAILURE_SECONDS) -> None:
        self._paused_until = time.monotonic() + seconds
        logger.warning("relevance API unavailable, passing cleaned outputs whole for %ds: %s", seconds, error)

    # ---- usage reports ----

    def report_result(
        self,
        *,
        call_id: str,
        tool: str,
        result: ShrinkResult,
        ms: int,
        integration: str,
        model: str | None,
        agent: str | None = None,
    ) -> None:
        """Queues the report of one shrunk result. Never blocks, never raises."""
        self._reports.add(
            {
                **({"model": model} if model else {}),
                **({"agent": agent} if agent else {}),
                "type": "result",
                "callId": _api_call_id(call_id),
                "tool": tool,
                "integration": integration,
                "kind": result.kind,
                "mode": result.mode,
                "reason": result.reason,
                "scorer": result.scorer,
                "tokensBefore": result.tokens_before,
                "tokensAfter": result.tokens_after,
                "ms": ms,
            }
        )

    def report_expand(
        self, *, call_id: str, tool: str, tokens: int, context_tokens: int | None, agent: str | None = None
    ) -> None:
        """Queues the report of one expand. Never blocks, never raises."""
        report: dict[str, Any] = {
            **({"agent": agent} if agent else {}),
            "type": "expand",
            "callId": _api_call_id(call_id),
            "tool": tool,
            "tokens": tokens,
        }
        if context_tokens is not None:
            report["contextTokens"] = context_tokens
        self._reports.add(report)

    def flush(self) -> None:
        """Sends queued reports now, and waits for them."""
        self._reports.flush()

    async def aflush(self) -> None:
        """``flush``, without blocking the event loop."""
        await asyncio.to_thread(self.flush)

    def close(self) -> None:
        """Sends queued reports, then releases connections."""
        self.flush()
        self._http.close()

    async def aclose(self) -> None:
        """``close``, without blocking the event loop."""
        await asyncio.to_thread(self.close)
        if self._async_http is not None:
            await self._async_http.aclose()

    def _send_reports(self, batch: list[dict[str, Any]]) -> None:
        try:
            response = self._http.post(
                f"{self._base_url}/api/v1/tool-results/report", json={"results": batch, **self._agent_field()}
            )
            response.raise_for_status()
        except Exception as error:  # reports must never break the agent
            logger.warning("usage report dropped (%d results): %s", len(batch), error)

    def _agent_field(self, agent: str | None = None) -> dict[str, str]:
        agent = agent or self._agent
        return {"agent": agent} if agent else {}


class _RejectedError(Exception):
    """The API refused one scoring request: no reason to pause scoring for the others."""


class _QuotaError(Exception):
    """The free tokens are used up: scoring pauses for long, since only an upgrade fixes it."""


class _ReportQueue:
    """Reports waiting to be sent, in batches, by a daemon thread.

    A batch leaves when it is full or ``REPORT_DELAY_SECONDS`` after its first report.
    The thread never keeps the process alive: short-lived processes call ``flush()``.
    """

    def __init__(self, send: Callable[[list[dict[str, Any]]], None]) -> None:
        self._send = send
        # The API unreachable for long: the oldest reports are dropped first.
        self._queue: deque[dict[str, Any]] = deque(maxlen=MAX_QUEUED_REPORTS)
        self._queued = threading.Event()
        self._batch_full = threading.Event()
        self._sending = threading.Lock()
        self._starting = threading.Lock()
        self._worker: threading.Thread | None = None

    def add(self, report: dict[str, Any]) -> None:
        self._queue.append(report)
        self._ensure_worker()
        self._queued.set()
        if len(self._queue) >= REPORT_BATCH:
            self._batch_full.set()

    def flush(self) -> None:
        # One sender at a time: a flush waits for the batch the worker is sending.
        with self._sending:
            while batch := self._take_batch():
                self._send(batch)

    def _take_batch(self) -> list[dict[str, Any]]:
        # Safe without a lock: senders are serialized, and appends never shorten the queue.
        return [self._queue.popleft() for _ in range(min(REPORT_BATCH, len(self._queue)))]

    def _ensure_worker(self) -> None:
        # Checked for liveness, not existence: threads don't survive a fork.
        if self._worker is not None and self._worker.is_alive():
            return
        with self._starting:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run, name="contextwaise-reports", daemon=True)
                self._worker.start()

    def _run(self) -> None:
        while True:
            self._queued.wait()
            self._batch_full.wait(REPORT_DELAY_SECONDS)
            # Cleared before sending: a report added meanwhile sets them again.
            self._queued.clear()
            self._batch_full.clear()
            self.flush()


def _clip(text: str, limit: int) -> str:
    """``text`` cut to at most ``limit`` UTF-16 code units, the unit the API measures in."""
    if len(text) <= limit // 2:  # fits even if every character takes two units
        return text
    # A code point is one or two units: slicing to `limit` code points first bounds the encoding work.
    return text[:limit].encode("utf-16-le")[: 2 * limit].decode("utf-16-le", "ignore")


def _api_call_id(call_id: str) -> str:
    """The id the API accepts for a tool call: the call id itself, or a digest of an unusual one."""
    if _CALL_ID.fullmatch(call_id):
        return call_id
    return "cw_" + hashlib.blake2b(call_id.encode(), digest_size=16).hexdigest()
