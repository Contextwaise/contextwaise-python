"""Test data and the Contextwaise API stub. No network."""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

BASE_URL = "http://app.test"
API_KEY = "sw_live_test"

TEST_RUN = "\n".join(
    [
        *(f"  ✓ module {i} works ({i % 9}ms)" for i in range(300)),
        "  ✗ checkout applies discount",
        "    AssertionError: expected 90 to equal 81 at src/checkout.ts:42",
        "Tests: 1 failed, 300 passed",
    ]
)

_GENERIC = {"list", "files", "find", "where", "read", "file"}


def relevance_stub(intent: str, sections: list[str]) -> list[float]:
    """The API's stand-in: a section is relevant when it contains one of the intent's
    distinctive words (4+ letters). Groups of lines are scored as one, like the real API."""
    needles = [w for w in re.findall(r"[a-z0-9]{4,}", intent.lower()) if w not in _GENERIC]
    return [1.0 if any(n in s.lower() for n in needles) else 0.0 for s in sections]


class ApiStub:
    """What the stubbed API received, and a switch to take it down."""

    def __init__(self) -> None:
        self.score_bodies: list[dict[str, Any]] = []
        self.report_bodies: list[dict[str, Any]] = []
        self.down = False
        self.reject_next = False

    def score(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.score_bodies.append(body)
        if self.down:
            return httpx.Response(503, text="down")
        if self.reject_next:
            self.reject_next = False
            return httpx.Response(422, json={"error": {"code": "invalid_request"}})
        # Like the real API: one oversized section rejects the whole call.
        if any(len(s) > 20_000 for s in body["sections"]):
            return httpx.Response(422, json={"error": {"code": "invalid_request"}})
        return httpx.Response(200, json={"scores": relevance_stub(body["intent"], body["sections"])})

    def report(self, request: httpx.Request) -> httpx.Response:
        self.report_bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"stored": 1})
