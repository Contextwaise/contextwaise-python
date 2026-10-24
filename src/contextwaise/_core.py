"""Contextwaise's decisions: how a tool result is shrunk, once, at birth, and never changed after.

Agent loops resend every tool result on every later turn, so a 20K-token test log
read at turn 3 of a 20-turn task is paid ~17 more times. A shrink gives the model a
short view of each result: noise removed (``_shapers``), then the parts relevant to
what the agent was doing, within a token budget, with a ref to get the rest back with
``expand``. Nothing is lost: the original stays in the ledger.

Cache safety: providers bill a repeated prompt start at ~0.1x only while it is
byte-identical, and newer Claude models reject edited history. So the view of a tool
call is decided once and replayed from the ledger every time it is asked for again.

Relevance is only judged by the API (Jev). Word matching can't (synonyms, other
languages): it only orders lines inside a group. Without the API nothing is cut: the
model reads the cleaned output whole.

The decisions are written once, free of I/O, as generators: they yield a
``ScoreRequest`` when they need the API and receive the scores back (None when it is
unavailable). The sync and async clients only differ in how they send that request
(``run_steps`` / ``arun_steps``).

Port of the TypeScript SDK's ``sdk/typescript/src/core.ts``: keep the two in step.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import math
import os
import re
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Generator, Iterable, Mapping, Set
from typing import Any, TypeVar, cast

from ._api import Api, ScoreRequest
from ._ledger import BoundedDict, Ledger, LedgerEntry, MemoryLedger
from ._relevance import word_overlap
from ._shapers import is_error_line, sections, shape, tokens
from ._types import Kind, Scorer, SectionLog, SectionReason, ShrinkResult

logger = logging.getLogger("contextwaise")

T = TypeVar("T")
Scores = list[float] | None
Steps = Generator[ScoreRequest, Scores, T]

DEFAULT_MAX_TOKENS = 600
# Budget for what one expand returns.
EXPAND_BUDGET = 3000
# How many budgets a cleaned output may use when relevance can't tell its parts apart.
UNIFORM_KEEP_FACTOR = 3
# Sections sent to the API for one result: about one per MIN_GROUP_TOKENS of text, at
# most MAX_REMOTE_SECTIONS (the API rejects more than 200). Each section is a Jev
# question (~280 billed tokens, measured), so this keeps scoring near 1.5x the text.
MIN_GROUP_TOKENS = 1000
MAX_REMOTE_SECTIONS = 120
# Weight of word overlap inside a group scored by the API: orders lines, never decides.
TIE_BREAK = 0.05
# A cut must save at least this many tokens, and this share of the cleaned output, or
# the result passes whole. When the model needs a cut part, expand costs one more model
# call resending the whole conversation: a small saving can't pay for that risk.
MIN_TRIM_SAVING = 1000
MIN_TRIM_SHARE = 0.3
# When the prompt size is known, a cut must also save this share of it: about what a wrong
# cut costs (one expand resends the prompt) times how often cuts get expanded. A cut saves on
# every later step, an expand costs once. At 0.25, agents with big prompts (20-50K tokens of
# instructions, tool lists and schemas) never cut even a 13K-token query result.
EXPAND_RISK = 0.1
# Below this a section is noise for the current intent: the API's Jev scores put
# "loosely related, not needed" up to ~0.4 and "useful evidence" from 0.5.
MIN_RELEVANCE = 0.45
# Bound on the intents noted per process, so a long-running server doesn't grow forever.
MAX_TRACKED_CALLS = 5000
MAX_NOTED_INTENT_CHARS = 500

EXPAND_TOOL: dict[str, Any] = {
    "name": "expand",
    "description": (
        "Get back parts of a tool result that Contextwaise trimmed. Pass the ref shown in the result's "
        "[contextwaise …] note, and a question describing what you need."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "ref": {"type": "string"},
            "question": {"type": "string", "description": "What you are looking for in the full output."},
        },
        "required": ["ref"],
        "additionalProperties": False,
    },
}
"""The ``expand`` tool to give the model, as a name, a description and a JSON Schema.

Map it to your provider's format: ``parameters`` is ``input_schema`` for Anthropic.
Call ``expand(**arguments)`` when the model uses it.
"""


class BaseContextwaise:
    """State and decisions shared by the sync and async clients."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        agent: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        budgets: Mapping[str, int] | None = None,
        full: Iterable[str] = (),
        ledger: Ledger | None = None,
    ) -> None:
        """
        Args:
            api_key: Your Contextwaise API key (Settings > API keys). Default: the
                ``CONTEXTWAISE_API_KEY`` environment variable. With a key, relevance is
                judged by the Contextwaise API, and every result is reported to your
                dashboard (sizes and tool names only, never tool output). Without one,
                only the local cleanup runs: noise is removed, nothing relevant is cut.
            base_url: Where your Contextwaise app runs. Default: the ``CONTEXTWAISE_URL``
                environment variable. Required with an API key.
            agent: Groups this agent's results on the dashboard, e.g. ``"support-triage"``.
            max_tokens: Budget for the model's view of one tool result.
            budgets: Per-tool budgets overriding ``max_tokens``, e.g. ``{"read_file": 1500}``:
                files up to that size pass whole (an agent may edit them); larger ones
                keep the relevant parts.
            full: Tools whose output the model must always see whole.
            ledger: Where originals and views live. Default: in memory, the most recent
                5,000 tool calls. Use a shared one for agents spread over processes.

        Raises:
            ValueError: An API key without a valid ``base_url``, or a budget below 1.
        """
        api_key = api_key if api_key is not None else os.environ.get("CONTEXTWAISE_API_KEY")
        base_url = base_url if base_url is not None else os.environ.get("CONTEXTWAISE_URL")
        budgets = dict(budgets or {})
        if min([max_tokens, *budgets.values()]) < 1:
            raise ValueError("contextwaise: token budgets must be at least 1.")
        self._api: Api | None = None
        if api_key:
            if not base_url or not base_url.startswith(("http://", "https://")):
                raise ValueError(
                    "contextwaise: an API key needs base_url (or CONTEXTWAISE_URL): "
                    "the http(s) address of your Contextwaise app."
                )
            self._api = Api(api_key=api_key, base_url=base_url, agent=agent)
        self._max_tokens = max_tokens
        self._budgets = budgets
        self._full = frozenset(full)
        self._ledger: Ledger = ledger if ledger is not None else MemoryLedger()
        self._intents: BoundedDict[str, str] = BoundedDict(MAX_TRACKED_CALLS)
        # Call id → tokens of the prompt that made the call.
        self._contexts: BoundedDict[str, int] = BoundedDict(MAX_TRACKED_CALLS)
        # Call id → parts the model already has (its view and earlier expands).
        self._shown: BoundedDict[str, set[int]] = BoundedDict(MAX_TRACKED_CALLS)

    def note_intent(self, call_id: str, text: str, *, context_tokens: int | None = None) -> None:
        """Records why the agent made a tool call, before its result is shrunk.

        Pass what the model was asked and the text it wrote just before the call
        ("let me find the auth middleware"): arguments alone often don't say (``ls .``).
        ``context_tokens``: the size of the prompt that made the call. A cut must save more
        the bigger it is, since a wrong cut costs an expand: one more model call resending all of it.
        """
        text = text.strip()
        if text:
            self._intents.set(call_id, text[-MAX_NOTED_INTENT_CHARS:])
        if context_tokens:
            self._contexts.set(call_id, context_tokens)

    # ---- shrink ----

    def _shrink_steps(
        self,
        call_id: str,
        tool: str,
        args: Any,
        output: Any,
        integration: str = "python",
        model: str | None = None,
        full: bool = False,
    ) -> Steps[ShrinkResult]:
        ref = _ref(call_id)
        known = self._ledger.get(ref)
        if known is not None:
            logger.debug("replay %s (%s): %d tokens", call_id, tool, known.result.tokens_after)
            return known.result

        started = time.perf_counter()
        original = output if isinstance(output, str) else _to_json_text(output)
        intent = " · ".join(filter(None, [self._intents.pop(call_id), _intent_of(tool, args)]))
        min_saving = max(MIN_TRIM_SAVING, EXPAND_RISK * (self._contexts.pop(call_id) or 0))
        size = tokens(original)
        # The untouched view: what the model reads when no smaller one is worth sending.
        base = ShrinkResult(
            text=original,
            mode="full",
            ref=ref,
            kind="text",
            tokens_before=size,
            tokens_after=size,
            intent=intent,
        )

        try:
            result = yield from self._view_steps(base, call_id, tool, original, min_saving, full)
        except Exception:  # a view must never cost the agent its result: the output passes as is
            logger.exception("shrink failed for %s (%s): output passed as is", call_id, tool)
            result = base
        if result.mode != "full" and result.tokens_after >= result.tokens_before:
            result = dataclasses.replace(
                base, kind=result.kind, notes=("no smaller view possible: sent as is",)
            )

        self._ledger.set(ref, LedgerEntry(call_id=call_id, tool=tool, original=original, result=result))
        ms = round((time.perf_counter() - started) * 1000)
        logger.debug(
            "shrink %s (%s): %s, %d → %d tokens", call_id, tool, result.mode, size, result.tokens_after
        )
        if self._api:
            self._api.report_result(
                call_id=call_id, tool=tool, result=result, ms=ms, integration=integration, model=model
            )
        return result

    def _view_steps(
        self, base: ShrinkResult, call_id: str, tool: str, original: str, min_saving: float, full: bool
    ) -> Steps[ShrinkResult]:
        """The view, from cheapest to most involved: untouched, noise removed, relevant parts."""
        if full:
            return dataclasses.replace(base, notes=("this call is set to pass whole",))
        if tool in self._full:
            return dataclasses.replace(base, notes=(f'"{tool}" is set to always pass whole',))

        def worth_cutting(whole_tokens: int, cut_tokens: int) -> bool:
            saving = whole_tokens - cut_tokens
            return saving >= min_saving and saving >= MIN_TRIM_SHARE * whole_tokens

        budget = self._budgets.get(tool, self._max_tokens)
        shaped = shape(original)

        def whole(notes: tuple[str, ...], scorer: Scorer = "none") -> ShrinkResult:
            if not notes:
                return dataclasses.replace(base, kind=shaped.kind, scorer=scorer)
            text = f"{shaped.text}\n[contextwaise: {'; '.join(notes)}. Everything else is shown in full.]"
            return dataclasses.replace(
                base,
                text=text,
                mode="shaped",
                kind=shaped.kind,
                tokens_after=tokens(text),
                notes=notes,
                scorer=scorer,
            )

        if tokens(shaped.text) <= budget or not worth_cutting(tokens(shaped.text), budget):
            return whole(shaped.notes)  # nothing to score: the API isn't called

        units = sections(shaped)
        scores = yield from self._score_steps(call_id, tool, base.intent, units)
        # No relevance, no cut: a guessed cut would silently cost answers.
        if scores is None:
            return whole(shaped.notes)
        # Every part equally relevant (e.g. "all of Acme's orders"): any cut would be
        # arbitrary and the agent would have to expand it all back. Keep the cleaned
        # output whole when it stays within a few budgets.
        if max(scores) - min(scores) < 0.05 and tokens(shaped.text) <= budget * UNIFORM_KEEP_FACTOR:
            return whole((*shaped.notes, "every part equally relevant: kept whole"), "jev")

        picked = _pick(units, scores, budget, shaped.kind)
        kept = sum(log.kept for log in picked)
        # Nothing fits the budget (one part bigger than all of it): an empty view only sends the
        # model to expand, one more call resending the conversation. Send the output instead.
        if not kept:
            return whole(shaped.notes, "jev")
        footer = _footer(
            base.ref,
            base.tokens_before,
            kept,
            len(units),
            _omitted_summary(units, picked, shaped.kind),
            shaped.notes,
        )
        text = f"{_render(units, picked, shaped.kind)}\n{footer}"
        # Errors are kept over budget: the actual cut can be smaller than planned.
        if not worth_cutting(tokens(shaped.text), tokens(text)):
            return whole(shaped.notes, "jev")
        return dataclasses.replace(
            base,
            text=text,
            mode="trimmed",
            kind=shaped.kind,
            tokens_after=tokens(text),
            notes=shaped.notes,
            sections=picked,
            scorer="jev",
        )

    # ---- expand ----

    def _expand_steps(self, ref: str, question: str | None, context_tokens: int | None) -> Steps[str]:
        entry = self._ledger.get(ref)
        if entry is None:
            return f"No output found for {ref}."
        shaped = shape(entry.original)
        units = sections(shaped)
        scores: Scores = None
        if question and question.strip():
            # Each question is its own scoring decision, replayed by the API like any tool call.
            scores = yield from self._score_steps(
                f"{entry.call_id}:x{_digest(question)}", entry.tool, question, units
            )
        # Parts the model already has, in the view or from an earlier expand, are not sent again:
        # asked twice, expand returns the next most relevant parts, so a model that didn't get what
        # it needed can't loop on the same answer (each loop is a model call resending everything).
        seen = self._shown.get(entry.call_id)
        if seen is None:
            seen = {i for i, log in enumerate(entry.result.sections) if log.kept}
        picked = _pick(units, scores, EXPAND_BUDGET, shaped.kind, seen) if scores is not None else None
        if picked is not None and seen and not any(log.kept and log.reason != "preview" for log in picked):
            text = f"Nothing more in {ref} matches that question: its relevant parts are already shown above."
        elif picked is not None:
            self._shown.set(entry.call_id, seen | {i for i, log in enumerate(picked) if log.kept})
            text = _render(units, picked, shaped.kind)
        elif tokens(entry.original) <= EXPAND_BUDGET:
            text = entry.original
        else:
            why = (
                "the part you asked for could not be found right now"
                if question
                else "ask expand with a question to get a specific part"
            )
            text = f"{entry.original[: EXPAND_BUDGET * 4]}\n… (truncated: {why})"
        logger.debug("expand %s (%s): %d tokens", ref, question, tokens(text))
        if self._api:
            self._api.report_expand(
                call_id=entry.call_id, tool=entry.tool, tokens=tokens(text), context_tokens=context_tokens
            )
        return text

    # ---- relevance ----

    def _score_steps(self, call_id: str, tool: str, intent: str, units: list[str]) -> Steps[Scores]:
        """Each unit's relevance (0 to 1) to ``intent``, judged by the API; None without it."""
        if self._api is None or not units or not intent.strip():
            return None
        # Jev bills a question per section: hundreds of one-line sections (logs,
        # listings) would cost several times the text itself (measured: 8.9x for a
        # 1,214-line log). Score consecutive groups instead, each line taking its
        # group's score; errors are still kept on their own.
        groups = _group_consecutive(units, MAX_REMOTE_SECTIONS)
        scores = yield ScoreRequest(call_id, tool, intent, [_for_scoring(text, intent) for text, _ in groups])
        if scores is None:
            logger.debug("no relevance for %s: cleaned output passed whole", call_id)
            return None
        # Lines of a group share its score; word overlap breaks the tie inside it (a
        # nudge too small to lift a line over the keep floor on its own).
        overlap = word_overlap(intent, units)
        out: list[float] = []
        group = 0
        for i, nudge in enumerate(overlap):
            while i >= groups[group][1]:
                group += 1
            out.append((1 - TIE_BREAK) * scores[group] + TIE_BREAK * nudge)
        return out


def run_steps(steps: Steps[T], score: Callable[[ScoreRequest], Scores]) -> T:
    """Runs a decision, sending its score requests with ``score``."""
    try:
        request = next(steps)
        while True:
            request = steps.send(score(request))
    except StopIteration as done:
        return cast(T, done.value)


async def arun_steps(steps: Steps[T], score: Callable[[ScoreRequest], Awaitable[Scores]]) -> T:
    """Runs a decision, sending its score requests with the coroutine ``score``."""
    try:
        request = next(steps)
        while True:
            request = steps.send(await score(request))
    except StopIteration as done:
        return cast(T, done.value)


# ---- helpers ----


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode(), digest_size=8).hexdigest()


def _ref(call_id: str) -> str:
    """The model's handle on a tool call: short, stable, and derived from the call id,
    so any process sharing the ledger can expand it."""
    return f"cw_{_digest(call_id)}"


def _to_json_text(value: Any) -> str:
    """A non-text tool output as compact JSON, the way providers send it (the honest size to
    measure savings against). Pydantic models and dataclasses become their fields."""

    def fields(item: Any) -> Any:
        if hasattr(item, "model_dump"):  # Pydantic
            return item.model_dump(mode="json")
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            return dataclasses.asdict(item)
        return str(item)

    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=fields)
    except ValueError:  # circular: no provider could send it as JSON either
        return str(value)


def _intent_of(tool: str, args: Any) -> str:
    """What the agent was after when it called the tool: the tool name and every argument value."""
    if isinstance(args, str):  # OpenAI passes arguments as JSON text
        with contextlib.suppress(ValueError):
            args = json.loads(args)
    values: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, (str, int, float)):
            values.append(str(value))
        elif isinstance(value, Mapping):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(args)
    return " ".join([re.sub(r"[_-]", " ", tool), *values])


_ASKED_NUMBER = re.compile(r"\b[0-9a-f]{12,}\b|\d{4,}", re.IGNORECASE | re.ASCII)
_DATETIME = re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]?(?:\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?Z?)?", re.ASCII)
_TIME = re.compile(r"\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b", re.ASCII)
_HEX_ID = re.compile(r"\b[0-9a-f]{12,}\b", re.IGNORECASE | re.ASCII)
_LONG_NUMBER = re.compile(r"\d{4,}", re.ASCII)
_SPACES = re.compile(r"[ \t]{2,}")


def _for_scoring(text: str, intent: str) -> str:
    """The copy of a section sent to be scored: dates, times and long numbers removed.

    They rarely decide relevance, and dense digits cost Jev up to ~2x the tokens of
    prose (measured on logs). The model still reads the original lines. Numbers the
    intent mentions are kept: when the agent asks about ORD-1337, that id is the
    needle, and masking it would make every order look the same.
    """
    asked = {m.lower() for m in _ASKED_NUMBER.findall(intent)}

    def mask(match: re.Match[str]) -> str:
        return match.group() if match.group().lower() in asked else "#"

    text = _DATETIME.sub("", text)
    text = _TIME.sub("", text)
    text = _HEX_ID.sub(mask, text)
    text = _LONG_NUMBER.sub(mask, text)
    return _SPACES.sub(" ", text)


def _group_consecutive(units: list[str], max_groups: int) -> list[tuple[str, int]]:
    """Joins consecutive units into groups of about ``MIN_GROUP_TOKENS``, at most ``max_groups``.

    Packed by size, not count: short lines share a group (one Jev question for many), while a
    unit that is big on its own (a record) is scored alone, never tied with its neighbour.
    Returns (text, exclusive end index) pairs.
    """
    packed: list[tuple[str, int]] = []
    for i, unit in enumerate(units):
        if packed and tokens(packed[-1][0]) + tokens(unit) <= MIN_GROUP_TOKENS:
            packed[-1] = (f"{packed[-1][0]}\n{unit}", i + 1)
        else:
            packed.append((unit, i + 1))
    if len(packed) <= max_groups:
        return packed
    size = math.ceil(len(units) / max_groups)
    return [
        ("\n".join(units[start : start + size]), min(start + size, len(units)))
        for start in range(0, len(units), size)
    ]


def _pick(
    units: list[str], scores: list[float], budget: int, kind: Kind, exclude: Set[int] = frozenset()
) -> tuple[SectionLog, ...]:
    """Keeps error sections, then the most relevant ones, within budget; drops the rest.

    Order is preserved. ``exclude``: parts never to pick (already shown to the model).
    """
    sizes = [tokens(u) for u in units]
    errors = [_is_error_unit(u, kind) for u in units]
    kept = [False] * len(units)
    reasons: list[SectionReason] = ["low_relevance"] * len(units)
    used = 0
    for i in sorted(range(len(units)), key=lambda i: (not errors[i], -scores[i], i)):
        if i in exclude:
            continue
        if not errors[i] and scores[i] < MIN_RELEVANCE:
            continue
        if used + sizes[i] > budget and used > 0:
            reasons[i] = "over_budget"
            continue
        used += sizes[i]
        kept[i] = True
        reasons[i] = "error" if errors[i] else "relevant"
    # Nothing matched the intent: show the start rather than an empty view, so the
    # model sees what kind of output this is and can expand with a better question.
    if "relevant" not in reasons:
        for i in range(len(units)):
            if kept[i] or i in exclude:
                continue
            if used + sizes[i] > budget / 2:
                break
            used += sizes[i]
            kept[i] = True
            reasons[i] = "preview"
    return tuple(
        SectionLog(
            preview=units[i][:120],
            tokens=sizes[i],
            score=round(scores[i], 2),
            kept=kept[i],
            reason=reasons[i],
        )
        for i in range(len(units))
    )


_ERROR_KEY = re.compile(r"(?:errors?|exception|fault)\b", re.IGNORECASE | re.ASCII)


def _is_error_unit(unit: str, kind: Kind) -> bool:
    """A part reporting a problem, kept first. In JSON only an error field counts: an item that
    merely mentions one ("payment": "failed", "timeout": 30) is data, and would crowd out the
    relevant ones."""
    if kind != "json":
        return any(is_error_line(line) for line in unit.split("\n"))
    value = json.loads(unit)
    return isinstance(value, dict) and any(_ERROR_KEY.match(key) for key in value)


def _render(units: list[str], picked: tuple[SectionLog, ...], kind: Kind) -> str:
    """The kept sections in their original order, each gap shown as one line saying which
    parts are missing ("… lines 12-240 omitted"), so the model knows what it didn't see."""
    lines_like = kind in ("listing", "log", "lines")
    unit = "line" if lines_like else "item" if kind == "json" else "section"
    out: list[str] = []
    gap_start: int | None = None

    def close_gap(end: int) -> None:
        if gap_start is not None:
            where = f"{unit} {gap_start + 1}" if end == gap_start + 1 else f"{unit}s {gap_start + 1}-{end}"
            out.append(f"… {where} omitted")

    for i, (text, log) in enumerate(zip(units, picked, strict=True)):
        if not log.kept:
            gap_start = i if gap_start is None else gap_start
            continue
        close_gap(i)
        gap_start = None
        out.append(text)
    close_gap(len(units))
    return ("\n" if lines_like or kind == "json" else "\n\n").join(out)


# What the omitted-parts summary shows: at most this many fields, each with few values.
MAX_SUMMARY_FIELDS = 3
MAX_FIELD_VALUES = 4
MAX_VALUE_CHARS = 40
_LOG_LEVEL = re.compile(r"\b(ERROR|WARN(?:ING)?|INFO|DEBUG|TRACE)\b", re.ASCII)
_ENVELOPE_KEY = re.compile(r"^\w+\[\d+\]$", re.ASCII)


def _omitted_summary(units: list[str], picked: tuple[SectionLog, ...], kind: Kind) -> str:
    """What the omitted parts hold, counted in code, so common follow-up questions ("how
    many are delivered?") are answered without an expand: the values of the omitted JSON
    items' categorical fields, or the levels of the omitted log lines. Empty when unknown."""
    omitted = [text for text, log in zip(units, picked, strict=True) if not log.kept]
    if kind == "log":
        levels = Counter(m[1] for line in omitted if (m := _LOG_LEVEL.search(line)))
        return (
            f"{len(omitted)} lines, " + ", ".join(f"{lvl} ×{n}" for lvl, n in levels.most_common())
            if levels
            else ""
        )
    if kind != "json":
        return ""
    records = [r for r in map(_record, omitted) if r is not None]
    values: dict[str, Counter[str]] = {}
    for record in records:
        for key, value in record.items():
            if isinstance(value, bool):
                value = "true" if value else "false"
            if isinstance(value, str) and len(value) <= MAX_VALUE_CHARS:
                values.setdefault(key, Counter())[value] += 1
    fields = [
        f"{key}: " + ", ".join(f"{v} ×{n}" for v, n in counts.most_common())
        for key, counts in values.items()
        if len(counts) <= MAX_FIELD_VALUES and counts.total() * 2 >= len(records)
    ][:MAX_SUMMARY_FIELDS]
    return f"{len(omitted)} items; " + "; ".join(fields) if fields else ""


def _record(unit: str) -> dict[str, Any] | None:
    """An omitted JSON unit as an object: an array item, or one under its envelope key (``orders[3]``)."""
    value = json.loads(unit)
    if isinstance(value, dict) and len(value) == 1:
        [(key, inner)] = value.items()
        if _ENVELOPE_KEY.match(key):
            value = inner
    return value if isinstance(value, dict) else None


def _footer(
    ref: str, original_tokens: int, kept: int, total: int, omitted: str, notes: tuple[str, ...]
) -> str:
    """Only trimmed views invite expand: an invitation on a view that lost nothing but
    noise makes careful models spend a whole extra round trip to "check for more"."""
    what = f" Omitted: {omitted}." if omitted else ""
    also = f" Also {'; '.join(notes)}." if notes else ""
    return (
        f"[contextwaise: kept the {kept} of {total} parts relevant to this step "
        f"(of {original_tokens} tokens).{what}{also} "
        f'If you need an omitted part, call expand({{ ref: "{ref}", question: "<what you need>" }})]'
    )
