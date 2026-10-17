"""The data Contextwaise hands back to callers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["json", "html", "test", "listing", "log", "lines", "text"]
"""What a tool output looks like, which decides how it is cleaned and cut. ``lines``: rows
without paragraphs (CSV, query results, schema dumps, tables), cut one line at a time."""

Mode = Literal["full", "shaped", "trimmed"]
Scorer = Literal["jev", "none"]
SectionReason = Literal["relevant", "error", "preview", "low_relevance", "over_budget"]


@dataclass(frozen=True, slots=True)
class SectionLog:
    """How one part of a trimmed result was judged: why it was kept or dropped."""

    preview: str
    tokens: int
    score: float
    kept: bool
    reason: SectionReason


@dataclass(frozen=True, slots=True)
class ShrinkResult:
    """The model's view of one tool result. Immutable: the same view is replayed for the same call.

    Attributes:
        text: What the model reads.
        mode: ``full``: untouched. ``shaped``: noise removed, fits the budget.
            ``trimmed``: relevant parts only, the rest is available through ``expand``.
        ref: The handle the model passes to ``expand`` to get trimmed parts back.
        kind: What the output looked like (JSON, a log, a test run, …).
        tokens_before: Estimated tokens of the original output.
        tokens_after: Estimated tokens of ``text``.
        notes: What the cleanup did, in words.
        sections: For trimmed results, how each part was judged.
        scorer: ``jev`` when the Contextwaise API judged relevance, else ``none``.
        intent: What the agent was after, as sent for relevance.
    """

    text: str
    mode: Mode
    ref: str
    kind: Kind
    tokens_before: int
    tokens_after: int
    notes: tuple[str, ...] = ()
    sections: tuple[SectionLog, ...] = ()
    scorer: Scorer = "none"
    intent: str = ""
