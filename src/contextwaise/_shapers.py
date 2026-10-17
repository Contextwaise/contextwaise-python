"""Free, deterministic cleanup of tool output, by content type.

No model, no network: the same input always gives the same output, which keeps the
model's view of a result stable (prompt caching) even when it is rebuilt later.

``shape()`` removes what is noise for any reader (color codes, progress bars, passing
tests, repeated log lines, empty JSON fields, HTML markup). ``sections()`` then cuts
the cleaned text into the units that relevance scoring keeps or drops. Errors are
never removed here.

Port of ``core/src/shapers.ts`` (``@contextwaise/core``): keep the two in step. Patterns use
``re.ASCII`` where JavaScript's ``\\d``, ``\\w`` and ``\\b`` are ASCII-only.
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass
from typing import Any

from ._types import Kind


@dataclass(frozen=True, slots=True)
class Shaped:
    """A cleaned tool output: its kind, the cleaned text, and what the cleanup did."""

    kind: Kind
    text: str
    notes: tuple[str, ...]


def tokens(text: str) -> int:
    """Rough token count, chars / 4: the same estimate the Contextwaise API uses."""
    return (len(text) + 3) // 4


_ERROR_LINE = re.compile(
    r"\b(?:error|fail(?:ed|ure)?|exception|traceback|panic|fatal|denied|refused|timeout|cannot"
    r"|undefined is not)\b|✗|✕",
    re.IGNORECASE | re.ASCII,
)


def is_error_line(line: str) -> bool:
    """Whether a line reports a problem: such lines are always kept."""
    return _ERROR_LINE.search(line) is not None


def shape(raw: str) -> Shaped:
    """Removes the noise from a tool output. Lossless for anything a reader would need."""
    notes: list[str] = []
    text = _strip_terminal_noise(raw, notes)
    value = _parse_json(text)
    if value is not _NOT_JSON:
        return Shaped("json", _shape_json(text, value, notes), tuple(notes))
    kind = _detect(text)
    if kind == "html":
        text = _shape_html(text, notes)
    elif kind == "test":
        text = _shape_tests(text, notes)
    elif kind == "listing":
        text = _shape_listing(text, notes)
    elif kind == "log":
        text = _collapse_repeats(text, notes)
    return Shaped(kind, text, tuple(notes))


def sections(shaped: Shaped) -> list[str]:
    """Cuts a shaped output into the units relevance is judged on."""
    if shaped.kind == "json":
        value = json.loads(shaped.text)
        if isinstance(value, list):
            return [_compact(item) for item in value]
        # API envelopes ({ meta, orders: [...] }) put the data in an array under one key:
        # each item becomes its own unit (keyed `orders[3]`), or the whole array would be one part.
        units: list[str] = []
        for key, item in value.items():
            if isinstance(item, list) and len(item) > 1:
                units.extend(_compact({f"{key}[{i}]": element}) for i, element in enumerate(item))
            else:
                units.append(_compact({key: item}))
        return units
    if shaped.kind in ("listing", "log", "lines"):
        return [line for line in shaped.text.split("\n") if line.strip()]  # listing: one directory per line
    if shaped.kind == "test":
        return [block for block in _TEST_BLOCK.split(shaped.text) if block.strip()]
    return [paragraph for paragraph in _PARAGRAPH.split(shaped.text) if paragraph.strip()]


_TEST_BLOCK = re.compile(r"\n(?=\S)")  # one block per unindented line
_PARAGRAPH = re.compile(r"\n\s*\n")


# ---- detection ----

_NOT_JSON = object()
_HTML = re.compile(r"<(?:html|body|div|p|head|article)[\s>]", re.IGNORECASE)
_TEST_RUN = re.compile(r"\b(?:\d+ (?:passing|passed|failing|failed)|Tests?:\s+\d+|PASS |FAIL )", re.ASCII)
_PATH = re.compile(r"[\w@.\-/]+(?:/[\w@.\-]+)+/?", re.ASCII)
_LOG_LINE = re.compile(r"^\[?\d{4}-\d{2}-\d{2}|^\[?\d{2}:\d{2}:\d{2}|\b(?:INFO|WARN|ERROR|DEBUG)\b", re.ASCII)
# Below this many lines, a listing or a log is too short to be worth reshaping.
_MIN_LINES = 15


def _parse_json(text: str) -> Any:
    """The parsed JSON object or array, or ``_NOT_JSON``."""
    if not text.startswith(("{", "[")):
        return _NOT_JSON
    try:
        return json.loads(text)
    except ValueError:
        return _NOT_JSON


def _detect(text: str) -> Kind:
    """The kind of a text that is not JSON."""
    if _HTML.search(text):
        return "html"
    if _TEST_RUN.search(text):
        return "test"
    lines = [line for line in text.split("\n") if line.strip()]
    if len(lines) >= _MIN_LINES:
        if sum(1 for line in lines if _PATH.fullmatch(line.strip())) / len(lines) > 0.7:
            return "listing"
        # Before logs: a CSV whose rows start with a date is data, and the log shaper
        # would collapse its rows, which differ only by numbers, into one.
        if _is_table(lines):
            return "lines"
        if sum(1 for line in lines if _LOG_LINE.search(line)) / len(lines) > 0.5:
            return "log"
        # No paragraphs to cut on (a schema dump, kubectl output): cut by line, or the
        # whole output is one part, kept or dropped as one.
        if max(len(p.split("\n")) for p in _PARAGRAPH.split(text)) >= 0.8 * len(lines):
            return "lines"
    return "text"


def _is_table(lines: list[str]) -> bool:
    """Rows with the same number of a delimiter (CSV, TSV, pipe tables) as the first."""
    for delimiter in (",", "\t", "|"):
        count = lines[0].count(delimiter)
        if count and sum(1 for line in lines if line.count(delimiter) == count) / len(lines) > 0.9:
            return True
    return False


# ---- shapers ----

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_PROGRESS_BAR = re.compile(r"\s*[\[(]?[=#>.\-\s]{8,}[\])]?\s*[0-9]{1,3}%")


def _strip_terminal_noise(raw: str, notes: list[str]) -> str:
    text, colors = _ANSI.subn("", raw)
    # A carriage return redraws the line: only its last state was ever visible.
    # Trailing ones are line endings (CRLF), not redraws.
    lines = [line.rstrip("\r").rsplit("\r", 1)[-1] for line in text.split("\n")]
    kept = [line for line in lines if not _PROGRESS_BAR.match(line)]
    if len(kept) < len(lines):
        notes.append(f"removed {len(lines) - len(kept)} progress-bar lines")
    if colors:
        notes.append("removed terminal color codes")
    return "\n".join(kept).strip()


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _shape_json(text: str, value: Any, notes: list[str]) -> str:
    dropped = 0

    def prune(node: Any) -> Any:
        nonlocal dropped
        if isinstance(node, list):
            return [prune(item) for item in node]
        if isinstance(node, dict):
            out = {}
            for key, item in node.items():
                if item is None or item == "" or (isinstance(item, (list, dict)) and not item):
                    dropped += 1
                else:
                    out[key] = prune(item)
            return out
        return node

    pruned = prune(value)
    if dropped:
        notes.append(f"dropped {dropped} empty JSON fields")
    # Pretty-printed or spaced JSON: the compact form is the same data in fewer tokens.
    if len(_compact(value)) < len(text):
        notes.append("compacted JSON whitespace")
    return _compact(pruned)


_HTML_NOISE = re.compile(r"<(script|style|nav|footer|header|aside|noscript|svg)[\s\S]*?</\1>", re.IGNORECASE)
# Cookie and consent banners: matched by class or id (no nesting inside them assumed).
_HTML_BANNER = re.compile(
    r"<(div|section)[^>]*\b(?:class|id)=\"[^\"]*(?:cookie|consent)[^\"]*\"[^>]*>[\s\S]*?</\1>", re.IGNORECASE
)
_HTML_BREAK = re.compile(r"<(?:br|/p|/div|/li|/h\d|/tr|/title|/article|/section)>", re.IGNORECASE)
_HTML_TAG = re.compile(r"<[^>]+>")
_HTML_ENTITIES = {"&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">"}
_HTML_ENTITY = re.compile("|".join(_HTML_ENTITIES))


def _shape_html(text: str, notes: list[str]) -> str:
    text = _HTML_NOISE.sub("", text)
    text = _HTML_BANNER.sub("", text)
    text = _HTML_BREAK.sub("\n", text)
    text = _HTML_TAG.sub("", text)
    text = _HTML_ENTITY.sub(lambda m: _HTML_ENTITIES[m.group()], text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*(?:\n\s*)+", "\n\n", text)
    notes.append("converted HTML to text (scripts, styles, navigation, banners and footer removed)")
    return text.strip()


_PASSING_TEST = re.compile(r"\s*(?:✓|✔|√|PASS\b|ok \d)", re.ASCII)


def _shape_tests(text: str, notes: list[str]) -> str:
    lines = text.split("\n")
    kept = [line for line in lines if not _PASSING_TEST.match(line) or is_error_line(line)]
    if len(kept) < len(lines):
        notes.append(f"removed {len(lines) - len(kept)} passing-test lines (summary kept)")
    return "\n".join(kept)


def _shape_listing(text: str, notes: list[str]) -> str:
    """A file listing, one line per directory, numbered series collapsed::

        src/admin/components/  Widget{0..29}.tsx
        src/auth/  middleware.ts, session.test.ts, session.ts

    Nothing is lost (every path can be rebuilt), and the model sees the whole shape of
    the tree at once instead of a slice of it.
    """
    dirs: dict[str, list[str]] = {}
    paths = [line.strip() for line in text.split("\n") if line.strip()]
    for path in paths:
        parent, slash, name = path.rpartition("/")
        dirs.setdefault(parent + slash, []).append(name)
    out = [f"{directory or './'}  {', '.join(_collapse_series(files))}" for directory, files in dirs.items()]
    notes.append(f"grouped {len(paths)} paths into {len(out)} directory lines")
    return "\n".join(out)


# The number is the last one in the name, without leading zeros: "img007.png" is
# "img00" + 7, so a series never loses its padding.
_NUMBERED = re.compile(r"(.*?)(0|[1-9][0-9]*)([^0-9]*)")


def _collapse_series(files: list[str]) -> list[str]:
    """``["Widget0.tsx", …, "Widget29.tsx"]`` → ``["Widget{0..29}.tsx"]``.

    Non-contiguous numbers stay listed.
    """
    series: dict[tuple[str, str], list[int]] = {}
    out: list[str] = []
    for name in files:
        match = _NUMBERED.fullmatch(name)
        if match:
            series.setdefault((match[1], match[3]), []).append(int(match[2]))
        else:
            out.append(name)
    for (prefix, suffix), numbers in series.items():
        numbers.sort()
        contiguous = all(b == a + 1 for a, b in itertools.pairwise(numbers))
        if len(numbers) >= 3 and contiguous:
            out.append(f"{prefix}{{{numbers[0]}..{numbers[-1]}}}{suffix}")
        else:
            out.extend(f"{prefix}{n}{suffix}" for n in numbers)
    return out


_DIGITS = re.compile(r"\d+", re.ASCII)


def _collapse_repeats(text: str, notes: list[str]) -> str:
    """Collapses runs of log lines that differ only by numbers (timestamps, ids, durations)."""
    lines = text.split("\n")
    out: list[str] = []
    collapsed = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        j = i
        if not is_error_line(line):
            pattern = _DIGITS.sub("#", line)
            while j + 1 < len(lines) and _DIGITS.sub("#", lines[j + 1]) == pattern:
                j += 1
        # The count goes on the same line: a separate "repeated … times" line would be
        # a section of its own, matched by any question about "timestamps".
        out.append(f"{line}  (×{j - i + 1})" if j > i else line)
        collapsed += j - i
        i = j + 1
    if collapsed:
        notes.append(f"collapsed {collapsed} repeated log lines")
    return "\n".join(out)
