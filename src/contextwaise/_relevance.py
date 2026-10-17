"""Word overlap between an intent and each section.

Never decides what is kept: it can't judge meaning (synonyms, other languages). It
only orders lines inside a group the API already judged relevant, so the matching
lines come first rather than whichever happen to be at the top. Any language: words
are split on Unicode letters and digits, and a word that doesn't match simply doesn't
count.

Port of ``core/src/relevance.ts`` (``@contextwaise/core``): keep the two in step.
"""

from __future__ import annotations

import math
import re

_STOPWORDS = frozenset(
    "a an and are as at be by do does for from how i if in is it of on or so that the this to was we what "
    "when where which who why with you".split()
)
_WORD = re.compile(r"[^\W_]+")  # runs of Unicode letters and digits
_ENDING = re.compile(r"(?:ing|ed|es|s)$")

# A word found in more than this share of sections is too common to signal relevance.
_COMMON_WORD_SHARE = 0.6
# Below this many sections, word frequencies say nothing reliable.
_MIN_SECTIONS_FOR_FREQUENCY = 5


def _stem(word: str) -> str:
    """Rough stem: drop a plural or verb ending, keep 5 letters ("tests"/"test", "failing"/"fail")."""
    return (_ENDING.sub("", word) if len(word) > 4 else word)[:5]


def _words(text: str) -> set[str]:
    """The stemmed content words of a text."""
    return {_stem(w) for w in _WORD.findall(text.lower()) if len(w) > 1 and w not in _STOPWORDS}


def word_overlap(query: str, sections: list[str]) -> list[float]:
    """For each section, the share of the query's words it contains, 0 to 1.

    Square root: matching a few words of a long query is already a strong signal.
    Words present in most sections ("src" in a file listing, "INFO" in a log) can't
    tell sections apart, so they don't count: only the rarer words decide.
    """
    section_words = [_words(s) for s in sections]
    query_words = _words(query)
    if len(sections) >= _MIN_SECTIONS_FOR_FREQUENCY:
        query_words = {
            w for w in query_words if sum(w in s for s in section_words) / len(sections) <= _COMMON_WORD_SHARE
        }
    if not query_words:
        return [0.0] * len(sections)
    return [math.sqrt(len(query_words & s) / len(query_words)) for s in section_words]
