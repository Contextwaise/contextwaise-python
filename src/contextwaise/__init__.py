"""contextwaise: shrink agent tool results at birth (cache-safe, reversible, intent-aware)."""

import logging

from ._client import AsyncContextwaise, Contextwaise
from ._core import EXPAND_TOOL
from ._ledger import Ledger, LedgerEntry, MemoryLedger
from ._shapers import Shaped, shape
from ._types import Kind, Mode, Scorer, SectionLog, SectionReason, ShrinkResult
from ._version import __version__

# A library logs, the application decides where logs go.
logging.getLogger("contextwaise").addHandler(logging.NullHandler())

__all__ = [
    "EXPAND_TOOL",
    "AsyncContextwaise",
    "Contextwaise",
    "Kind",
    "Ledger",
    "LedgerEntry",
    "MemoryLedger",
    "Mode",
    "Scorer",
    "SectionLog",
    "SectionReason",
    "Shaped",
    "ShrinkResult",
    "__version__",
    "shape",
]
