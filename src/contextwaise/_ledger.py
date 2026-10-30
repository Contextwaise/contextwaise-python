"""Where original outputs and their views live, so a view can be replayed and expanded."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from ._types import ShrinkResult

K = TypeVar("K")
V = TypeVar("V")


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    call_id: str
    tool: str
    original: str
    result: ShrinkResult
    agent: str | None = None


class Ledger(Protocol):
    """Storage for ledger entries, keyed by ref (one per tool call id).

    The default keeps entries in memory. For agents spread over several processes,
    implement these two methods over Redis or a database: a ref handed to the model
    in one process can then be expanded in another.
    """

    def get(self, key: str) -> LedgerEntry | None: ...

    def set(self, key: str, entry: LedgerEntry) -> None: ...


class BoundedDict(Generic[K, V]):
    """A thread-safe dict that forgets the least recently written key past ``max_size``.

    Keeps per-process state bounded, so a long-running server doesn't grow forever.
    """

    def __init__(self, max_size: int) -> None:
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        self._max_size = max_size
        self._items: OrderedDict[K, V] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: K) -> V | None:
        with self._lock:
            return self._items.get(key)

    def set(self, key: K, value: V) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            if len(self._items) > self._max_size:
                self._items.popitem(last=False)

    def pop(self, key: K) -> V | None:
        with self._lock:
            return self._items.pop(key, None)


class MemoryLedger(BoundedDict[str, LedgerEntry]):
    """An in-memory ledger keeping the most recent ``max_entries`` tool calls."""

    def __init__(self, max_entries: int = 5000) -> None:
        super().__init__(max_entries)
