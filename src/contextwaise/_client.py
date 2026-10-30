"""The two clients: same decisions (``_core``), sync or async I/O."""

from __future__ import annotations

from types import TracebackType
from typing import Any

from ._api import ScoreRequest
from ._core import BaseContextwaise, Scores, arun_steps, run_steps
from ._types import ShrinkResult


class Contextwaise(BaseContextwaise):
    """Shrinks agent tool results before the model reads them: cache-safe, reversible, intent-aware.

    Thread-safe. Use one per agent (or per process) and close it when done, or use it
    as a context manager, so pending dashboard reports are sent::

        with Contextwaise(agent="support-triage") as cw:
            view = cw.shrink(call_id=call.id, tool=call.name, args=call.input, output=output)
    """

    def shrink(
        self,
        *,
        call_id: str,
        tool: str,
        output: Any,
        args: Any = None,
        integration: str = "python",
        model: str | None = None,
        full: bool = False,
        agent: str | None = None,
    ) -> ShrinkResult:
        """The model's view of a tool result. Same ``call_id``, same view, always.

        Args:
            call_id: The tool call's id, as the model provider gave it.
            tool: The tool's name.
            output: What the tool returned: text, or anything JSON-serializable
                (Pydantic models and dataclasses are serialized by their fields).
            args: The arguments the model called the tool with (a dict, or JSON text).
            integration: Which adapter called, shown on the dashboard. Adapters set it.
            model: The model id that will read the result (e.g. ``"claude-sonnet-4-5"``), shown
                with the agent on the dashboard.
            full: This call's output must reach the model whole (e.g. a file the agent will
                edit), like the tools in the ``full`` option. Schemas and lists of actions are
                detected and passed whole on their own.
            agent: The agent this call belongs to, overriding the ``agent`` option. For apps
                hosting many agents on one client.

        Returns:
            The view; send ``result.text`` to the model as the tool result.
        """
        return run_steps(
            self._shrink_steps(call_id, tool, args, output, integration, model, full, agent), self._score
        )

    def expand(self, ref: str, question: str | None = None, *, context_tokens: int | None = None) -> str:
        """The original output behind a ref, or its parts answering ``question``.

        This is what the ``expand`` tool (``EXPAND_TOOL``) returns to the model.

        Args:
            context_tokens: The input tokens of the model call that asked for this expand
                (the provider's ``usage``). Expanding costs one more model call about that
                size: the dashboard counts it against the savings. Adapters pass it for you.
        """
        return run_steps(self._expand_steps(ref, question, context_tokens), self._score)

    def flush(self) -> None:
        """Sends pending dashboard reports now. Call it before a short-lived process exits."""
        if self._api:
            self._api.flush()

    def close(self) -> None:
        """Sends pending dashboard reports and releases connections."""
        if self._api:
            self._api.close()

    def __enter__(self) -> Contextwaise:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()

    def _score(self, request: ScoreRequest) -> Scores:
        return self._api.score(request) if self._api else None


class AsyncContextwaise(BaseContextwaise):
    """``Contextwaise`` for asyncio agents: the relevance API never blocks the event loop.

    ::

        async with AsyncContextwaise(agent="support-triage") as cw:
            view = await cw.shrink(call_id=call.id, tool=call.name, args=call.input, output=output)
    """

    async def shrink(
        self,
        *,
        call_id: str,
        tool: str,
        output: Any,
        args: Any = None,
        integration: str = "python",
        model: str | None = None,
        full: bool = False,
        agent: str | None = None,
    ) -> ShrinkResult:
        """The model's view of a tool result. Same ``call_id``, same view, always.

        See ``Contextwaise.shrink``.
        """
        return await arun_steps(
            self._shrink_steps(call_id, tool, args, output, integration, model, full, agent), self._score
        )

    async def expand(
        self, ref: str, question: str | None = None, *, context_tokens: int | None = None
    ) -> str:
        """The original output behind a ref, or its parts answering ``question``.

        See ``Contextwaise.expand``.
        """
        return await arun_steps(self._expand_steps(ref, question, context_tokens), self._score)

    async def flush(self) -> None:
        """Sends pending dashboard reports now. Await it before a short-lived process exits."""
        if self._api:
            await self._api.aflush()

    async def aclose(self) -> None:
        """Sends pending dashboard reports and releases connections."""
        if self._api:
            await self._api.aclose()

    async def __aenter__(self) -> AsyncContextwaise:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self.aclose()

    async def _score(self, request: ScoreRequest) -> Scores:
        return await self._api.ascore(request) if self._api else None
