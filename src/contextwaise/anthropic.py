"""Anthropic Messages API adapter: you run the loop, Contextwaise shrinks what goes back.

    response = client.messages.create(model=..., messages=messages, tools=[*my_tools, EXPAND_TOOL])
    messages.append({"role": "assistant", "content": response.content})
    messages.append({"role": "user", "content": tool_results(cw, response, run_tool, request=question)})

Works with the ``anthropic`` package's objects or plain dicts; it isn't imported.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from ._client import AsyncContextwaise, Contextwaise
from ._core import EXPAND_TOOL as _EXPAND
from ._tool_calls import RunTool, ToolCall, ToolOutcome, arun_tool_calls, field, run_tool_calls, usage_tokens

EXPAND_TOOL: dict[str, Any] = {
    "name": _EXPAND["name"],
    "description": _EXPAND["description"],
    "input_schema": _EXPAND["parameters"],
}
"""The ``expand`` tool in Anthropic's format: add it to ``tools``."""


def _calls(response: Any) -> list[ToolCall]:
    calls, said = [], ""
    for block in field(response, "content") or []:
        kind = field(block, "type")
        if kind == "text":
            said += f"{field(block, 'text') or ''} "
        elif kind == "tool_use":
            calls.append(
                ToolCall(field(block, "id"), field(block, "name"), field(block, "input") or {}, said.strip())
            )
    return calls


def _input_tokens(response: Any) -> int | None:
    # Anthropic counts cached input apart from the rest: the call read all three.
    return usage_tokens(response, "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


def _blocks(outcomes: list[ToolOutcome]) -> list[dict[str, Any]]:
    return [
        {
            "type": "tool_result",
            "tool_use_id": o.id,
            "content": o.content,
            **({"is_error": True} if o.is_error else {}),
        }
        for o in outcomes
    ]


def tool_results(
    cw: Contextwaise, response: Any, run: RunTool, *, request: str | None = None
) -> list[dict[str, Any]]:
    """Runs the response's tool calls and returns their ``tool_result`` blocks, in order:
    each output shrunk, ``expand`` answered, failures marked ``is_error``. Empty when the
    model called no tool. ``request``: the user's request, which sharpens relevance."""
    return _blocks(run_tool_calls(cw, _calls(response), run, request, "anthropic", _input_tokens(response)))


async def atool_results(
    cw: AsyncContextwaise,
    response: Any,
    run: Callable[[str, Any], Any | Awaitable[Any]],
    *,
    request: str | None = None,
) -> list[dict[str, Any]]:
    """``tool_results`` for asyncio: the calls run concurrently; ``run`` may be async."""
    return _blocks(
        await arun_tool_calls(cw, _calls(response), run, request, "anthropic", _input_tokens(response))
    )


__all__ = ["EXPAND_TOOL", "atool_results", "tool_results"]
