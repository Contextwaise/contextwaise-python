"""OpenAI adapters: you run the loop, Contextwaise shrinks what goes back. Both APIs:

    # Chat Completions
    message = completion.choices[0].message
    messages.append(message)
    messages.extend(
        tool_messages(cw, message, run_tool, request=question, prompt_tokens=completion.usage.prompt_tokens)
    )

    # Responses
    response = client.responses.create(model=..., input=items, tools=[*my_tools, EXPAND_TOOL_RESPONSES])
    items.extend(response.output)
    items.extend(function_outputs(cw, response, run_tool, request=question))

Works with the ``openai`` package's objects or plain dicts; it isn't imported.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from ._client import AsyncContextwaise, Contextwaise
from ._core import EXPAND_TOOL as _EXPAND
from ._tool_calls import (
    RunTool,
    ToolCall,
    ToolOutcome,
    arun_tool_calls,
    field,
    parse_args,
    run_tool_calls,
    usage_tokens,
)

EXPAND_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": _EXPAND["name"],
        "description": _EXPAND["description"],
        "parameters": _EXPAND["parameters"],
    },
}
"""The ``expand`` tool for Chat Completions: add it to ``tools``."""

EXPAND_TOOL_RESPONSES: dict[str, Any] = {
    "type": "function",
    "name": _EXPAND["name"],
    "description": _EXPAND["description"],
    "parameters": _EXPAND["parameters"],
    "strict": False,  # `question` is optional
}
"""The ``expand`` tool for the Responses API."""

_Run = Callable[[str, Any], Any | Awaitable[Any]]


def _chat_calls(message: Any) -> list[ToolCall]:
    said = (field(message, "content") or "").strip()
    calls = []
    for call in field(message, "tool_calls") or []:
        function = field(call, "function")
        if function is None:
            continue
        args, error = parse_args(field(function, "arguments"))
        calls.append(ToolCall(field(call, "id"), field(function, "name"), args, said, error))
    return calls


def _tool_messages(outcomes: list[ToolOutcome]) -> list[dict[str, Any]]:
    return [{"role": "tool", "tool_call_id": o.id, "content": o.content} for o in outcomes]


def tool_messages(
    cw: Contextwaise,
    message: Any,
    run: RunTool,
    *,
    request: str | None = None,
    prompt_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Runs a Chat Completions message's tool calls and returns the ``tool`` messages to
    send back, in order: each output shrunk, ``expand`` answered, failures as "Error: …".
    ``prompt_tokens``: the completion's ``usage.prompt_tokens`` (a message doesn't carry
    it), so the dashboard can count what each expand cost."""
    return _tool_messages(run_tool_calls(cw, _chat_calls(message), run, request, "openai", prompt_tokens))


async def atool_messages(
    cw: AsyncContextwaise,
    message: Any,
    run: _Run,
    *,
    request: str | None = None,
    prompt_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """``tool_messages`` for asyncio: the calls run concurrently; ``run`` may be async."""
    calls = _chat_calls(message)
    return _tool_messages(await arun_tool_calls(cw, calls, run, request, "openai", prompt_tokens))


def _response_calls(response: Any) -> list[ToolCall]:
    calls, said = [], ""
    for item in field(response, "output") or []:
        kind = field(item, "type")
        if kind == "message":
            said += "".join(field(part, "text") or "" for part in field(item, "content") or []) + " "
        elif kind == "function_call":
            args, error = parse_args(field(item, "arguments"))
            calls.append(ToolCall(field(item, "call_id"), field(item, "name"), args, said.strip(), error))
    return calls


def _function_outputs(outcomes: list[ToolOutcome]) -> list[dict[str, Any]]:
    return [{"type": "function_call_output", "call_id": o.id, "output": o.content} for o in outcomes]


def function_outputs(
    cw: Contextwaise, response: Any, run: RunTool, *, request: str | None = None
) -> list[dict[str, Any]]:
    """Runs a Responses API response's function calls and returns the
    ``function_call_output`` items to send back, in order."""
    calls = _response_calls(response)
    return _function_outputs(
        run_tool_calls(cw, calls, run, request, "openai", usage_tokens(response, "input_tokens"))
    )


async def afunction_outputs(
    cw: AsyncContextwaise, response: Any, run: _Run, *, request: str | None = None
) -> list[dict[str, Any]]:
    """``function_outputs`` for asyncio: the calls run concurrently; ``run`` may be async."""
    calls = _response_calls(response)
    return _function_outputs(
        await arun_tool_calls(cw, calls, run, request, "openai", usage_tokens(response, "input_tokens"))
    )


__all__ = [
    "EXPAND_TOOL",
    "EXPAND_TOOL_RESPONSES",
    "afunction_outputs",
    "atool_messages",
    "function_outputs",
    "tool_messages",
]
