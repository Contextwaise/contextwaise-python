"""What every "you run the loop" adapter shares (``anthropic``, ``openai``): tool calls in,
what to send back out. The provider modules only read their response shape and format
their result type; the decisions live here, once. Port of ``sdk/typescript/src/tool-calls.ts``.

Responses are read by shape, never imported: an SDK object or a plain dict both work.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ._client import AsyncContextwaise, Contextwaise
from ._core import EXPAND_TOOL

RunTool = Callable[[str, Any], Any]
"""Runs one of your tools: ``run(name, args)`` returns its raw output (text or any JSON
value). Raise to report a failure to the model. For the async adapters it may be async."""


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool call, whatever the provider: ``text_before`` is what the model wrote before it."""

    id: str
    name: str
    args: Any
    text_before: str
    args_error: str | None = None


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    id: str
    content: str
    is_error: bool


def field(obj: Any, name: str) -> Any:
    """A field of an SDK object or of a dict."""
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def parse_args(text: Any) -> tuple[Any, str | None]:
    """Arguments sent as JSON text (OpenAI): parsed, or the reason they couldn't be.
    Non-text arguments pass as they are."""
    if text is None or text == "":
        return {}, None
    if not isinstance(text, str):
        return text, None
    try:
        return json.loads(text), None
    except ValueError:
        return text, "the tool arguments were not valid JSON"


def usage_tokens(response: Any, *names: str) -> int | None:
    """The sum of the response's ``usage`` fields (the provider's own count), or None without usage."""
    usage = field(response, "usage")
    if usage is None:
        return None
    return sum(field(usage, name) or 0 for name in names)


def _intent(call: ToolCall, request: str | None) -> str:
    return " ".join(part for part in (request, call.text_before) if part)


def _expand_args(call: ToolCall) -> tuple[str, str | None]:
    args = call.args if isinstance(call.args, dict) else {}
    question = args.get("question")
    return str(args.get("ref") or ""), question if isinstance(question, str) else None


def _error(call: ToolCall, message: str) -> ToolOutcome:
    return ToolOutcome(call.id, f"Error: {message}", True)


def run_tool_calls(
    cw: Contextwaise,
    calls: list[ToolCall],
    run: RunTool,
    request: str | None,
    integration: str,
    context_tokens: int | None,
    agent: str | None = None,
) -> list[ToolOutcome]:
    """Runs the calls in order and returns what the model should read for each: ``expand``
    answered by Contextwaise, other tools by ``run`` with their output shrunk, and a tool
    that raises (or arguments that don't parse) as an error the model can see.
    ``context_tokens``: the input tokens of the model call that made them (see ``expand``)."""
    outcomes = []
    for call in calls:
        if call.args_error:
            outcomes.append(_error(call, call.args_error))
        elif call.name == EXPAND_TOOL["name"]:
            answer = cw.expand(*_expand_args(call), context_tokens=context_tokens)
            outcomes.append(ToolOutcome(call.id, answer, False))
        else:
            cw.note_intent(call.id, _intent(call, request))
            try:
                output = run(call.name, call.args)
            except Exception as error:  # the model should see the failure, not the agent crash
                outcomes.append(_error(call, str(error)))
                continue
            view = cw.shrink(
                call_id=call.id,
                tool=call.name,
                args=call.args,
                output=output,
                integration=integration,
                agent=agent,
            )
            outcomes.append(ToolOutcome(call.id, view.text, False))
    return outcomes


async def arun_tool_calls(
    cw: AsyncContextwaise,
    calls: list[ToolCall],
    run: Callable[[str, Any], Any | Awaitable[Any]],
    request: str | None,
    integration: str,
    context_tokens: int | None,
    agent: str | None = None,
) -> list[ToolOutcome]:
    """``run_tool_calls`` for asyncio: the calls run concurrently; ``run`` may be sync or async."""

    async def one(call: ToolCall) -> ToolOutcome:
        if call.args_error:
            return _error(call, call.args_error)
        if call.name == EXPAND_TOOL["name"]:
            return ToolOutcome(
                call.id, await cw.expand(*_expand_args(call), context_tokens=context_tokens), False
            )
        cw.note_intent(call.id, _intent(call, request))
        try:
            output = run(call.name, call.args)
            if inspect.isawaitable(output):
                output = await output
        except Exception as error:  # the model should see the failure, not the agent crash
            return _error(call, str(error))
        view = await cw.shrink(
            call_id=call.id,
            tool=call.name,
            args=call.args,
            output=output,
            integration=integration,
            agent=agent,
        )
        return ToolOutcome(call.id, view.text, False)

    return list(await asyncio.gather(*(one(call) for call in calls)))
