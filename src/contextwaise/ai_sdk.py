"""Vercel AI SDK for Python (``ai``) adapter: an ``ai.Agent`` whose model reads
Contextwaise's view of each tool result.

    agent = ContextwaiseAgent(cw, tools=[get_orders, search_logs])
    async with agent.run(model, messages) as stream:
        async for event in stream: ...

- The loop is ``ai.Agent``'s default one, plus two things: it records each tool call's
  intent (the user's latest request and what the model wrote before the call), and it
  sets each tool result's ``model_input`` to Contextwaise's view. ``result`` stays raw,
  so the app (events, UI) keeps the full output; the model reads the view, and the
  history (``stream.messages``) carries it, so later turns resend the same bytes
  (cache-safe).
- An ``expand`` tool is added so the model can get any trimmed part back. Each expand
  reports the size of the step that asked for it: the extra step it causes costs about
  as much, and the dashboard counts that against the savings.
- Errors, ``expand``'s own answers and media (images, files) reach the model untouched.

Needs ``pip install "contextwaise[ai]"`` (the ``ai`` package, Python 3.12+), and an
``AsyncContextwaise``: the agent runs on asyncio.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Sequence
from typing import Any

import ai
from ai.types.messages import ContentOutput, Message, Part, ToolCallPart, ToolResultPart

from ._client import AsyncContextwaise
from ._core import EXPAND_TOOL, _to_json_text
from ._shapers import tokens

INTEGRATION = "vercel-ai"  # same label as the TypeScript adapter


def expand_tool(cw: AsyncContextwaise) -> ai.AgentTool:
    """The ``expand`` tool, answered by ``cw``. ``ContextwaiseAgent`` adds it for you."""
    return _expand_tool(cw, {})


def _expand_tool(cw: AsyncContextwaise, step_tokens: dict[str, int]) -> ai.AgentTool:
    """``expand``, reporting the size of the step that asked for it (``step_tokens``, by tool call id)."""

    async def expand(ref: str, question: str | None = None) -> str:
        context_tokens = step_tokens.pop(ai.current_tool_call().id, None)
        return await cw.expand(ref, question, context_tokens=context_tokens)

    expand.__doc__ = EXPAND_TOOL["description"]
    return ai.tool(expand)


class ContextwaiseAgent(ai.Agent):
    """An ``ai.Agent`` whose model reads Contextwaise's view of each tool result.

    Use it like ``ai.Agent(tools=...)``, with the client first. Subclass it and override
    ``loop`` only if you also replace the default loop: the shrinking lives there.
    """

    def __init__(self, cw: AsyncContextwaise, *, tools: Sequence[ai.AgentTool]) -> None:
        # Tool call id → size of the step that asked for that expand, until it runs.
        self._step_tokens: dict[str, int] = {}
        super().__init__(tools=[*tools, _expand_tool(cw, self._step_tokens)])
        self._cw = cw

    async def loop(self, context: ai.Context) -> AsyncGenerator[Any]:
        while context.keep_running():
            said = _last_user_text(context.messages)
            args: dict[str, Any] = {}
            async with ai.stream(context=context) as stream, ai.ToolRunner() as runner:
                async for event in ai.util.merge(stream, runner.events()):
                    yield event
                    if isinstance(event, ai.events.TextDelta):
                        said += event.chunk
                    elif isinstance(event, ai.events.ToolEnd):
                        call = event.tool_call
                        self._cw.note_intent(call.tool_call_id, said)
                        if call.tool_name == EXPAND_TOOL["name"]:
                            # The step's input, estimated: the provider's count only
                            # arrives after the tools have started running.
                            self._step_tokens[call.tool_call_id] = _model_tokens(context.messages)
                        args[call.tool_call_id] = json.loads(call.tool_args) if call.tool_args else {}
                        runner.schedule(self.resolve(call))
                context.add(stream.message)
                results = runner.get_tool_message()
                if results is not None:
                    await asyncio.gather(*(self._shrink(p, args) for p in results.tool_results))
                context.add(results)

    async def _shrink(self, part: ToolResultPart, args: dict[str, Any]) -> None:
        output = part.get_model_input()
        if part.is_error or part.tool_name == EXPAND_TOOL["name"] or isinstance(output, ContentOutput):
            return
        view = await self._cw.shrink(
            call_id=part.tool_call_id,
            tool=part.tool_name,
            args=args.get(part.tool_call_id),
            output=output,
            integration=INTEGRATION,
        )
        part.set_model_input(view.text)


def _model_tokens(messages: Sequence[Message]) -> int:
    """About what the model reads of ``messages``: the views of tool results, never the raw
    results the history also keeps for the app."""
    return sum(tokens(_model_text(part)) for message in messages for part in message.parts)


def _model_text(part: Part) -> str:
    if isinstance(part, ToolResultPart):
        output = part.get_model_input()
        return output if isinstance(output, str) else _to_json_text(output)
    if isinstance(part, ToolCallPart):
        return part.tool_args
    return getattr(part, "text", "")


def _last_user_text(messages: Sequence[Message]) -> str:
    """The user's latest request: what the whole agent turn is for."""
    for message in reversed(messages):
        if message.role == "user" and message.text:
            return message.text + " "
    return ""


__all__ = ["ContextwaiseAgent", "expand_tool"]
