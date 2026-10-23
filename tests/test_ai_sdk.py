"""The Vercel AI SDK for Python adapter on a scripted model (``ai.testing.FakeModel``). No network."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

pytest.importorskip("ai")  # the `ai` package needs Python 3.12+

import ai
from ai.testing import FakeModel, tool_call

from contextwaise import AsyncContextwaise
from contextwaise._core import _ref
from contextwaise.ai_sdk import ContextwaiseAgent
from tests.stubs import API_KEY, BASE_URL, ApiStub

ORDERS = json.dumps(
    [
        {
            "id": "ORD-1337" if i == 77 else f"ORD-{1000 + i}",
            "status": "lost_in_transit" if i == 77 else "delivered",
            "carrier": "Hermes" if i == 77 else "DHL",
            "note": "x" * 200,
        }
        for i in range(150)
    ]
)
QUESTION = "What is the status of order ORD-1337?"


@ai.tool
async def get_orders(customer: str) -> str:
    """Lists a customer's orders."""
    return ORDERS


@ai.tool
async def flaky(id: str) -> str:
    """Looks something up, and fails."""
    raise RuntimeError("upstream unavailable")


def tool_parts(messages: list[Any]) -> dict[str, Any]:
    return {p.tool_name: p for m in messages if m.role == "tool" for p in m.tool_results}


def run(script: list[Any], tools: list[Any], **options: Any) -> list[Any]:
    async def go() -> list[Any]:
        async with AsyncContextwaise(api_key=API_KEY, base_url=BASE_URL, max_tokens=120, **options) as cw:
            agent = ContextwaiseAgent(cw, tools=tools)
            async with agent.run(FakeModel(script), [script[0]]) as stream:
                async for _ in stream:
                    pass
            await cw.flush()
            return list(stream.messages)

    return asyncio.run(go())


def test_model_reads_the_view_and_the_app_keeps_the_raw_output(api: ApiStub) -> None:
    call = tool_call(get_orders, customer="acme")
    script = [
        ai.user_message(QUESTION),
        ai.assistant_message("Let me look up the orders.", call),
        ai.assistant_message("Lost in transit."),
    ]
    part = tool_parts(run(script, [get_orders]))["get_orders"]
    view = part.get_model_input()
    assert part.result == ORDERS, "the app keeps the raw output"
    assert "ORD-1337" in view
    assert "omitted" in view
    assert len(view) < len(ORDERS) / 4
    [score] = api.score_bodies
    assert "ORD-1337" in score["intent"], "the user's request is the intent"
    assert "Let me look up the orders." in score["intent"], "plus what the model said before the call"
    [result] = [r for body in api.report_bodies for r in body["results"]]
    assert result["integration"] == "vercel-ai"


def test_expand_brings_back_what_was_trimmed(api: ApiStub) -> None:
    orders = tool_call(get_orders, customer="acme")
    expand = tool_call("expand", ref=_ref(orders.tool_call_id), question="Which carrier ships ORD-1337?")
    script = [
        ai.user_message("What happened to order ORD-1001?"),
        ai.assistant_message(orders),
        ai.assistant_message(expand),
        ai.assistant_message("Done."),
    ]
    parts = tool_parts(run(script, [get_orders]))
    view = parts["get_orders"].get_model_input()
    assert '"carrier":"Hermes"' not in view, "the order was trimmed away at first"
    assert "carrier: DHL ×148, Hermes ×1" in view, "the note counts what was cut"
    assert "Hermes" in parts["expand"].get_model_input(), "expand brought it back, unshrunk"
    [expand] = [r for body in api.report_bodies for r in body["results"] if r["type"] == "expand"]
    assert len(view) // 4 < expand["contextTokens"] < len(ORDERS) // 4, (
        "the step that asked for it: the history as the model read it, not the raw results"
    )


def test_tool_errors_reach_the_model_untouched(api: ApiStub) -> None:
    script = [
        ai.user_message("look up x"),
        ai.assistant_message(tool_call(flaky, id="x")),
        ai.assistant_message("It failed."),
    ]
    part = tool_parts(run(script, [flaky]))["flaky"]
    assert part.is_error
    assert "upstream unavailable" in str(part.get_model_input())
    assert api.score_bodies == []
