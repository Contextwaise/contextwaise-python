"""The Anthropic and OpenAI adapters on hand-written responses (dicts and SDK-like objects)."""

from __future__ import annotations

import asyncio
import json
import re
from types import SimpleNamespace as Obj
from typing import Any

import pytest

from contextwaise import AsyncContextwaise, Contextwaise
from contextwaise import anthropic as cw_anthropic
from contextwaise import openai as cw_openai
from tests.stubs import API_KEY, BASE_URL, ApiStub

ORDERS = {
    "orders": [
        {
            "id": "ORD-1337" if i == 77 else f"ORD-{1000 + i}",
            "status": "lost_in_transit" if i == 77 else "delivered",
            "note": "x" * 200,
        }
        for i in range(150)
    ]
}


def run(name: str, args: Any) -> Any:
    if name == "get_orders":
        return ORDERS
    if name == "get_weather":
        return {"tempC": 23}
    raise RuntimeError(f"upstream down for {args.get('customer', name)}")


def connected() -> Contextwaise:
    return Contextwaise(api_key=API_KEY, base_url=BASE_URL, max_tokens=120)


def test_expand_tool_in_each_provider_format() -> None:
    assert cw_anthropic.EXPAND_TOOL["input_schema"]["required"] == ["ref"]
    assert cw_openai.EXPAND_TOOL["function"]["name"] == "expand"
    assert cw_openai.EXPAND_TOOL_RESPONSES["strict"] is False


def test_anthropic_results_in_order_shrunk_errors_marked_intent_combined(api: ApiStub) -> None:
    response = Obj(
        content=[
            Obj(type="text", text="Let me check acme's orders and the weather."),
            Obj(type="tool_use", id="toolu_1", name="get_orders", input={"customer": "acme"}),
            Obj(type="tool_use", id="toolu_2", name="get_weather", input={"city": "Lisbon"}),
            Obj(type="tool_use", id="toolu_3", name="get_invoices", input={"customer": "acme"}),
        ]
    )
    with connected() as cw:
        results = cw_anthropic.tool_results(cw, response, run, request="Where is order ORD-1337?")
    assert [r["tool_use_id"] for r in results] == ["toolu_1", "toolu_2", "toolu_3"]
    assert re.search(r"ORD-1337.*lost_in_transit", results[0]["content"])
    assert len(results[0]["content"]) < len(json.dumps(ORDERS)) / 5
    assert results[1]["content"] == '{"tempC":23}'
    assert results[2] == {
        "type": "tool_result",
        "tool_use_id": "toolu_3",
        "content": "Error: upstream down for acme",
        "is_error": True,
    }
    intent = api.score_bodies[0]["intent"]
    assert "Where is order ORD-1337?" in intent
    assert "Let me check acme's orders" in intent


def test_anthropic_expand_is_answered_by_contextwaise(api: ApiStub) -> None:
    with connected() as cw:
        [first] = cw_anthropic.tool_results(
            cw,
            {"content": [{"type": "tool_use", "id": "t1", "name": "get_orders", "input": {}}]},
            run,
            request="How many orders are delivered?",
        )
        ref = re.search(r'ref: "(cw_[^"]+)"', first["content"])
        assert ref, "the trimmed view carries a ref"

        def never(name: str, args: Any) -> Any:
            raise AssertionError("expand must not reach your tools")

        [back] = cw_anthropic.tool_results(
            cw,
            {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t2",
                        "name": "expand",
                        "input": {"ref": ref[1], "question": "lost_in_transit"},
                    }
                ],
                "usage": {
                    "input_tokens": 50,
                    "cache_creation_input_tokens": 1000,
                    "cache_read_input_tokens": 9000,
                },
            },
            never,
        )
    assert "ORD-1337" in back["content"]
    [expand] = [r for body in api.report_bodies for r in body["results"] if r["type"] == "expand"]
    assert expand["contextTokens"] == 10_050, "the step that asked for it, cached input included"


def test_no_tool_calls_nothing_to_send() -> None:
    assert (
        cw_anthropic.tool_results(Contextwaise(), {"content": [{"type": "text", "text": "Done."}]}, run) == []
    )
    assert cw_openai.tool_messages(Contextwaise(), {"content": "Done.", "tool_calls": None}, run) == []


def test_chat_completions_messages_args_parsed_bad_json_reported(api: ApiStub) -> None:
    message = Obj(
        content="Checking.",
        tool_calls=[
            Obj(
                id="call_1", type="function", function=Obj(name="get_orders", arguments='{"customer":"acme"}')
            ),
            Obj(id="call_2", type="function", function=Obj(name="get_weather", arguments="{not json")),
        ],
    )
    with connected() as cw:
        out = cw_openai.tool_messages(cw, message, run, request="Status of ORD-1337?")
    assert [(m["role"], m["tool_call_id"]) for m in out] == [("tool", "call_1"), ("tool", "call_2")]
    assert "ORD-1337" in out[0]["content"]
    assert out[1]["content"] == "Error: the tool arguments were not valid JSON"


def test_responses_api_function_outputs_with_message_text_as_intent(api: ApiStub) -> None:
    response = {
        "output": [
            {"type": "reasoning"},
            {"type": "message", "content": [{"type": "output_text", "text": "I'll look up the order."}]},
            {
                "type": "function_call",
                "call_id": "fc_1",
                "name": "get_orders",
                "arguments": '{"customer":"acme"}',
            },
        ]
    }
    with connected() as cw:
        out = cw_openai.function_outputs(cw, response, run, request="Where is ORD-1337?")
    assert out[0]["type"] == "function_call_output"
    assert out[0]["call_id"] == "fc_1"
    assert "ORD-1337" in out[0]["output"]
    assert "I'll look up the order." in api.score_bodies[0]["intent"]


@pytest.mark.parametrize("adapter", ["anthropic", "chat", "responses"])
def test_async_adapters_run_async_tools_concurrently(api: ApiStub, adapter: str) -> None:
    running = 0
    peak = 0

    async def arun(name: str, args: Any) -> Any:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.05)
        running -= 1
        return run(name, args)

    async def go() -> list[str]:
        async with AsyncContextwaise(api_key=API_KEY, base_url=BASE_URL, max_tokens=120) as cw:
            if adapter == "anthropic":
                blocks = [
                    {"type": "tool_use", "id": f"a{i}", "name": "get_orders", "input": {}} for i in range(3)
                ]
                return [
                    r["content"]
                    for r in await cw_anthropic.atool_results(
                        cw, {"content": blocks}, arun, request="ORD-1337"
                    )
                ]
            if adapter == "chat":
                calls = [
                    {"id": f"c{i}", "function": {"name": "get_orders", "arguments": "{}"}} for i in range(3)
                ]
                return [
                    m["content"]
                    for m in await cw_openai.atool_messages(
                        cw, {"content": None, "tool_calls": calls}, arun, request="ORD-1337"
                    )
                ]
            items = [
                {"type": "function_call", "call_id": f"r{i}", "name": "get_orders", "arguments": "{}"}
                for i in range(3)
            ]
            return [
                o["output"]
                for o in await cw_openai.afunction_outputs(cw, {"output": items}, arun, request="ORD-1337")
            ]

    contents = asyncio.run(go())
    assert len(contents) == 3
    assert all("ORD-1337" in c for c in contents)
    assert peak == 3, "the three tools ran at the same time"


def test_reports_name_the_adapter(api: ApiStub) -> None:
    with connected() as cw:
        cw_openai.tool_messages(
            cw, {"tool_calls": [{"id": "x1", "function": {"name": "get_orders", "arguments": "{}"}}]}, run
        )
        cw_anthropic.tool_results(
            cw, {"content": [{"type": "tool_use", "id": "x2", "name": "get_orders", "input": {}}]}, run
        )
    integrations = {r["callId"]: r["integration"] for body in api.report_bodies for r in body["results"]}
    assert integrations == {"x1": "openai", "x2": "anthropic"}
