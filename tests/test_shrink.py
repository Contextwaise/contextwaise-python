from __future__ import annotations

import asyncio
import json
import re

import pytest

from contextwaise import AsyncContextwaise, Contextwaise
from tests.stubs import API_KEY, BASE_URL, TEST_RUN, ApiStub


def connected(**options: object) -> Contextwaise:
    return Contextwaise(api_key=API_KEY, base_url=BASE_URL, **options)  # type: ignore[arg-type]


LISTING = "\n".join([*(f"src/feature{i}/index.ts" for i in range(200)), "src/auth/middleware.ts"])


def test_trims_to_the_parts_matching_the_intent_and_replays_the_same_view(api: ApiStub) -> None:
    with connected(max_tokens=120) as cw:
        cw.note_intent("call_1", "Let me find the auth middleware.")
        result = cw.shrink(call_id="call_1", tool="list_files", args={"dir": "."}, output=LISTING)
        assert result.mode == "trimmed"
        assert result.scorer == "jev"
        assert re.search(r"src/auth/\s+middleware\.ts", result.text)
        assert result.tokens_after < result.tokens_before / 5

        again = cw.shrink(call_id="call_1", tool="list_files", args={"dir": "."}, output="anything")
        assert again.text == result.text, "same call id → byte-identical view"
        assert re.search(r"src/feature42/\s+index\.ts", cw.expand(result.ref, "feature42"))


def test_each_item_of_an_api_envelope_is_its_own_part(api: ApiStub) -> None:
    orders = [
        {
            "id": "ORD-1337" if i == 117 else f"ORD-{1000 + i}",
            "status": "lost_in_transit" if i == 117 else "delivered",
            "note": "x" * 300,
        }
        for i in range(180)
    ]
    with connected(max_tokens=300) as cw:
        cw.note_intent("call_env", "What is the status of ORD-1337?")
        result = cw.shrink(
            call_id="call_env", tool="get_orders", args={}, output={"page": 1, "orders": orders}
        )
    assert result.mode == "trimmed"
    assert '"orders[117]":{"id":"ORD-1337","status":"lost_in_transit"' in result.text
    assert result.tokens_after < result.tokens_before / 5


def test_ids_the_intent_asks_about_reach_the_scorer_unmasked(api: ApiStub) -> None:
    rows = "\n".join(
        f"order ORD-{1000 + i * 2 + 1} shipped by carrier {i % 3}, reference 2026{i}" for i in range(300)
    )
    with connected(max_tokens=200) as cw:
        cw.note_intent("call_id", "Where is order ORD-1337?")
        result = cw.shrink(call_id="call_id", tool="search", args={}, output=rows)
    sent = [s for body in api.score_bodies for s in body["sections"]]
    assert any("ORD-1337" in s for s in sent), "the asked id is sent as is"
    assert not any("ORD-1339" in s for s in sent), "other ids are masked"
    assert "ORD-1337" in result.text


def test_small_outputs_and_full_tools_pass_untouched() -> None:
    cw = Contextwaise(full=["read_file"])
    assert cw.shrink(call_id="a", tool="get_time", output="10:42").text == "10:42"
    big = "line\n" * 5000
    assert cw.shrink(call_id="b", tool="read_file", args={"path": "x"}, output=big).text == big


def test_equally_relevant_parts_are_kept_whole_instead_of_cut_arbitrarily(api: ApiStub) -> None:
    orders: list[dict[str, object]] = [
        {"id": f"ord_{i}", "customer": "acme", "total": 10 + i, "note": None, "tags": []} for i in range(20)
    ]
    with connected(max_tokens=200) as cw:
        result = cw.shrink(call_id="o1", tool="get_orders", args={"customer": "acme"}, output=orders)
    assert result.mode == "shaped"
    assert result.text.count('"id":"ord_') == 20, "no order dropped"
    assert "null" not in result.text, "empty fields still removed"


def test_a_view_that_only_lost_noise_does_not_invite_expand() -> None:
    result = Contextwaise().shrink(call_id="t9", tool="run_tests", output=TEST_RUN)
    assert result.mode == "shaped"
    assert "expand" not in result.text
    assert "Everything else is shown in full" in result.text


def test_inside_a_relevant_group_the_lines_matching_the_intent_come_first(api: ApiStub) -> None:
    listing = "\n".join(
        [
            "src/admin/index.ts",
            *(f"src/feature{i}/service.ts" for i in range(300)),
            "src/auth/middleware.ts",
            "src/auth/session.ts",
        ]
    )
    with connected(max_tokens=120) as cw:
        cw.note_intent("l1", "Where is the auth middleware? ")
        result = cw.shrink(call_id="l1", tool="list_files", args={"dir": "src"}, output=listing)
    assert re.search(r"src/auth/\s+middleware\.ts", result.text)


def test_per_tool_budgets(api: ApiStub) -> None:
    with connected(budgets={"read_file": 1500}) as cw:
        small = "export const a = 1;\n" * 100
        assert cw.shrink(call_id="f1", tool="read_file", args={"path": "a.ts"}, output=small).text == small
        big = "\n\n".join(f"export const value{i} = compute({i});" for i in range(3000))
        assert cw.shrink(call_id="f2", tool="read_file", args={"path": "b.ts"}, output=big).mode == "trimmed"


def test_a_view_is_never_bigger_than_the_original(api: ApiStub) -> None:
    text = "\n\n".join(f"value {i} is computed here" for i in range(40))
    with connected(max_tokens=50) as cw:
        result = cw.shrink(call_id="n1", tool="read_file", args={"path": "x.ts"}, output=text)
    assert result.tokens_after <= result.tokens_before


def test_without_the_api_nothing_is_cut(api: ApiStub) -> None:
    api.down = True
    for cw in (Contextwaise(max_tokens=60), connected(max_tokens=60)):
        with cw:
            result = cw.shrink(call_id="w1", tool="list_files", args={"dir": "."}, output=LISTING)
        assert result.mode != "trimmed"
        assert result.scorer == "none"
        assert "feature199" in result.text


def test_openai_style_json_arguments_count_as_intent(api: ApiStub) -> None:
    with connected(max_tokens=120) as cw:
        cw.shrink(call_id="j1", tool="list_files", args=json.dumps({"query": "middleware"}), output=LISTING)
    assert api.score_bodies[0]["intent"] == "list files middleware"


def test_python_objects_are_serialized_by_their_fields() -> None:
    from dataclasses import dataclass

    @dataclass
    class Order:
        id: str
        note: None = None

    result = Contextwaise().shrink(call_id="d1", tool="get_order", output=[Order("ORD-1")])
    assert '"id":"ORD-1"' in result.text


def test_an_unknown_ref_expands_to_a_clear_message() -> None:
    assert Contextwaise().expand("cw_nope") == "No output found for cw_nope."


def test_async_client_makes_the_same_decisions(api: ApiStub) -> None:
    async def run() -> tuple[str, str]:
        async with AsyncContextwaise(api_key=API_KEY, base_url=BASE_URL, max_tokens=120) as cw:
            cw.note_intent("call_1", "Let me find the auth middleware.")
            result = await cw.shrink(call_id="call_1", tool="list_files", args={"dir": "."}, output=LISTING)
            return result.text, await cw.expand(result.ref, "feature42")

    text, expanded = asyncio.run(run())
    with connected(max_tokens=120) as cw:
        cw.note_intent("call_1", "Let me find the auth middleware.")
        assert cw.shrink(call_id="call_1", tool="list_files", args={"dir": "."}, output=LISTING).text == text
    assert "feature42" in expanded


def test_an_api_key_without_an_address_is_a_clear_error() -> None:
    with pytest.raises(ValueError, match="base_url"):
        Contextwaise(api_key="k")
    with pytest.raises(ValueError, match="base_url"):
        Contextwaise(api_key="k", base_url="app.test")
    Contextwaise()  # no key: cleanup only


def test_budgets_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        Contextwaise(budgets={"read_file": 0})


def test_json_outputs_are_measured_compact_as_providers_send_them() -> None:
    output = {"orders": [{"id": "a", "total": 1}, {"id": "b", "total": 2}]}
    result = Contextwaise().shrink(call_id="j2", tool="get_orders", output=output)
    assert result.tokens_before == (len(json.dumps(output, separators=(",", ":"))) + 3) // 4


SCHEMA = "\n".join(
    [
        *(
            f"TABLE app_events_{i} (id text, kind text, value jsonb, created_at timestamptz)"
            for i in range(70)
        ),
        "TABLE billing_refunds (id text, charge_id text, amount_cents int, reason text)",
        *(f"TABLE ops_stats_{i} (id text, metric text, value int)" for i in range(30)),
    ]
)


def test_line_oriented_text_is_cut_by_line_not_dropped_whole(api: ApiStub) -> None:
    """A schema dump, CSV or query result has no blank lines: one line per unit, or the
    whole output is a single part, kept or dropped as one."""
    with connected(max_tokens=200) as cw:
        cw.note_intent("call_schema", "Which table holds refunds? Let me look at the schema.")
        result = cw.shrink(call_id="call_schema", tool="describe_schema", args={}, output=SCHEMA)
        assert result.mode == "trimmed"
        assert result.kind == "lines"
        assert "TABLE billing_refunds (id text" in result.text
        assert re.search(r"… lines \d+-\d+ omitted", result.text), "the note says which lines are missing"
        assert "\n\n" not in result.text.split("\n[contextwaise:")[0], "rows stay one per line"
        assert result.tokens_after < result.tokens_before / 5
        assert "TABLE ops_stats_7 (" in cw.expand(result.ref, "ops_stats_7")


def test_a_cut_that_saves_too_little_is_not_made_and_costs_no_scoring(api: ApiStub) -> None:
    # About 1,300 tokens over a 600 budget: cutting would save ~700, less than one expand can cost.
    medium = "\n".join(f"src/feature{i}/index.ts" for i in range(220))
    with connected(max_tokens=600) as cw:
        cw.note_intent("m1", "Let me find feature 7.")
        result = cw.shrink(call_id="m1", tool="list_files", args={}, output=medium)
    assert result.mode != "trimmed"
    assert "expand" not in result.text
    assert api.score_bodies == [], "the API isn't asked when no cut would be worth it"


# Distinct words, so the log lines don't collapse as repeats of one another.
WORDS = [f"{a}{b}{c}" for a in "bcdfghk" for b in "aeiou" for c in "lmnprst"][:430]


def test_the_note_counts_the_omitted_log_lines_by_level(api: ApiStub) -> None:
    log = "\n".join(
        [
            *(f"2026-10-15 10:00:00 INFO served {w}" for w in WORDS[:400]),
            *(f"2026-10-15 11:00:00 DEBUG cache warm for {w}" for w in WORDS[400:430]),
            "2026-10-15 12:00:00 WARN payment provider slow for invoice INV-77",
        ]
    )
    with connected(max_tokens=200) as cw:
        cw.note_intent("l1", "Why is invoice INV-77 slow?")
        result = cw.shrink(call_id="l1", tool="search_logs", args={}, output=log)
    assert result.mode == "trimmed"
    assert "INV-77" in result.text
    assert re.search(r"Omitted: \d+ lines, INFO ×\d+\.", result.text)


def _orders(n: int, **extra: object) -> list[dict[str, object]]:
    return [
        {
            "id": f"ORD-{1000 + i}",
            "customer": f"Customer {i}",
            "items": [{"sku": f"SKU-{i}-{j}", "name": f"Product {j} with a long name"} for j in range(4)],
            **{k: v(i) if callable(v) else v for k, v in extra.items()},
        }
        for i in range(n)
    ]


def test_any_tool_output_reaches_the_model(api: ApiStub) -> None:
    circular: dict[str, object] = {}
    circular["self"] = circular
    with connected() as cw:
        for i, output in enumerate([None, circular]):
            assert cw.shrink(call_id=f"odd{i}", tool="t", output=output).mode == "full"


def test_data_in_a_format_that_is_not_plain_json_is_still_cut(api: ApiStub) -> None:
    data = json.dumps({"orders": _orders(80)})
    outputs = {
        "fenced": f"```json\n{data}\n```",
        "introduced": f"Found 80 orders:\n{data}",
        "one_line": " ".join(f"{o['id']} for {o['customer']}." for o in _orders(300)),
        "xml_one_line": "<orders>"
        + "".join(f'<order id="{o["id"]}"><c>{o["customer"]}</c></order>' for o in _orders(200))
        + "</orders>",
    }
    with connected(max_tokens=300) as cw:
        for name, output in outputs.items():
            view = cw.shrink(call_id=name, tool="lookup", args={"id": "ORD-1042"}, output=output)
            assert view.mode == "trimmed", name
            assert "ORD-1042" in view.text, name


def test_json_items_mentioning_a_failure_are_data_not_errors(api: ApiStub) -> None:
    output = _orders(80, payment=lambda i: "failed" if i % 2 else "ok")
    with connected(max_tokens=300) as cw:
        view = cw.shrink(call_id="pay", tool="lookup", args={"id": "ORD-1042"}, output=output)
    assert view.mode == "trimmed"
    assert "ORD-1042" in view.text


def test_a_call_marked_full_passes_whole_and_the_model_id_is_reported(api: ApiStub) -> None:
    with connected(max_tokens=60) as cw:
        view = cw.shrink(
            call_id="f1", tool="list_files", output=LISTING, full=True, model="claude-sonnet-4-5"
        )
        cw.flush()
    assert view.mode == "full"
    assert api.score_bodies == []
    assert api.report_bodies[0]["results"][0]["model"] == "claude-sonnet-4-5"


def test_a_cut_must_save_more_in_a_long_conversation(api: ApiStub) -> None:
    output = _orders(40)
    with connected(max_tokens=300) as cw:
        cw.note_intent("short", "Find ORD-1012", context_tokens=2_000)
        cw.note_intent("long", "Find ORD-1012", context_tokens=60_000)
        assert cw.shrink(call_id="short", tool="lookup", output=output).mode == "trimmed"
        assert cw.shrink(call_id="long", tool="lookup", output=output).mode != "trimmed"


def test_a_view_that_would_keep_nothing_is_not_sent(api: ApiStub) -> None:
    blob = "zzzz " * 4000
    with connected(max_tokens=200) as cw:
        view = cw.shrink(call_id="blob", tool="dump", args={"q": "invoices"}, output=blob)
    assert view.mode != "trimmed"


def test_asked_again_expand_returns_parts_not_shown_yet_then_says_so(api: ApiStub) -> None:
    rows = "\n".join(f"row {i} carrier {'Hermes' if i % 50 == 7 else 'DHL'} " + "x" * 40 for i in range(400))
    with connected(max_tokens=200) as cw:
        cw.note_intent("e1", "Which rows ship with Hermes?")
        view = cw.shrink(call_id="e1", tool="rows", output=rows)
        answers = [cw.expand(view.ref, "Hermes") for _ in range(4)]
    assert view.mode == "trimmed"
    assert answers[0] != answers[1] or "Nothing more" in answers[1]
    assert "Nothing more" in answers[-1]


def test_a_request_the_api_rejects_does_not_pause_scoring(api: ApiStub) -> None:
    with connected(max_tokens=60) as cw:
        cw.note_intent("big", "src/feature42")
        api.reject_next = True
        first = cw.shrink(call_id="big", tool="list_files", output=LISTING)
        second = cw.shrink(call_id="next", tool="list_files", args={"q": "feature42"}, output=LISTING)
    assert first.scorer == "none"
    assert second.scorer == "jev", "the next result is scored"
