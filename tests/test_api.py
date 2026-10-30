"""The SDK's side of the Contextwaise API: scoring, reports, failure handling."""

from __future__ import annotations

import json
import logging

import pytest

from contextwaise import Contextwaise, LedgerEntry, MemoryLedger, ShrinkResult
from tests.stubs import API_KEY, BASE_URL, ApiStub

LISTING = "\n".join(
    [*(f"src/feature{i}/components/View{i * 3}.tsx" for i in range(200)), "src/auth/middleware.ts"]
)


def test_relevance_comes_from_the_api_and_results_and_expands_are_reported_in_one_batch(api: ApiStub) -> None:
    with Contextwaise(api_key=API_KEY, base_url=BASE_URL + "/", agent="triage", max_tokens=60) as cw:
        cw.note_intent("c1", "Where is the auth middleware?")
        result = cw.shrink(call_id="c1", tool="list_files", args={"dir": "."}, output=LISTING)
        assert result.scorer == "jev"
        assert "src/auth/" in result.text
        cw.expand(result.ref, "feature 12")
        cw.flush()

    assert api.score_bodies[0]["agent"] == "triage"
    [report] = api.report_bodies
    assert report["agent"] == "triage"
    assert [r["type"] for r in report["results"]] == ["result", "expand"]
    assert report["results"][0]["tool"] == "list_files"
    assert report["results"][0]["integration"] == "python"
    assert report["results"][0]["reason"] == "cut"
    assert "middleware" not in json.dumps(report), "reports never carry tool output"


def test_api_down_nothing_is_cut_and_the_api_is_paused(
    api: ApiStub, caplog: pytest.LogCaptureFixture
) -> None:
    api.down = True
    with (
        caplog.at_level(logging.WARNING, "contextwaise"),
        Contextwaise(api_key=API_KEY, base_url=BASE_URL, max_tokens=60) as cw,
    ):
        a = cw.shrink(call_id="a", tool="list_files", output=LISTING)
        b = cw.shrink(call_id="b", tool="list_files", output=LISTING)
    assert (a.scorer, b.scorer) == ("none", "none")
    assert "feature199" in a.text, "the whole cleaned output reaches the model"
    assert len(api.score_bodies) == 1, "paused after the first failure"
    assert "relevance API unavailable" in caplog.text


def test_requests_stay_within_the_api_limits(api: ApiStub) -> None:
    odd_call_id = "call with spaces/" + "x" * 200
    blob = "y" * 200_000  # one unit far over the API's per-section limit
    with Contextwaise(api_key=API_KEY, base_url=BASE_URL, max_tokens=60) as cw:
        result = cw.shrink(
            call_id=odd_call_id, tool="read", args={"content": "z" * 10_000}, output=blob + "\n\nend"
        )
        cw.flush()
    [body] = api.score_bodies
    assert len(body["callId"]) <= 128
    assert " " not in body["callId"]
    assert len(body["intent"]) <= 4000
    assert all(len(s) <= 20_000 for s in body["sections"])
    assert result.scorer == "jev", "the API accepted the call"
    assert api.report_bodies[0]["results"][0]["callId"] == body["callId"]


def test_refs_expand_in_any_process_sharing_the_ledger(api: ApiStub) -> None:
    ledger = MemoryLedger()
    with Contextwaise(api_key=API_KEY, base_url=BASE_URL, max_tokens=60, ledger=ledger) as first:
        result = first.shrink(call_id="c1", tool="list_files", output=LISTING)
    with Contextwaise(api_key=API_KEY, base_url=BASE_URL, ledger=ledger) as second:
        assert "feature12/" in second.expand(result.ref, "feature12")


def test_the_memory_ledger_keeps_only_the_most_recent_calls() -> None:
    ledger = MemoryLedger(3)
    for call_id in "1234":
        view = ShrinkResult(
            text=call_id, mode="full", ref=call_id, kind="text", tokens_before=1, tokens_after=1
        )
        ledger.set(call_id, LedgerEntry(call_id=call_id, tool="t", original=call_id, result=view))
    assert ledger.get("1") is None
    entry = ledger.get("4")
    assert entry is not None
    assert entry.original == "4"


def test_one_client_many_agents_each_call_is_scored_and_reported_under_its_own_agent(api: ApiStub) -> None:
    with Contextwaise(api_key=API_KEY, base_url=BASE_URL, agent="platform", max_tokens=60) as cw:
        cw.note_intent("a", "Where is the auth middleware?")
        result = cw.shrink(call_id="a", tool="list_files", output=LISTING, agent="Support bot")
        cw.expand(result.ref, "feature 12")
        cw.shrink(call_id="b", tool="echo", output="ok")
    assert [b.get("agent") for b in api.score_bodies] == ["Support bot", "Support bot"]
    results = [r for body in api.report_bodies for r in body["results"]]
    assert [r.get("agent") for r in results] == ["Support bot", "Support bot", "platform"]


def test_free_tokens_used_up_scoring_pauses_for_an_hour(
    api: ApiStub, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    api.quota_used_up = True
    now = time.monotonic()
    with (
        caplog.at_level(logging.WARNING, "contextwaise"),
        Contextwaise(api_key=API_KEY, base_url=BASE_URL, max_tokens=60) as cw,
    ):
        cw.shrink(call_id="a", tool="list_files", output=LISTING)
        monkeypatch.setattr(time, "monotonic", lambda: now + 31)
        b = cw.shrink(call_id="b", tool="list_files", output=LISTING)
    assert len(api.score_bodies) == 1, "still paused after 31s"
    assert b.reason == "api_unavailable"
    assert "used up" in caplog.text
