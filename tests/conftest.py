"""Shared fixtures. No network: the Contextwaise API is a respx stub."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import respx

from tests.stubs import BASE_URL, ApiStub


@pytest.fixture(autouse=True)
def _no_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's own key must never reach the tests."""
    monkeypatch.delenv("CONTEXTWAISE_API_KEY", raising=False)
    monkeypatch.delenv("CONTEXTWAISE_URL", raising=False)


@pytest.fixture
def api() -> Iterator[ApiStub]:
    stub = ApiStub()
    with respx.mock(base_url=BASE_URL, assert_all_called=False) as router:
        router.post("/api/v1/tool-results").mock(side_effect=stub.score)
        router.post("/api/v1/tool-results/report").mock(side_effect=stub.report)
        yield stub
