from __future__ import annotations

from types import SimpleNamespace

import pytest

from lib.integrations.notion import NotionService


def test_notion_page_reads_all_direct_block_pages(monkeypatch) -> None:
    monkeypatch.setenv("NOTION_MIN_REQUEST_INTERVAL_SECONDS", "0")

    class FakeBlocks:
        def __init__(self) -> None:
            self.calls = []

        def list(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return {
                    "results": [{"id": "block-1"}],
                    "has_more": True,
                    "next_cursor": "next",
                }
            return {"results": [{"id": "block-2"}], "has_more": False}

    page = {"id": "page-id"}
    blocks = FakeBlocks()
    client = SimpleNamespace(
        pages=SimpleNamespace(retrieve=lambda **kwargs: page),
        blocks=SimpleNamespace(children=blocks),
    )

    result = NotionService(client=client).get_page("page-id")

    assert result == {"page": page, "blocks": [{"id": "block-1"}, {"id": "block-2"}]}
    assert blocks.calls[1]["start_cursor"] == "next"


def test_notion_requires_internal_integration_token(monkeypatch) -> None:
    monkeypatch.delenv("NOTION_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="NOTION_TOKEN"):
        NotionService()


def test_notion_database_query_supports_current_data_source_api(monkeypatch) -> None:
    monkeypatch.setenv("NOTION_MIN_REQUEST_INTERVAL_SECONDS", "0")
    calls = []

    def query_data_source(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {
                "results": [{"id": "page-1"}],
                "has_more": True,
                "next_cursor": "next",
            }
        return {"results": [{"id": "page-2"}], "has_more": False}

    client = SimpleNamespace(
        databases=SimpleNamespace(
            retrieve=lambda **kwargs: {"data_sources": [{"id": "source-id"}]},
        ),
        data_sources=SimpleNamespace(query=query_data_source),
    )

    result = NotionService(client=client).query_database("database-id")

    assert result == [{"id": "page-1"}, {"id": "page-2"}]
    assert calls[0]["data_source_id"] == "source-id"
    assert calls[1]["start_cursor"] == "next"
