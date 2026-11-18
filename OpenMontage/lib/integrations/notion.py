"""Notion API adapter for reading production briefs and scripts."""

from __future__ import annotations

import os
import threading
import time
from typing import Any

from notion_client import Client

from lib.env_loader import load_env

_request_lock = threading.Lock()
_last_request_at = 0.0


class NotionService:
    """Small adapter around the Notion API with cursor-based pagination."""

    def __init__(self, token: str | None = None, client: Any | None = None) -> None:
        load_env()
        resolved_token = (token if token is not None else os.environ.get("NOTION_TOKEN", "")).strip()
        if client is None and not resolved_token:
            raise RuntimeError("NOTION_TOKEN is required to use the Notion integration")

        self._client = client if client is not None else Client(auth=resolved_token)

    def get_page(self, page_id: str) -> dict[str, Any]:
        """Return a page's properties and all of its direct content blocks."""
        page_id = page_id.strip()
        if not page_id:
            raise ValueError("page_id must not be empty")

        page = self._request(self._client.pages.retrieve, page_id=page_id)
        return {
            "page": page,
            "blocks": self._list_block_children(page_id),
        }

    def query_database(self, database_id: str, **filters: Any) -> list[dict[str, Any]]:
        """Return all pages matching a database query, following pagination."""
        database_id = database_id.strip()
        if not database_id:
            raise ValueError("database_id must not be empty")

        results: list[dict[str, Any]] = []

        if hasattr(self._client.databases, "query"):
            query_targets = [(self._client.databases.query, {"database_id": database_id})]
        else:
            database = self._request(self._client.databases.retrieve, database_id=database_id)
            data_sources = database.get("data_sources", [])
            if not data_sources:
                raise RuntimeError(f"Notion database {database_id!r} has no queryable data sources")
            query_targets = [
                (self._client.data_sources.query, {"data_source_id": source["id"]})
                for source in data_sources
            ]

        for operation, target in query_targets:
            query = dict(filters)
            query.setdefault("page_size", 100)
            while True:
                response = self._request(operation, **target, **query)
                results.extend(response["results"])
                if not response.get("has_more"):
                    break

                next_cursor = response.get("next_cursor")
                if not next_cursor:
                    raise RuntimeError("Notion returned has_more without a next_cursor")
                query["start_cursor"] = next_cursor

        return results

    def _list_block_children(self, block_id: str) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        cursor: str | None = None

        while True:
            params: dict[str, Any] = {"block_id": block_id, "page_size": 100}
            if cursor is not None:
                params["start_cursor"] = cursor
            response = self._request(self._client.blocks.children.list, **params)
            blocks.extend(response["results"])
            if not response.get("has_more"):
                return blocks

            cursor = response.get("next_cursor")
            if not cursor:
                raise RuntimeError("Notion returned has_more without a next_cursor")

    @staticmethod
    def _request(operation: Any, **kwargs: Any) -> Any:
        global _last_request_at

        interval = float(os.environ.get("NOTION_MIN_REQUEST_INTERVAL_SECONDS", "0.4"))
        if interval < 0:
            raise ValueError("NOTION_MIN_REQUEST_INTERVAL_SECONDS must be non-negative")

        with _request_lock:
            now = time.monotonic()
            delay = interval - (now - _last_request_at)
            if delay > 0:
                time.sleep(delay)
            _last_request_at = time.monotonic()

        return operation(**kwargs)
