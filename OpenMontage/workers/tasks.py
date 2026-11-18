"""Celery tasks for external integrations."""

from __future__ import annotations

from celery import Task
from notion_client.errors import APIResponseError

from workers.celery_app import celery_app


class NotionReadTask(Task):
    max_retries = 5


@celery_app.task(
    bind=True,
    base=NotionReadTask,
    name="workers.tasks.read_notion_page",
    rate_limit="2/s",
)
def read_notion_page(self: NotionReadTask, page_id: str) -> dict:
    """Fetch a Notion page and its direct content blocks in the background."""
    from lib.integrations.notion import NotionService

    try:
        return NotionService().get_page(page_id)
    except APIResponseError as exc:
        status = getattr(exc, "status", 0)
        if status == 429 or status >= 500:
            headers = getattr(exc, "headers", {}) or {}
            try:
                countdown = float(headers.get("Retry-After", 0))
            except (TypeError, ValueError):
                countdown = 0
            if countdown <= 0:
                countdown = min(2 ** self.request.retries, 60)
            raise self.retry(exc=exc, countdown=countdown)
        raise
