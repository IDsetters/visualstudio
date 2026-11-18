"""Celery application configuration for OpenMontage background tasks."""

from __future__ import annotations

import os

from celery import Celery

from lib.env_loader import load_env

load_env()

celery_app = Celery(
    "openmontage",
    broker=os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
    include=["workers.tasks"],
)

celery_app.conf.update(
    task_default_queue=os.environ.get("CELERY_TASK_DEFAULT_QUEUE", "openmontage"),
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    worker_concurrency=int(os.environ.get("CELERY_WORKER_CONCURRENCY", "1")),
    broker_connection_retry_on_startup=True,
    result_expires=3600,
    timezone="UTC",
    enable_utc=True,
)
