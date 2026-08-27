"""Fan-out of Studio events to every connected browser.

Worker threads publish; the SSE endpoint consumes. Each subscriber gets its own
unbounded queue, because dropping a "run finished" event to save memory would
strand a screen mid-generation forever — and the events are small and bounded
in rate (telemetry is throttled at the source, jobs.py).

A short backlog is replayed to a subscriber that arrives mid-run, so a browser
reloaded during a 7-minute generation redraws the live screen immediately
instead of waiting for the next tick.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

BACKLOG = 32


class Subscriber:
    """One connected client's queue."""

    def __init__(self) -> None:
        self._queue: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()

    def put(self, event: dict[str, Any]) -> None:
        self._queue.put(event)

    def get(self, timeout: float) -> dict[str, Any] | None:
        """Next event, or None if `timeout` seconds pass without one.

        Blocking: call it off the event loop (`asyncio.to_thread`). The None
        is how the SSE endpoint knows to send a keep-alive comment.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


class EventBus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: set[Subscriber] = set()
        self._backlog: list[dict[str, Any]] = []

    def publish(self, kind: str, **payload: Any) -> dict[str, Any]:
        event = {"type": kind, **payload}
        with self._lock:
            self._backlog.append(event)
            del self._backlog[:-BACKLOG]
            targets = list(self._subscribers)
        for subscriber in targets:
            subscriber.put(event)
        return event

    @contextmanager
    def subscribe(self, *, replay: bool = True) -> Iterator[Subscriber]:
        subscriber = Subscriber()
        with self._lock:
            self._subscribers.add(subscriber)
            backlog = list(self._backlog) if replay else []
        for event in backlog:
            subscriber.put(event)
        try:
            yield subscriber
        finally:
            with self._lock:
                self._subscribers.discard(subscriber)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)
