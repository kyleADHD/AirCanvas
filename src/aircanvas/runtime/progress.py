"""Live run telemetry: the seam a UI reads while a generation is happening.

`pipe.report()` answers "where did the time go" once a run is over. A progress
UI needs the same facts *during* the run, and the three places that know are
the orchestrator (phase boundaries), the diffusers denoise loop (steps) and
the StreamingEngine (blocks, bytes, prefetch hits). The first two push into a
`RunObserver`; the third is polled through `AirPipeline.live_stats()`, because
block loads happen thousands of times per run and a callback per load would
cost more than the telemetry is worth.

Observers are advisory. Anything they raise is logged and swallowed
(`notify`): a UI that disconnects mid-run must never take the generation with
it.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

#: Phase names, in the order they occur (ARCHITECTURE.md §3.3).
PHASES = ("encode", "denoise", "decode")


@runtime_checkable
class RunObserver(Protocol):
    """Receives phase and step events from one `generate()` call."""

    def phase_started(self, phase: str) -> None:
        """`phase` is about to run. Called once per phase, in PHASES order."""

    def phase_finished(self, phase: str, seconds: float) -> None:
        """`phase` is over, having taken `seconds` of measured wall time."""

    def step(self, index: int, total: int) -> None:
        """Denoise step `index` (1-based) of `total` has just completed."""


class NullObserver:
    """Does nothing. The default, so callers never branch on None."""

    def phase_started(self, phase: str) -> None:
        return None

    def phase_finished(self, phase: str, seconds: float) -> None:
        return None

    def step(self, index: int, total: int) -> None:
        return None


def notify(observer: RunObserver | None, method: str, *args: Any) -> None:
    """Call `observer.method(*args)`, logging and swallowing any failure."""
    if observer is None:
        return
    fn = getattr(observer, method, None)
    if not callable(fn):
        return
    try:
        fn(*args)
    except Exception:  # noqa: BLE001 — telemetry must never break a run
        logger.exception("Run observer %s.%s failed; continuing", type(observer).__name__, method)
