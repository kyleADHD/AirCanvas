"""Studio state that outlives the process, and where it is kept.

One JSON file under the AirCanvas cache root — never inside a project, the
same hard rule the shard cache follows (CONTRIBUTING.md). It holds the choices
a user would be annoyed to re-make (mode, theme, prompt, budget caps, the LoRA
stack) and the gallery index; it holds no telemetry, because telemetry belongs
to a run and a run does not survive the process.

Writes are atomic (temp file + `os.replace`) and serialised under one lock: a
generation thread appending an output and an HTTP request toggling a setting
are genuinely concurrent, and a half-written state file would lose the gallery.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from aircanvas.config import cache_root

logger = logging.getLogger(__name__)

STATE_VERSION = 1
STATE_NAME = "studio.json"

#: Shape presets the Simple desk offers, in the order it shows them.
SHAPES: dict[str, tuple[int, int]] = {
    "square": (1024, 1024),
    "portrait": (832, 1216),
    "landscape": (1216, 832),
    "widescreen": (1344, 768),
}

#: Quality presets, as step counts. "Best" is the model's own default.
QUALITY_STEPS: dict[str, float] = {"fast": 0.4, "balanced": 0.7, "best": 1.0}


def studio_root() -> Path:
    return cache_root() / "studio"


def default_state() -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "mode": "simple",
        "theme": "dark",
        "seenWelcome": False,
        "desk": {
            "modelId": None,
            "format": None,
            "prompt": "",
            "negativePrompt": "",
            "shape": "square",
            "quality": "best",
            "width": 1024,
            "height": 1024,
            "steps": 20,
            "seed": None,
            "guidance": 3.5,
            "frames": 1,
            "startFrame": None,
            "loras": [],
            "caps": {"vramBytes": None, "ramBytes": None},
        },
        "settings": {
            "embeddingCache": True,
            "backgroundPrefetch": True,
            "cachePath": None,
        },
        "setup": {"selection": {}},
        "outputs": [],
    }


def _merge_defaults(state: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict[str, Any]:
    """Fill in keys a state file written by an older build does not have.

    Recursive on dicts only: a list in the state file (the LoRA stack, the
    gallery) is the user's, and merging into it would resurrect deleted rows.
    """
    merged = dict(state)
    for key, value in defaults.items():
        if key not in merged:
            merged[key] = copy.deepcopy(value)
        elif isinstance(value, Mapping) and isinstance(merged[key], Mapping):
            merged[key] = _merge_defaults(merged[key], value)
    return merged


class Store:
    """Thread-safe, atomically-persisted Studio state."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (studio_root() / STATE_NAME)
        self._lock = threading.RLock()
        self._state = self._read()

    # -- io ----------------------------------------------------------------

    def _read(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default_state()
        except (OSError, ValueError) as e:
            logger.warning("Unreadable Studio state at %s (%s); starting fresh", self.path, e)
            return default_state()
        if not isinstance(raw, dict) or raw.get("version") != STATE_VERSION:
            logger.info("Studio state version %s is not %s; starting fresh", raw, STATE_VERSION)
            return default_state()
        return _merge_defaults(raw, default_state())

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    # -- access ------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """A deep copy nothing else can mutate underneath the caller."""
        with self._lock:
            return copy.deepcopy(self._state)

    @contextmanager
    def edit(self) -> Iterator[dict[str, Any]]:
        """Mutate the state under the lock; persisted on a clean exit."""
        with self._lock:
            yield self._state
            self._write()

    def update(self, fn: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        with self.edit() as state:
            fn(state)
        return self.snapshot()

    def patch(self, section: str, values: Mapping[str, Any]) -> dict[str, Any]:
        """Shallow-merge `values` into one top-level section."""

        def apply(state: dict[str, Any]) -> None:
            target = state.setdefault(section, {})
            if not isinstance(target, dict):
                raise TypeError(f"State section {section!r} is not an object")
            target.update(values)

        return self.update(apply)

    def set(self, key: str, value: Any) -> dict[str, Any]:
        return self.update(lambda state: state.__setitem__(key, value))

    # -- gallery -----------------------------------------------------------

    def outputs_dir(self) -> Path:
        path = self.path.parent / "outputs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def add_output(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        """Prepend one gallery entry (newest first) and persist it."""
        record = dict(entry)
        record.setdefault("id", new_id())
        record.setdefault("createdAt", time.time())
        record.setdefault("reproducedCount", 0)
        with self.edit() as state:
            outputs = state.setdefault("outputs", [])
            outputs.insert(0, record)
        return record

    def update_output(self, output_id: str, values: Mapping[str, Any]) -> dict[str, Any] | None:
        found: dict[str, Any] | None = None
        with self.edit() as state:
            for entry in state.get("outputs", []):
                if entry.get("id") == output_id:
                    entry.update(values)
                    found = dict(entry)
                    break
        return found

    def get_output(self, output_id: str) -> dict[str, Any] | None:
        for entry in self.snapshot().get("outputs", []):
            if entry.get("id") == output_id:
                return entry
        return None

    def remove_output(self, output_id: str) -> bool:
        removed = False
        with self.edit() as state:
            outputs = state.get("outputs", [])
            keep = [e for e in outputs if e.get("id") != output_id]
            removed = len(keep) != len(outputs)
            state["outputs"] = keep
        return removed

    def last_measurement(self, model_id: str, *, steps: int, pixels: int) -> float | None:
        """Seconds per step from the most recent comparable finished run.

        Comparable means the same model at the same shape and step count:
        streaming cost per step is set by the shard set, but wall time per step
        is set by the workload, so a 512² measurement must not be presented as
        an estimate for 1024². This is what makes the Desk's estimate improve
        as the user actually uses the machine.
        """
        for entry in self.snapshot().get("outputs", []):
            if entry.get("state") != "done" or entry.get("modelId") != model_id:
                continue
            if int(entry.get("steps") or 0) != steps:
                continue
            width, height = int(entry.get("width") or 0), int(entry.get("height") or 0)
            frames = max(1, int(entry.get("frames") or 1))
            if width * height * frames != pixels:
                continue
            per_step = entry.get("secondsPerStep")
            if isinstance(per_step, (int, float)) and per_step > 0:
                return float(per_step)
        return None


def new_id() -> str:
    """Short, sortable-enough, filesystem-safe id for a run or an output."""
    return uuid.uuid4().hex[:12]
