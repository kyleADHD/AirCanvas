"""Three-stage prefetch pipeline (M3).

  NVMe --worker thread--> pinned CPU ring --dedicated CUDA copy stream--> GPU slots

All buffers are allocated ONCE at construction (CLAUDE.md hard rule: no GPU or
pinned allocations in the per-block hot loop):

- The worker reads a shard's raw data blob straight into a pinned ring buffer
  (`readinto`, GIL released) and builds zero-copy tensor views by parsing the
  safetensors header ourselves (`ShardHeader`).
- On CUDA it then enqueues ONE flat byte copy into a reusable GPU slot buffer
  on the copy stream and records a ready event; per-tensor GPU views are
  metadata only. Slot reuse is gated GPU-side (copy stream waits the previous
  consumer's compute event); pinned-buffer reuse is gated CPU-side
  (synchronize on the buffer's last copy event — instant by the time a ring
  slot cycles).
- The consumer (engine pre-hook) makes the default stream wait the ready
  event and binds the views. A single worker suffices: reads are large and
  sequential, and empirically one-block lookahead hides most transfer cost
  (RESEARCH.md §4).

The denoise schedule is deterministic (ADR #7), so the worker just walks the
recorded schedule cyclically. If execution ever diverges from it, get() raises
PrefetchMismatch and the engine falls back to synchronous loads for the rest
of the run.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

import torch

from aircanvas.config import StreamConfig
from aircanvas.sharding.manifest import Manifest

logger = logging.getLogger(__name__)

_SAFETENSORS_DTYPES: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


class PrefetchError(RuntimeError):
    """The pipeline cannot run (unsupported shard layout, worker died, timeout)."""


class PrefetchMismatch(RuntimeError):
    """Execution order diverged from the recorded schedule."""


@dataclass(frozen=True)
class _TensorMeta:
    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    start: int  # byte offsets into the data blob
    end: int


class ShardHeader:
    """Parsed safetensors header: offsets for zero-copy views into a raw blob."""

    def __init__(self, path: Path, data_start: int, blob_size: int, tensors: list[_TensorMeta]):
        self.path = path
        self.data_start = data_start
        self.blob_size = blob_size
        self.tensors = tensors
        # Views require each tensor's byte offset to be aligned to its element
        # size. True for all shards we write today (uniform 2/4-byte floats);
        # quantized shards (M4/M5) must pad at split time to stay eligible.
        self.aligned = all(
            m.start % max(1, m.dtype.itemsize) == 0 for m in tensors if m.end > m.start
        )

    @classmethod
    def parse(cls, path: Path) -> ShardHeader:
        with open(path, "rb") as f:
            header_len = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(header_len))
        data_start = 8 + header_len
        blob_size = path.stat().st_size - data_start
        tensors = []
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtype = _SAFETENSORS_DTYPES.get(meta["dtype"])
            if dtype is None:
                raise PrefetchError(f"{path.name}: unsupported dtype {meta['dtype']!r}")
            start, end = meta["data_offsets"]
            tensors.append(_TensorMeta(name, dtype, tuple(meta["shape"]), start, end))
        return cls(path, data_start, blob_size, tensors)

    def read_blob_into(self, buf: torch.Tensor) -> None:
        """Fill `buf[:blob_size]` (uint8, CPU) with the shard's data blob."""
        with open(self.path, "rb", buffering=0) as f:
            f.seek(self.data_start)
            n = f.readinto(memoryview(buf.numpy())[: self.blob_size])
        if n != self.blob_size:
            raise PrefetchError(f"Short read on {self.path.name}: {n} != {self.blob_size}")

    def build_views(self, blob: torch.Tensor) -> dict[str, torch.Tensor]:
        """Zero-copy tensor views into `blob` (uint8, CPU or CUDA)."""
        out: dict[str, torch.Tensor] = {}
        for m in self.tensors:
            raw = blob[m.start : m.end]
            out[m.name] = raw.view(m.dtype).reshape(m.shape)
        return out


class _PinnedBuf:
    __slots__ = ("buf", "event")

    def __init__(self, buf: torch.Tensor):
        self.buf = buf
        self.event: torch.cuda.Event | None = None  # last H2D copy out of this buf


class _Slot:
    __slots__ = ("buf", "reuse_event")

    def __init__(self, buf: torch.Tensor):
        self.buf = buf
        self.reuse_event: torch.cuda.Event | None = None  # last consumer's compute


@dataclass
class ReadyItem:
    name: str
    views: dict[str, torch.Tensor]
    ready_event: torch.cuda.Event | None
    pinned: _PinnedBuf
    slot: _Slot | None


_POLL_S = 0.1


class Prefetcher:
    """Walks the recorded schedule cyclically, staying ahead of the consumer."""

    def __init__(
        self,
        manifest: Manifest,
        cache_dir: Path,
        schedule: tuple[str, ...],
        start_index: int,
        device: torch.device,
        config: StreamConfig | None = None,
    ) -> None:
        config = config or StreamConfig()
        if config.ring_depth < 1 or config.gpu_slots < 2:
            raise PrefetchError("ring_depth must be >= 1 and gpu_slots >= 2")
        by_name = {b.name: b for b in manifest.blocks}
        self._headers = {
            name: ShardHeader.parse(cache_dir / by_name[name].file) for name in schedule
        }
        bad = [n for n, h in self._headers.items() if not h.aligned]
        if bad:
            raise PrefetchError(f"Shards with misaligned tensors (re-split needed): {bad[:3]}")
        max_blob = max(h.blob_size for h in self._headers.values())
        self._schedule = schedule
        self._cuda = device.type == "cuda"
        self._device = device

        self._pinned: queue.Queue[_PinnedBuf] = queue.Queue()
        for _ in range(config.ring_depth):
            self._pinned.put(
                _PinnedBuf(torch.empty(max_blob, dtype=torch.uint8, pin_memory=self._cuda))
            )
        if self._cuda:
            self._copy_stream = torch.cuda.Stream(device)
            self._slots: queue.Queue[_Slot] = queue.Queue()
            for _ in range(config.gpu_slots):
                self._slots.put(_Slot(torch.empty(max_blob, dtype=torch.uint8, device=device)))
        self._ready: queue.Queue[ReadyItem] = queue.Queue(maxsize=config.ring_depth)
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, args=(start_index,), name="aircanvas-prefetch", daemon=True
        )
        self._thread.start()

    # -- consumer side -----------------------------------------------------

    def get(self, name: str, timeout: float = 120.0) -> ReadyItem:
        """Wait for the next prefetched block; it must be `name`."""
        deadline = timeout
        while True:
            if self._error is not None:
                raise PrefetchError(f"Prefetch worker died: {self._error}") from self._error
            try:
                item = self._ready.get(timeout=min(_POLL_S, deadline))
                break
            except queue.Empty:
                deadline -= _POLL_S
                if deadline <= 0:
                    raise PrefetchError(f"Timed out waiting for prefetch of {name}") from None
        if item.name != name:
            raise PrefetchMismatch(f"prefetched {item.name!r} but execution asked for {name!r}")
        if item.ready_event is not None:
            torch.cuda.current_stream().wait_event(item.ready_event)
        return item

    def release(self, item: ReadyItem) -> None:
        """Return buffers after the consumer's post-hook. On CUDA, gate slot
        reuse on the work enqueued so far (the block's compute)."""
        if item.slot is not None:
            ev = torch.cuda.Event()
            ev.record()
            item.slot.reuse_event = ev
            self._slots.put(item.slot)
            item.pinned.event = item.ready_event
        else:
            item.pinned.event = None
        self._pinned.put(item.pinned)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    # -- worker side -------------------------------------------------------

    def _acquire(self, q: queue.Queue) -> object | None:
        while not self._stop.is_set():
            try:
                return q.get(timeout=_POLL_S)
            except queue.Empty:
                continue
        return None

    def _run(self, start_index: int) -> None:
        i = start_index
        n = len(self._schedule)
        try:
            while not self._stop.is_set():
                name = self._schedule[i % n]
                header = self._headers[name]
                pinned = self._acquire(self._pinned)
                if pinned is None:
                    return
                if pinned.event is not None:
                    pinned.event.synchronize()  # prior copy out of this buf done
                header.read_blob_into(pinned.buf)

                if self._cuda:
                    slot = self._acquire(self._slots)
                    if slot is None:
                        return
                    with torch.cuda.stream(self._copy_stream):
                        if slot.reuse_event is not None:
                            self._copy_stream.wait_event(slot.reuse_event)
                        slot.buf[: header.blob_size].copy_(
                            pinned.buf[: header.blob_size], non_blocking=True
                        )
                        ready = torch.cuda.Event()
                        ready.record(self._copy_stream)
                    item = ReadyItem(name, header.build_views(slot.buf), ready, pinned, slot)
                else:
                    item = ReadyItem(name, header.build_views(pinned.buf), None, pinned, None)

                while not self._stop.is_set():
                    try:
                        self._ready.put(item, timeout=_POLL_S)
                        break
                    except queue.Full:
                        continue
                i += 1
        except BaseException as e:  # surfaced to the consumer in get()
            logger.exception("Prefetch worker failed")
            self._error = e
