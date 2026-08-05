"""Three-stage prefetch pipeline (M3), with an fp8 decompress stage (M4).

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

**Compressed shard caches (M4, ADR #8).** When the manifest says fp8, the raw
blob is no longer bindable — payloads are `float8_e4m3fn` and carry sibling
scales. A `DecompressPlan` (built once per shard, at construction) turns the
raw blob into compute-dtype tensors with two stream-ordered kernels per tensor
(`copy_` does the dtype conversion, `mul_` applies the scale). To keep the
no-allocation rule intact this needs TWO pre-allocated pools:

    pinned ring (raw) --H2D--> raw staging slot --upcast--> gpu slot --> compute

The raw staging slots hold compressed bytes only and are handed back by the
worker as soon as the upcast kernels are enqueued (stream ordering makes that
safe), so `raw_slots=2` is enough to overlap the next H2D with the current
upcast. VRAM cost is therefore `gpu_slots x materialised + 2 x compressed`
instead of `gpu_slots x materialised` — the price of halving disk traffic.
With `compression=None` neither the plan nor the raw pool exists and the M2/M3
zero-copy path is untouched, which is what keeps the bitwise gate exact.

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
from typing import TypeVar

import torch

from aircanvas.config import Compression, StreamConfig
from aircanvas.sharding import quant
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
        # size. True for uniform-dtype shards (2/4-byte floats) and for the
        # mixed-dtype shards the splitter writes, which are ordered largest
        # element size first (sharding/quant.py: order_for_alignment).
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
            # ndarray.data is the buffer typeshed knows about; memoryview(ndarray)
            # works at runtime but is not typed as a Buffer.
            n = f.readinto(buf.numpy().data[: self.blob_size])
        if n != self.blob_size:
            raise PrefetchError(f"Short read on {self.path.name}: {n} != {self.blob_size}")

    def build_views(self, blob: torch.Tensor) -> dict[str, torch.Tensor]:
        """Zero-copy tensor views into `blob` (uint8, CPU or CUDA)."""
        out: dict[str, torch.Tensor] = {}
        for m in self.tensors:
            raw = blob[m.start : m.end]
            out[m.name] = raw.view(m.dtype).reshape(m.shape)
        return out


@dataclass(frozen=True)
class _Move:
    """One tensor's journey from the raw blob to the materialised buffer."""

    name: str
    shape: tuple[int, ...]
    src_dtype: torch.dtype
    src_start: int
    src_end: int
    out_dtype: torch.dtype
    dst_start: int
    dst_end: int
    scale_start: int = -1  # byte range of the fp32 scale in the raw blob, -1 if none
    scale_end: int = -1


class DecompressPlan:
    """Precomputed byte layout for materialising one compressed shard.

    Built once per shard at Prefetcher construction; `run` only issues
    `copy_`/`mul_` against slices of two buffers the caller already owns, so
    nothing allocates in the hot loop.
    """

    def __init__(self, moves: tuple[_Move, ...], blob_size: int, out_bytes: int) -> None:
        self.moves = moves
        self.blob_size = blob_size
        self.out_bytes = out_bytes

    @classmethod
    def build(cls, header: ShardHeader, compute_dtype: torch.dtype) -> DecompressPlan:
        scales = {
            quant.payload_name(m.name): m for m in header.tensors if quant.is_scale_key(m.name)
        }
        moves: list[_Move] = []
        offset = 0
        for m in header.tensors:
            if quant.is_scale_key(m.name):
                continue
            out_dtype = quant.compressed_dtype(m.dtype, compute_dtype)
            itemsize = out_dtype.itemsize
            offset += (-offset) % itemsize  # keep every destination view alignable
            n_bytes = itemsize * torch.Size(m.shape).numel()
            scale = scales.get(m.name)
            moves.append(
                _Move(
                    name=m.name,
                    shape=m.shape,
                    src_dtype=m.dtype,
                    src_start=m.start,
                    src_end=m.end,
                    out_dtype=out_dtype,
                    dst_start=offset,
                    dst_end=offset + n_bytes,
                    scale_start=-1 if scale is None else scale.start,
                    scale_end=-1 if scale is None else scale.end,
                )
            )
            offset += n_bytes
        return cls(tuple(moves), header.blob_size, offset)

    def run(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """Materialise `src` (raw blob, uint8) into `dst` (uint8), in place."""
        for m in self.moves:
            out = dst[m.dst_start : m.dst_end].view(m.out_dtype).reshape(m.shape)
            out.copy_(src[m.src_start : m.src_end].view(m.src_dtype).reshape(m.shape))
            if m.scale_start >= 0:
                # 0-dim operand => type promotion keeps the in-place result in
                # `out_dtype` (a 1-element 1-D view would promote to float32
                # and make mul_ raise).
                out.mul_(src[m.scale_start : m.scale_end].view(torch.float32).reshape(()))

    def views(self, dst: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            m.name: dst[m.dst_start : m.dst_end].view(m.out_dtype).reshape(m.shape)
            for m in self.moves
        }


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


#: Both buffer wrappers are acquired from queues through the same helper.
_BufT = TypeVar("_BufT", _Slot, _PinnedBuf)


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
        if not schedule:
            raise PrefetchError("Nothing to prefetch: the recorded schedule is empty")
        by_name = {b.name: b for b in manifest.blocks}
        self._headers = {
            name: ShardHeader.parse(cache_dir / by_name[name].file) for name in schedule
        }
        bad = [n for n, h in self._headers.items() if not h.aligned]
        if bad:
            raise PrefetchError(f"Shards with misaligned tensors (re-split needed): {bad[:3]}")

        self._compression: Compression = manifest.compression
        self._plans: dict[str, DecompressPlan] | None = None
        if self._compression is not None:
            compute_dtype = manifest.torch_compute_dtype()
            assert isinstance(compute_dtype, torch.dtype)
            self._plans = {
                name: DecompressPlan.build(h, compute_dtype) for name, h in self._headers.items()
            }

        max_blob = max(h.blob_size for h in self._headers.values())
        max_out = (
            max(p.out_bytes for p in self._plans.values()) if self._plans is not None else max_blob
        )
        self._schedule = schedule
        self._cuda = device.type == "cuda"
        self._device = device

        self._pinned: queue.Queue[_PinnedBuf] = queue.Queue()
        for _ in range(config.ring_depth):
            self._pinned.put(
                _PinnedBuf(torch.empty(max_blob, dtype=torch.uint8, pin_memory=self._cuda))
            )
        # Materialised-weight slots: what the consumer binds. Present on CUDA
        # always, and on CPU only when a decompress stage needs a destination.
        self._slots: queue.Queue[_Slot] = queue.Queue()
        self._has_slots = self._cuda or self._plans is not None
        if self._has_slots:
            for _ in range(config.gpu_slots):
                self._slots.put(_Slot(torch.empty(max_out, dtype=torch.uint8, device=device)))
        # Compressed staging on the compute device, released early (ADR #8).
        self._raw: queue.Queue[_Slot] | None = None
        n_raw = 0
        if self._cuda and self._plans is not None:
            n_raw = max(2, config.raw_slots)
            self._raw = queue.Queue()
            for _ in range(n_raw):
                self._raw.put(_Slot(torch.empty(max_blob, dtype=torch.uint8, device=device)))
        if self._cuda:
            self._copy_stream = torch.cuda.Stream(device)
        self._slot_bytes = (config.gpu_slots * max_out if self._has_slots else 0) + n_raw * max_blob
        self._pinned_bytes = config.ring_depth * max_blob

        self._ready: queue.Queue[ReadyItem] = queue.Queue(maxsize=config.ring_depth)
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, args=(start_index,), name="aircanvas-prefetch", daemon=True
        )
        self._thread.start()

    # -- introspection -----------------------------------------------------

    @property
    def slot_bytes(self) -> int:
        """Compute-device bytes held by the pre-allocated slot pools."""
        return self._slot_bytes

    @property
    def pinned_bytes(self) -> int:
        """Host bytes held by the pinned ring."""
        return self._pinned_bytes

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
            if self._cuda:
                ev = torch.cuda.Event()
                ev.record()
                item.slot.reuse_event = ev
            self._slots.put(item.slot)
        item.pinned.event = item.ready_event
        self._pinned.put(item.pinned)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    # -- worker side -------------------------------------------------------

    def _acquire(self, q: queue.Queue[_BufT]) -> _BufT | None:
        """Block for a free buffer, or return None once close() is called."""
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

                item = self._stage(name, header, pinned)
                if item is None:
                    return

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

    def _stage(self, name: str, header: ShardHeader, pinned: _PinnedBuf) -> ReadyItem | None:
        """Move a freshly read blob onto the compute device (and decompress)."""
        plan = None if self._plans is None else self._plans[name]

        if not self._cuda:
            if plan is None:
                return ReadyItem(name, header.build_views(pinned.buf), None, pinned, None)
            out = self._acquire(self._slots)
            if out is None:
                return None
            plan.run(pinned.buf, out.buf)
            return ReadyItem(name, plan.views(out.buf), None, pinned, out)

        out = self._acquire(self._slots)
        if out is None:
            return None
        raw = None
        if plan is not None:
            raw_q = self._raw
            assert raw_q is not None
            raw = self._acquire(raw_q)
            if raw is None:
                self._slots.put(out)
                return None

        with torch.cuda.stream(self._copy_stream):
            if out.reuse_event is not None:
                self._copy_stream.wait_event(out.reuse_event)
            if plan is None:
                out.buf[: header.blob_size].copy_(pinned.buf[: header.blob_size], non_blocking=True)
                views = header.build_views(out.buf)
            else:
                assert raw is not None
                if raw.reuse_event is not None:
                    self._copy_stream.wait_event(raw.reuse_event)
                raw.buf[: header.blob_size].copy_(pinned.buf[: header.blob_size], non_blocking=True)
                plan.run(raw.buf, out.buf)
                views = plan.views(out.buf)
            ready = torch.cuda.Event()
            ready.record(self._copy_stream)

        if raw is not None:
            # Safe to recycle now: the next writer waits on `ready`, which is
            # ordered after every upcast kernel that read from this buffer.
            raw.reuse_event = ready
            assert self._raw is not None
            self._raw.put(raw)
        return ReadyItem(name, views, ready, pinned, out)
