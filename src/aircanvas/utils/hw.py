"""Hardware probing for the budget solver (M4).

- free/total VRAM: ``torch.cuda.mem_get_info``
- free/total RAM: psutil-free — ``GlobalMemoryStatusEx`` via ctypes on Windows,
  ``/proc/meminfo`` on Linux, ``sysconf`` elsewhere — so the runtime dependency
  list stays at what ARCHITECTURE.md §8 declares.
- disk read bandwidth: a timed REAL read of a real shard file, cached in a
  sidecar next to the shard cache. Deliberately not a platform API
  (ARCHITECTURE.md §5): what matters is what this file system actually
  delivers to `readinto`, including whatever the OS, the driver and any
  sync client put in the way.

Every probe degrades to a documented default rather than raising: a budget
solver that cannot run is worse than one running on an estimate, as long as
the estimate is visible in the plan (`aircanvas doctor`, `pipe.report()`).
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

#: Assumed when a probe is impossible: a slow-ish PCIe 3.0 NVMe (RESEARCH.md §6).
DEFAULT_DISK_BW = 2.0e9
#: Below this a disk is SATA/HDD class and streaming overhead becomes visible.
SATA_DISK_BW = 1.2e9
#: A sample smaller than this measures the page cache, not the disk.
MIN_PROBE_BYTES = 32 << 20
#: Never read more than this while probing (a probe should cost < 1 s on NVMe).
MAX_PROBE_BYTES = 512 << 20

BW_SIDECAR = "disk_bandwidth.json"


# -- VRAM ------------------------------------------------------------------


def cuda_available() -> bool:
    return torch.cuda.is_available()


def _mem_get_info(device: torch.device | int | None) -> tuple[int, int]:
    if not torch.cuda.is_available():
        return (0, 0)
    index = device.index if isinstance(device, torch.device) else device
    free, total = torch.cuda.mem_get_info(index)
    return int(free), int(total)


def free_vram_bytes(device: torch.device | int | None = None) -> int:
    """Free VRAM as the driver sees it (0 without CUDA).

    Note this is measured *after* the CUDA context exists if torch has already
    initialised — which is what we want, since the context is not ours to spend.
    """
    return _mem_get_info(device)[0]


def total_vram_bytes(device: torch.device | int | None = None) -> int:
    return _mem_get_info(device)[1]


def device_name(device: torch.device | int | None = None) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    index = device.index if isinstance(device, torch.device) else device
    return torch.cuda.get_device_name(index)


# -- RAM -------------------------------------------------------------------


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = (
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    )


def _windows_ram() -> tuple[int, int]:
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
        raise OSError(ctypes.get_last_error())
    return int(status.ullAvailPhys), int(status.ullTotalPhys)


def _linux_ram() -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts:
            values[key] = int(parts[0]) * 1024  # kB
    total = values.get("MemTotal", 0)
    free = values.get("MemAvailable") or (
        values.get("MemFree", 0) + values.get("Cached", 0) + values.get("Buffers", 0)
    )
    return free, total


def _sysconf_ram() -> tuple[int, int]:
    page = os.sysconf("SC_PAGE_SIZE")
    return int(os.sysconf("SC_AVPHYS_PAGES") * page), int(os.sysconf("SC_PHYS_PAGES") * page)


def _ram() -> tuple[int, int]:
    try:
        if sys.platform == "win32":
            return _windows_ram()
        if sys.platform.startswith("linux"):
            return _linux_ram()
        return _sysconf_ram()
    except Exception as e:  # noqa: BLE001 — a failed probe must not kill a run
        logger.warning("Could not probe system RAM (%s); assuming 8 GB total / 4 GB free", e)
        return (4_000_000_000, 8_000_000_000)


def free_ram_bytes() -> int:
    return _ram()[0]


def total_ram_bytes() -> int:
    return _ram()[1]


# -- disk ------------------------------------------------------------------


def free_disk_bytes(path: Path | str) -> int:
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    return shutil.disk_usage(p).free


def probe_disk_bandwidth(
    sample_file: Path | str,
    *,
    cache_dir: Path | str | None = None,
    force: bool = False,
) -> float:
    """Bytes/second, measured once per shard cache and cached in a sidecar.

    Reads up to MAX_PROBE_BYTES of `sample_file` with buffering disabled. The
    result is a *floor*: a second read of the same file would be served from
    the page cache, which is exactly why we persist the first measurement
    instead of re-probing.
    """
    sample = Path(sample_file)
    sidecar = Path(cache_dir) / BW_SIDECAR if cache_dir is not None else None

    if sidecar is not None and sidecar.is_file() and not force:
        try:
            cached = json.loads(sidecar.read_text(encoding="utf-8"))
            return float(cached["bytes_per_s"])
        except (OSError, ValueError, KeyError):
            logger.debug("Ignoring unreadable disk-bandwidth sidecar %s", sidecar)

    if not sample.is_file():
        logger.info("No sample file to probe; assuming %.1f GB/s", DEFAULT_DISK_BW / 1e9)
        return DEFAULT_DISK_BW

    size = sample.stat().st_size
    to_read = min(size, MAX_PROBE_BYTES)
    buf = bytearray(1 << 22)
    view = memoryview(buf)
    read = 0
    start = time.perf_counter()
    try:
        with open(sample, "rb", buffering=0) as f:
            while read < to_read:
                n = f.readinto(view[: min(len(buf), to_read - read)])
                if not n:
                    break
                read += n
    except OSError as e:
        logger.warning("Disk probe failed (%s); assuming %.1f GB/s", e, DEFAULT_DISK_BW / 1e9)
        return DEFAULT_DISK_BW
    elapsed = time.perf_counter() - start

    if read < MIN_PROBE_BYTES or elapsed <= 0:
        # Too small to distinguish disk from page cache — do NOT cache this.
        logger.debug(
            "Disk sample only %.1f MB (<%.0f MB); assuming %.1f GB/s",
            read / 1e6,
            MIN_PROBE_BYTES / 1e6,
            DEFAULT_DISK_BW / 1e9,
        )
        return DEFAULT_DISK_BW

    bw = read / elapsed
    logger.info("Disk read bandwidth: %.2f GB/s (%.0f MB sample)", bw / 1e9, read / 1e6)
    if sidecar is not None:
        try:
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text(
                json.dumps(
                    {"bytes_per_s": bw, "sample_bytes": read, "sample": sample.name},
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as e:
            logger.debug("Could not write disk-bandwidth sidecar: %s", e)
    return bw


# -- aggregate -------------------------------------------------------------


@dataclass(frozen=True)
class HardwareProfile:
    """What the budget solver needs to know about this machine."""

    vram_bytes: int
    ram_bytes: int
    disk_bw_bytes_s: float
    device: str = "cuda"
    gpu_name: str = ""
    vram_total_bytes: int = 0
    ram_total_bytes: int = 0

    @classmethod
    def probe(
        cls,
        device: torch.device | str | None = None,
        *,
        sample_file: Path | str | None = None,
        cache_dir: Path | str | None = None,
    ) -> HardwareProfile:
        dev = torch.device(device) if device is not None else None
        cuda = torch.cuda.is_available() and (dev is None or dev.type == "cuda")
        free_v, total_v = _mem_get_info(dev) if cuda else (0, 0)
        free_r, total_r = _ram()
        bw = (
            probe_disk_bandwidth(sample_file, cache_dir=cache_dir)
            if sample_file is not None
            else DEFAULT_DISK_BW
        )
        return cls(
            vram_bytes=free_v,
            ram_bytes=free_r,
            disk_bw_bytes_s=bw,
            device="cuda" if cuda else "cpu",
            gpu_name=device_name(dev) if cuda else platform.processor() or "cpu",
            vram_total_bytes=total_v,
            ram_total_bytes=total_r,
        )

    def describe(self) -> str:
        lines = [
            f"device        {self.device}  {self.gpu_name}",
            f"VRAM          {self.vram_bytes / 1e9:.2f} GB free "
            f"/ {self.vram_total_bytes / 1e9:.2f} GB total",
            f"RAM           {self.ram_bytes / 1e9:.2f} GB free "
            f"/ {self.ram_total_bytes / 1e9:.2f} GB total",
            f"disk read     {self.disk_bw_bytes_s / 1e9:.2f} GB/s",
        ]
        return "\n".join(lines)
