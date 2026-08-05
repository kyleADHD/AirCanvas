"""Hardware probing for the budget solver (M4).

- free VRAM: torch.cuda.mem_get_info
- free RAM: psutil-free (ctypes GlobalMemoryStatusEx on Windows, /proc/meminfo
  on Linux) to avoid a dependency
- disk read bandwidth: timed real read of the first shard file (~1GB), cached
  per cache-dir in a sidecar file — NOT platform APIs (ARCHITECTURE.md §5)
"""

from __future__ import annotations


def free_vram_bytes() -> int:
    raise NotImplementedError("M4")


def free_ram_bytes() -> int:
    raise NotImplementedError("M4")


def probe_disk_bandwidth(sample_file: str) -> float:
    """Bytes/second, measured once and cached."""
    raise NotImplementedError("M4")
