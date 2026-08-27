"""What the Studio knows about this machine and what is already on its disk.

Everything here is read from the live box — `torch.cuda.mem_get_info`, the
platform's own RAM call, a timed read of a real shard, the shard cache's own
manifests. Nothing is remembered from a previous session except the disk
bandwidth sidecar, which is persisted deliberately: a second read of the same
file is served from the page cache and would measure RAM, not the disk
(utils/hw.py).

The one thing that is *not* probed on demand is the disk. It costs up to
512 MB of reads, so the UI asks for it explicitly ("Re-run probe") and gets
the sidecar value otherwise, with `probed_at` so the screen can say how old
the measurement is instead of implying it is live.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from aircanvas.config import Compression, cache_root
from aircanvas.sharding.manifest import MANIFEST_NAME, Manifest
from aircanvas.utils.hw import (
    BW_SIDECAR,
    DEFAULT_DISK_BW,
    HardwareProfile,
    free_disk_bytes,
    probe_disk_bandwidth,
)

logger = logging.getLogger(__name__)

GB = 1_000_000_000


@dataclass(frozen=True)
class MachineProbe:
    """One reading of the machine, as both screens present it."""

    gpu: str
    device: str
    vram_free_bytes: int
    vram_total_bytes: int
    ram_free_bytes: int
    ram_total_bytes: int
    disk_bw_bytes_s: float
    disk_probed: bool
    disk_probed_at: float | None
    cache_path: str
    cache_free_bytes: int

    @property
    def ram_committed_pct(self) -> int:
        if self.ram_total_bytes <= 0:
            return 0
        used = self.ram_total_bytes - self.ram_free_bytes
        return round(100 * used / self.ram_total_bytes)

    def hardware(self) -> HardwareProfile:
        """The solver's view of this reading."""
        return HardwareProfile(
            vram_bytes=self.vram_free_bytes,
            ram_bytes=self.ram_free_bytes,
            disk_bw_bytes_s=self.disk_bw_bytes_s,
            device=self.device,
            gpu_name=self.gpu,
            vram_total_bytes=self.vram_total_bytes,
            ram_total_bytes=self.ram_total_bytes,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "gpu": self.gpu,
            "device": self.device,
            "vramFreeBytes": self.vram_free_bytes,
            "vramTotalBytes": self.vram_total_bytes,
            "ramFreeBytes": self.ram_free_bytes,
            "ramTotalBytes": self.ram_total_bytes,
            "ramCommittedPct": self.ram_committed_pct,
            "diskBytesPerSecond": self.disk_bw_bytes_s,
            "diskProbed": self.disk_probed,
            "diskProbedAt": self.disk_probed_at,
            "cachePath": self.cache_path,
            "cacheFreeBytes": self.cache_free_bytes,
        }


def _largest_shard(root: Path) -> Path | None:
    """The biggest shard already on disk — the only honest probe sample."""
    try:
        shards = list(root.rglob("block_*.safetensors"))
    except OSError:  # pragma: no cover - unreadable cache root
        return None
    return max(shards, key=lambda p: p.stat().st_size) if shards else None


def _sidecar_age(root: Path) -> tuple[bool, float | None]:
    """(was the bandwidth measured, when) from the persisted probe sidecar."""
    for sidecar in root.rglob(BW_SIDECAR):
        try:
            return True, sidecar.stat().st_mtime
        except OSError:  # pragma: no cover
            continue
    return False, None


def probe(*, probe_disk: bool = False, cache_dir: Path | None = None) -> MachineProbe:
    """Read the machine. `probe_disk=True` re-measures the disk for real."""
    root = cache_dir or cache_root()
    root.mkdir(parents=True, exist_ok=True)
    hardware = HardwareProfile.probe()

    sample = _largest_shard(root)
    probed, probed_at = _sidecar_age(root)
    bandwidth = hardware.disk_bw_bytes_s
    if sample is not None:
        bandwidth = probe_disk_bandwidth(sample, cache_dir=sample.parent, force=probe_disk)
        if probe_disk:
            probed, probed_at = True, time.time()
        elif not probed:
            probed, probed_at = _sidecar_age(root)
    elif probe_disk:
        logger.info("No shard cache to probe yet; disk bandwidth stays assumed")
        bandwidth = DEFAULT_DISK_BW

    return MachineProbe(
        gpu=hardware.gpu_name or hardware.device,
        device=hardware.device,
        vram_free_bytes=hardware.vram_bytes,
        vram_total_bytes=hardware.vram_total_bytes,
        ram_free_bytes=hardware.ram_bytes,
        ram_total_bytes=hardware.ram_total_bytes,
        disk_bw_bytes_s=bandwidth,
        disk_probed=probed,
        disk_probed_at=probed_at,
        cache_path=str(root),
        cache_free_bytes=free_disk_bytes(root),
    )


# -- what is already installed ---------------------------------------------


@dataclass(frozen=True)
class InstalledModel:
    """A shard cache that is complete enough to generate from."""

    cache_dir: str
    source: str
    model_class: str
    adapter: str
    compression: Compression
    compute_dtype: str
    subfolder: str
    n_blocks: int
    disk_bytes: int
    largest_block_bytes: int
    complete: bool
    from_gguf: str | None
    modified_at: float

    @property
    def fmt(self) -> str:
        return self.compression or "bf16"

    def as_dict(self) -> dict[str, object]:
        return {
            "cacheDir": self.cache_dir,
            "source": self.source,
            "modelClass": self.model_class,
            "adapter": self.adapter,
            "format": self.fmt,
            "computeDtype": self.compute_dtype,
            "subfolder": self.subfolder,
            "blocks": self.n_blocks,
            "diskBytes": self.disk_bytes,
            "largestBlockBytes": self.largest_block_bytes,
            "complete": self.complete,
            "fromGguf": self.from_gguf,
            "modifiedAt": self.modified_at,
        }


def _dir_bytes(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:  # pragma: no cover - file vanished mid-scan
            continue
    return total


def installed_models(cache_dir: Path | None = None) -> list[InstalledModel]:
    """Every shard cache under the AirCanvas root, complete or interrupted.

    Interrupted caches are listed too (`complete=False`): a half-finished
    split is exactly what the setup screen has to offer a Resume for, and
    hiding it would make the disk usage in Settings a lie.
    """
    root = cache_dir or cache_root()
    found: list[InstalledModel] = []
    if not root.is_dir():
        return found
    for manifest_path in sorted(root.rglob(MANIFEST_NAME)):
        cache = manifest_path.parent
        try:
            manifest = Manifest.load(cache)
        except (OSError, ValueError, KeyError, TypeError) as e:
            logger.warning("Ignoring unreadable shard cache %s (%s)", cache, e)
            continue
        found.append(
            InstalledModel(
                cache_dir=str(cache),
                source=manifest.source,
                model_class=manifest.model_class,
                adapter=manifest.adapter,
                compression=manifest.compression,
                compute_dtype=manifest.compute_dtype,
                subfolder=manifest.subfolder,
                n_blocks=len(manifest.blocks),
                disk_bytes=_dir_bytes(cache),
                largest_block_bytes=max((b.n_bytes for b in manifest.blocks), default=0),
                complete=manifest.is_complete(cache),
                from_gguf=manifest.gguf_file,
                modified_at=manifest_path.stat().st_mtime,
            )
        )
    return found


def cache_summary(cache_dir: Path | None = None) -> dict[str, object]:
    """Settings' shard-cache card: location, total, free, and the breakdown."""
    root = cache_dir or cache_root()
    models = installed_models(root)
    return {
        "path": str(root),
        "totalBytes": sum(m.disk_bytes for m in models),
        "freeBytes": free_disk_bytes(root),
        "models": [m.as_dict() for m in models],
    }


# -- Hugging Face ----------------------------------------------------------


def hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def hf_token() -> str | None:
    """The token this machine would actually use, or None."""
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value.strip()
    token_file = hf_home() / "token"
    try:
        return token_file.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def mask_token(token: str | None) -> str | None:
    """`hf_••••2f9d` — enough to recognise, not enough to use."""
    if not token:
        return None
    tail = token[-4:]
    prefix = "hf_" if token.startswith("hf_") else ""
    return f"{prefix}{'•' * 24}{tail}"


def hf_account() -> dict[str, object]:
    """Token presence and, if the Hub is reachable, who it belongs to."""
    token = hf_token()
    info: dict[str, object] = {"authenticated": bool(token), "masked": mask_token(token)}
    if not token:
        return info
    try:  # a name is a nicety; never block the screen on a network call
        from huggingface_hub import HfApi

        info["user"] = HfApi().whoami(token=token).get("name")
    except Exception as e:  # noqa: BLE001 — offline is the normal case here
        logger.debug("Could not resolve the Hugging Face account name: %s", e)
    return info


def repo_cache_bytes(repo_id: str) -> tuple[int, bool]:
    """(bytes of `repo_id` already fetched, is a download in flight).

    Read from the Hub cache's own layout — completed blobs plus the
    `.incomplete` partials huggingface_hub writes while downloading — so a
    paused or interrupted download reports the bytes it really has instead of
    a percentage we invented.
    """
    folder = hf_home() / "hub" / ("models--" + repo_id.replace("/", "--"))
    blobs = folder / "blobs"
    if not blobs.is_dir():
        return 0, False
    total = 0
    partial = False
    for blob in blobs.iterdir():
        try:
            total += blob.stat().st_size
        except OSError:  # pragma: no cover - vanished mid-download
            continue
        if blob.name.endswith(".incomplete"):
            partial = True
    return total, partial


def read_json(path: Path) -> dict[str, object]:
    """Small helper: a JSON file, or {} if it is missing or malformed."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}
