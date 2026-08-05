"""Shard-cache manifest: schema, (de)serialization, completeness checks.

manifest.json is written LAST during a split; each shard file gets a `.done`
marker (written after an atomic temp->rename). A missing marker means that
shard is re-split. The manifest records the source so a stale/incompatible
cache is detected instead of silently reused.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from aircanvas.config import Compression

MANIFEST_VERSION = 1
MANIFEST_NAME = "manifest.json"
DONE_SUFFIX = ".done"


@dataclass(frozen=True)
class BlockShard:
    name: str  # module path in the model, e.g. "transformer_blocks.0"
    file: str  # relative filename, e.g. "block_0000.safetensors"
    n_bytes: int
    sha256: str | None = None


@dataclass(frozen=True)
class Manifest:
    source: str  # repo id or local path, as given
    revision: str | None
    subfolder: str
    model_class: str  # e.g. "FluxTransformer2DModel"
    adapter: str  # adapter key, e.g. "flux"
    compression: Compression
    compute_dtype: str  # e.g. "bfloat16", or "source" (no cast)
    blocks: tuple[BlockShard, ...] = ()
    resident_file: str = "resident.safetensors"
    resident_bytes: int = 0
    # Wan 2.2: {"high_noise": [block indices], "low_noise": [...]} with a
    # per-timestep switching rule; None for dense models.
    expert_groups: dict[str, list[int]] | None = None
    manifest_version: int = MANIFEST_VERSION

    def save(self, cache_dir: Path) -> None:
        payload = asdict(self)
        payload["blocks"] = [asdict(b) for b in self.blocks]
        tmp = cache_dir / (MANIFEST_NAME + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, cache_dir / MANIFEST_NAME)

    @classmethod
    def load(cls, cache_dir: Path) -> Manifest:
        data = json.loads((cache_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
        version = data.get("manifest_version")
        if version != MANIFEST_VERSION:
            raise ValueError(
                f"Unsupported manifest version {version} (expected {MANIFEST_VERSION}) "
                f"in {cache_dir} — re-split with `aircanvas split`."
            )
        data["blocks"] = tuple(BlockShard(**b) for b in data["blocks"])
        return cls(**data)

    def shard_files(self) -> tuple[str, ...]:
        return tuple(b.file for b in self.blocks) + (self.resident_file,)

    def is_complete(self, cache_dir: Path) -> bool:
        """True iff every shard file and its .done marker exist."""
        return all(
            (cache_dir / f).is_file() and (cache_dir / (f + DONE_SUFFIX)).is_file()
            for f in self.shard_files()
        )

    def compatible_with(
        self, *, compression: Compression, compute_dtype: str, model_class: str
    ) -> bool:
        return (
            self.compression == compression
            and self.compute_dtype == compute_dtype
            and self.model_class == model_class
        )
