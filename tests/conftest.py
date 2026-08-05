"""Shared fixtures: tiny FLUX-shaped checkpoints on disk (CPU-only, no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

N_DOUBLE = 4
N_SINGLE = 6


def make_flux_like_tensors() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)

    def t() -> torch.Tensor:
        return torch.randn(8, 8, generator=g)

    tensors: dict[str, torch.Tensor] = {}
    for i in range(N_DOUBLE):
        p = f"transformer_blocks.{i}."
        tensors[p + "attn.to_q.weight"] = t()
        tensors[p + "attn.to_out.0.weight"] = t()  # nested numbered module inside a block
        tensors[p + "norm1.linear.weight"] = t()
    for i in range(N_SINGLE):
        p = f"single_transformer_blocks.{i}."
        tensors[p + "proj_mlp.weight"] = t()
        tensors[p + "norm.linear.weight"] = t()
    tensors["x_embedder.weight"] = t()
    tensors["norm_out.linear.weight"] = t()
    tensors["proj_out.weight"] = t()
    return tensors


def write_checkpoint(
    root: Path,
    tensors: dict[str, torch.Tensor],
    *,
    sharded: bool,
    class_name: str = "FluxTransformer2DModel",
) -> Path:
    """Write a diffusers-style transformer/ dir: config.json + weights."""
    src = root / "transformer"
    src.mkdir(parents=True)
    (src / "config.json").write_text(json.dumps({"_class_name": class_name}))
    if not sharded:
        save_file(tensors, str(src / "diffusion_pytorch_model.safetensors"))
        return root
    names = list(tensors)
    half = len(names) // 2
    parts = {
        "diffusion_pytorch_model-00001-of-00002.safetensors": names[:half],
        "diffusion_pytorch_model-00002-of-00002.safetensors": names[half:],
    }
    weight_map: dict[str, str] = {}
    for fname, ns in parts.items():
        save_file({n: tensors[n] for n in ns}, str(src / fname))
        weight_map.update(dict.fromkeys(ns, fname))
    index = {"metadata": {}, "weight_map": weight_map}
    (src / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(index))
    return root


class ToyBlock(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin1 = torch.nn.Linear(dim, dim, bias=False)
        self.lin2 = torch.nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.lin2(torch.tanh(self.lin1(x)))


class ToyDiT(torch.nn.Module):
    """Runnable stand-in for a DiT: resident embed/proj + a streamable
    `blocks` ModuleList (generic-adapter shaped)."""

    def __init__(self, dim: int = 8, n_blocks: int = 6) -> None:
        super().__init__()
        self.x_embedder = torch.nn.Linear(dim, dim, bias=False)
        self.blocks = torch.nn.ModuleList(ToyBlock(dim) for _ in range(n_blocks))
        self.proj_out = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.x_embedder(x)
        for block in self.blocks:
            x = block(x)
        return self.proj_out(x)


@pytest.fixture()
def flux_tensors() -> dict[str, torch.Tensor]:
    return make_flux_like_tensors()


@pytest.fixture()
def flux_checkpoint(tmp_path: Path, flux_tensors: dict[str, torch.Tensor]) -> Path:
    return write_checkpoint(tmp_path / "model", flux_tensors, sharded=False)


@pytest.fixture()
def flux_checkpoint_sharded(tmp_path: Path, flux_tensors: dict[str, torch.Tensor]) -> Path:
    return write_checkpoint(tmp_path / "model_sharded", flux_tensors, sharded=True)
