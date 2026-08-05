"""M5 (first slice): Qwen-Image adapter against a real-shaped tensor layout.

Names mirror the verified Qwen/Qwen-Image index structure (60 contiguous
transformer_blocks + 6 resident modules), shrunk to 6 blocks for speed.
"""

import pytest
import torch
from tests.conftest import write_checkpoint

from aircanvas.adapters import resolve
from aircanvas.adapters.qwen_image import QwenImageAdapter
from aircanvas.sharding.splitter import split_model

RESIDENTS = (
    "img_in.weight",
    "txt_in.weight",
    "txt_norm.weight",
    "time_text_embed.timestep_embedder.linear_1.weight",
    "norm_out.linear.weight",
    "proj_out.weight",
)


def qwen_like_names(n_blocks: int = 6) -> list[str]:
    names: list[str] = []
    for i in range(n_blocks):
        p = f"transformer_blocks.{i}."
        names += [
            p + "attn.add_k_proj.weight",
            p + "attn.norm_added_q.weight",
            p + "attn.to_out.0.weight",
            p + "img_mlp.net.0.proj.weight",
            p + "txt_mlp.net.2.weight",
            p + "img_mod.1.weight",
        ]
    return names + list(RESIDENTS)


def test_resolve_qwen_image() -> None:
    assert isinstance(resolve("QwenImageTransformer2DModel"), QwenImageAdapter)


def test_qwen_plan_uniform_blocks() -> None:
    plan = QwenImageAdapter().block_plan(qwen_like_names())
    assert plan.block_lists == ("transformer_blocks",)
    assert plan.n_blocks == 6
    assert set(plan.resident_tensors) == set(RESIDENTS)
    # every block claims the same tensor structure (uniformity)
    shapes = {
        tuple(n.removeprefix(b + ".") for n in plan.tensors_by_block[b]) for b in plan.block_names
    }
    assert len(shapes) == 1


def test_qwen_token_count() -> None:
    assert QwenImageAdapter().token_count(1024, 1024) == 4096
    assert QwenImageAdapter().token_count(2048, 1024) == 8192


def test_qwen_like_checkpoint_splits(tmp_path) -> None:
    g = torch.Generator().manual_seed(0)
    tensors = {n: torch.randn(4, 4, generator=g) for n in qwen_like_names()}
    checkpoint = write_checkpoint(
        tmp_path / "model", tensors, sharded=True, class_name="QwenImageTransformer2DModel"
    )
    manifest = split_model(str(checkpoint), cache_dir=tmp_path / "cache", compute_dtype=None)
    assert manifest.adapter == "qwen_image"
    assert len(manifest.blocks) == 6
    assert manifest.blocks[0].name == "transformer_blocks.0"


@pytest.mark.network
def test_real_qwen_image_index_matches_adapter() -> None:
    """Validate the adapter against the real Qwen/Qwen-Image index (small JSON
    download; deselected by default via the network marker)."""
    import json

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "Qwen/Qwen-Image", "transformer/diffusion_pytorch_model.safetensors.index.json"
    )
    with open(path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    plan = QwenImageAdapter().block_plan(list(weight_map))
    assert plan.block_lists == ("transformer_blocks",)
    assert plan.n_blocks == 60
    assert all(
        r.split(".")[0]
        in {"img_in", "txt_in", "txt_norm", "time_text_embed", "norm_out", "proj_out"}
        for r in plan.resident_tensors
    )
