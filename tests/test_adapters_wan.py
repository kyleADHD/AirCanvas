"""M6 (first slice): Wan adapter against the real-shaped tensor layout.

Names mirror the verified Wan-AI/*-Diffusers index structure (30/40 uniform
`blocks` + 4 resident modules), shrunk to 6 blocks for speed.
"""

import pytest
import torch
from tests.conftest import write_checkpoint

from aircanvas.adapters import resolve
from aircanvas.adapters.wan import WanAdapter
from aircanvas.sharding.splitter import split_model

RESIDENTS = (
    "patch_embedding.weight",
    "condition_embedder.text_embedder.linear_1.weight",
    "condition_embedder.time_embedder.linear_1.weight",
    "proj_out.weight",
    "scale_shift_table",
)


def wan_like_names(n_blocks: int = 6) -> list[str]:
    names: list[str] = []
    for i in range(n_blocks):
        p = f"blocks.{i}."
        names += [
            p + "attn1.to_q.weight",
            p + "attn1.norm_q.weight",
            p + "attn2.to_k.bias",
            p + "ffn.net.0.proj.weight",
            p + "norm2.weight",
            p + "scale_shift_table",
        ]
    return names + list(RESIDENTS)


def test_resolve_wan() -> None:
    assert isinstance(resolve("WanTransformer3DModel"), WanAdapter)


def test_wan_plan_structure() -> None:
    plan = WanAdapter().block_plan(wan_like_names())
    assert plan.block_lists == ("blocks",)
    assert plan.n_blocks == 6
    assert set(plan.resident_tensors) == set(RESIDENTS)


def test_wan_token_count_temporal_compression() -> None:
    adapter = WanAdapter()
    # 480x832 at 81 frames: 30*52 spatial x 21 latent frames + text budget
    assert adapter.token_count(832, 480, frames=81) == 30 * 52 * 21 + 512
    # a single frame behaves like an image model
    assert adapter.token_count(832, 480, frames=1) == 30 * 52 + 512
    # frames=0/omitted degrade to 1, never 0
    assert adapter.token_count(832, 480) == 30 * 52 + 512


def test_wan_component_facts() -> None:
    adapter = WanAdapter()
    assert adapter.encode_prompt_outputs == ("prompt_embeds", "negative_prompt_embeds")
    assert not adapter.guidance_distilled


def test_wan_like_checkpoint_splits(tmp_path) -> None:
    g = torch.Generator().manual_seed(0)
    tensors = {n: torch.randn(4, 4, generator=g) for n in wan_like_names()}
    checkpoint = write_checkpoint(
        tmp_path / "model", tensors, sharded=True, class_name="WanTransformer3DModel"
    )
    manifest = split_model(str(checkpoint), cache_dir=tmp_path / "cache", compute_dtype=None)
    assert manifest.adapter == "wan"
    assert len(manifest.blocks) == 6
    assert manifest.blocks[0].name == "blocks.0"


@pytest.mark.network
@pytest.mark.parametrize(
    ("repo", "n_blocks"),
    [
        ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", 30),
        ("Wan-AI/Wan2.1-T2V-14B-Diffusers", 40),
    ],
)
def test_real_wan_index_matches_adapter(repo: str, n_blocks: int) -> None:
    import json

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, "transformer/diffusion_pytorch_model.safetensors.index.json")
    with open(path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    plan = WanAdapter().block_plan(list(weight_map))
    assert plan.block_lists == ("blocks",)
    assert plan.n_blocks == n_blocks
    top = {r.split(".")[0] for r in plan.resident_tensors}
    assert top == {"condition_embedder", "patch_embedding", "proj_out", "scale_shift_table"}
