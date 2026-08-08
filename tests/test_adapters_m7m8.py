"""M7/M8 adapters: HunyuanVideo, SD3, CogVideoX.

Offline tests pin structure and component facts; network tests validate
against live indexes where repos are public (SD3.5 is license-gated, so its
layout is pinned from the model card instead).
"""

import pytest

from aircanvas.adapters import resolve
from aircanvas.adapters.cogvideox import CogVideoXAdapter
from aircanvas.adapters.hunyuan_video import HunyuanVideoAdapter
from aircanvas.adapters.sd3 import SD3Adapter


def test_resolution() -> None:
    assert isinstance(resolve("HunyuanVideoTransformer3DModel"), HunyuanVideoAdapter)
    assert isinstance(resolve("SD3Transformer2DModel"), SD3Adapter)
    assert isinstance(resolve("CogVideoXTransformer3DModel"), CogVideoXAdapter)


def test_hunyuan_facts() -> None:
    a = HunyuanVideoAdapter()
    assert a.expected_block_lists == ("transformer_blocks", "single_transformer_blocks")
    assert a.encode_prompt_outputs == (
        "prompt_embeds",
        "pooled_prompt_embeds",
        "prompt_attention_mask",
    )
    assert a.guidance_distilled  # one forward per step, no CFG doubling
    # 720p x 129f: 45*80 spatial x 33 latent frames + text budget
    assert a.token_count(1280, 720, frames=129) == 45 * 80 * 33 + 256


def test_sd3_facts() -> None:
    a = SD3Adapter()
    assert a.encode_prompt_outputs == (
        "prompt_embeds",
        "negative_prompt_embeds",
        "pooled_prompt_embeds",
        "negative_pooled_prompt_embeds",
    )
    assert not a.guidance_distilled


def test_cogvideox_facts() -> None:
    a = CogVideoXAdapter()
    assert a.encode_prompt_outputs == ("prompt_embeds", "negative_prompt_embeds")
    assert a.token_count(720, 480, frames=49) == 30 * 45 * 13 + 226


def _plan_from_hub(repo: str):
    import json

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, "transformer/diffusion_pytorch_model.safetensors.index.json")
    with open(path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    return list(weight_map)


@pytest.mark.network
def test_real_hunyuan_index_matches_adapter() -> None:
    plan = HunyuanVideoAdapter().block_plan(_plan_from_hub("hunyuanvideo-community/HunyuanVideo"))
    assert plan.block_lists == ("transformer_blocks", "single_transformer_blocks")
    assert plan.n_blocks == 20 + 40


@pytest.mark.network
def test_real_cogvideox_index_matches_adapter() -> None:
    plan = CogVideoXAdapter().block_plan(_plan_from_hub("zai-org/CogVideoX-5b"))
    assert plan.block_lists == ("transformer_blocks",)
    assert plan.n_blocks == 42
