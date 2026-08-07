"""Wan 2.1 / 2.2 adapter (M6): WanTransformer3DModel.

Verified against the live checkpoint indexes (Wan-AI/*-Diffusers): 1.3B has
30 and 14B has 40 IDENTICAL blocks in `blocks` (~350M each on 14B); residents
are condition_embedder, patch_embedding, proj_out, scale_shift_table. One
shard plan covers the whole family (SkyReels-V2, CausVid, VACE and Phantom
share the backbone).

Latent geometry: Wan-VAE compresses 4x in time and 8x in space; patchify is
(1,2,2) -> spatial stride 16 (base default), temporal frames//4+1. 480x832 at
81 frames = 30*52*21 = 32,760 video tokens — the workload class where block
streaming hides almost entirely behind compute (RESEARCH.md §4).

Runtime facts: WanPipeline.encode_prompt(prompt, negative_prompt, ...)
returns (prompt_embeds, negative_prompt_embeds) in ONE call — pass
negative_prompt via encode_kwargs, not the orchestrator's separate negative
pass. CFG runs as TWO transformer forwards per step (not a batched double),
which the engine's cyclic prefetch schedule follows naturally. TE: UMT5-XXL
5.7B (11.4 GB bf16) — auto-routes to CPU on small cards, cached after. VAE:
AutoencoderKLWan, spatial tiling upstream since diffusers 0.39.

Wan 2.2 A14B (two per-timestep experts in transformer/ + transformer_2/) is
M7: the pipeline swaps `current_model` at the boundary timestep, which the
engine sees as schedule divergence today — expert_groups scheduling lands
there.
"""

from __future__ import annotations

from aircanvas.adapters.base import ModelAdapter, register


@register
class WanAdapter(ModelAdapter):
    key = "wan"
    model_classes = ("WanTransformer3DModel",)
    expected_block_lists = ("blocks",)
    encode_prompt_outputs = ("prompt_embeds", "negative_prompt_embeds")
    text_tokens = 512  # UMT5 budget; conservative for the activation estimate
    guidance_distilled = False  # true CFG, two forwards per step

    def token_count(self, width: int, height: int, frames: int = 1) -> int:
        latent_frames = (max(1, frames) - 1) // 4 + 1  # causal 4x temporal VAE
        spatial = max(1, height // self.latent_stride) * max(1, width // self.latent_stride)
        return spatial * latent_frames + self.text_tokens
