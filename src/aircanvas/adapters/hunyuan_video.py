"""HunyuanVideo adapter (M7): HunyuanVideoTransformer3DModel.

12.8B: 20 dual-stream (`transformer_blocks`) + 40 single-stream
(`single_transformer_blocks`), hidden 3072 — FLUX's two-species layout at
video scale. GUIDANCE-DISTILLED: no CFG batch doubling, one forward per step.

encode_prompt returns (prompt_embeds, pooled_prompt_embeds,
prompt_attention_mask) — verified against diffusers 0.39 source. The TE is
Llava-Llama-3-8B (~15 GB bf16): unloadable on small-RAM boxes by stock
loaders — split and STREAM it (the UMT5 recipe from M6; see
docs/ROADMAP.md M6 notes).

Latent geometry: 3D causal VAE, 4x time / 8x space, patch 2x2x1 ->
latent_stride 16, temporal frames//4+1. 720p x 129f = 118,800 tokens; the
~2.9 GB FFN transient, not weights, is the VRAM floor at that size.
"""

from __future__ import annotations

from aircanvas.adapters.base import ModelAdapter, register


@register
class HunyuanVideoAdapter(ModelAdapter):
    key = "hunyuan_video"
    model_classes = ("HunyuanVideoTransformer3DModel",)
    expected_block_lists = ("transformer_blocks", "single_transformer_blocks")
    encode_prompt_outputs = ("prompt_embeds", "pooled_prompt_embeds", "prompt_attention_mask")
    text_tokens = 256  # Llava template budget
    guidance_distilled = True

    def token_count(self, width: int, height: int, frames: int = 1) -> int:
        latent_frames = (max(1, frames) - 1) // 4 + 1
        spatial = max(1, height // self.latent_stride) * max(1, width // self.latent_stride)
        return spatial * latent_frames + self.text_tokens
