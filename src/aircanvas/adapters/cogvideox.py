"""CogVideoX adapter (M8): CogVideoXTransformer3DModel.

2B (30 blocks, hidden 1920) / 5B (42 blocks, hidden 3072) in
`transformer_blocks`. The best-behaved diffusers citizen (native VAE tiling
and slicing) — the correctness/CI reference for the video path.

encode_prompt returns (prompt_embeds, negative_prompt_embeds) — verified
against diffusers 0.39 source; pass negative_prompt via encode_kwargs. TE:
T5-XXL. Latents: 4x time / 8x space, patch 2 -> stride 16, frames//4+1.
"""

from __future__ import annotations

from aircanvas.adapters.base import ModelAdapter, register


@register
class CogVideoXAdapter(ModelAdapter):
    key = "cogvideox"
    model_classes = ("CogVideoXTransformer3DModel",)
    expected_block_lists = ("transformer_blocks",)
    encode_prompt_outputs = ("prompt_embeds", "negative_prompt_embeds")
    text_tokens = 226  # T5 max_sequence_length in the pipeline
    guidance_distilled = False

    def token_count(self, width: int, height: int, frames: int = 1) -> int:
        latent_frames = (max(1, frames) - 1) // 4 + 1
        spatial = max(1, height // self.latent_stride) * max(1, width // self.latent_stride)
        return spatial * latent_frames + self.text_tokens
