"""Qwen-Image adapter (M5): QwenImageTransformer2DModel.

20.4B MMDiT — the ideal uniform shard plan, verified against the real
checkpoint index (Qwen/Qwen-Image, 1933 tensors): exactly 60 contiguous
double-stream blocks in `transformer_blocks` (~340M / ~680MB bf16 each);
residents are img_in, txt_in, txt_norm, time_text_embed, norm_out, proj_out.

Tokens: 16-ch latent, 8x VAE, 2x2 patchify -> (H/16)*(W/16) image tokens
(1024^2 = 4096) + prompt tokens from the TE.

Still M4/M5-dependent (not in this adapter yet): the Qwen2.5-VL-7B text
encoder (~16.6GB bf16) must be group-offloaded or CPU-run — never co-resident
with the DiT; VAE is Wan-2.1 family (16ch).
"""

from __future__ import annotations

from aircanvas.adapters.base import ModelAdapter, register


@register
class QwenImageAdapter(ModelAdapter):
    key = "qwen_image"
    model_classes = ("QwenImageTransformer2DModel",)
    expected_block_lists = ("transformer_blocks",)

    def token_count(self, width: int, height: int, frames: int = 1) -> int:
        return (height // 16) * (width // 16)
