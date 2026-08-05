"""Qwen-Image adapter (M5): QwenImageTransformer2DModel.

20.4B MMDiT — the ideal uniform shard plan, verified against the real
checkpoint index (Qwen/Qwen-Image, 1933 tensors): exactly 60 contiguous
double-stream blocks in `transformer_blocks` (~340M / ~680MB bf16 each);
residents are img_in, txt_in, txt_norm, time_text_embed, norm_out, proj_out.

Latent geometry: 16-ch latent, 8x VAE, 2x2 patchify -> latent_stride 16 (the
base default) -> 1024^2 = 4096 image tokens.

TE: Qwen2.5-VL-7B (~16.6GB bf16) run as a VLM with a task system prompt —
must NEVER be co-resident with the DiT; the orchestrator's load-run-evict
handles it, CPU mode recommended below 24GB VRAM. VAE: Wan-2.1 family (16ch).
"""

from __future__ import annotations

from aircanvas.adapters.base import ModelAdapter, register


@register
class QwenImageAdapter(ModelAdapter):
    key = "qwen_image"
    model_classes = ("QwenImageTransformer2DModel",)
    expected_block_lists = ("transformer_blocks",)
    # QwenImagePipeline.encode_prompt returns (prompt_embeds, prompt_embeds_mask)
    # — verified against diffusers 0.39 source; both are accepted __call__ kwargs.
    encode_prompt_outputs = ("prompt_embeds", "prompt_embeds_mask")
    text_tokens = 1024  # encode_prompt max_sequence_length default
    guidance_distilled = False  # true CFG: negative prompt doubles the batch
