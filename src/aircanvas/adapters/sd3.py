"""SD3 / SD3.5 adapter (M8): SD3Transformer2DModel.

SD3.5-Large: 8B, 38 MMDiT `transformer_blocks`, hidden 2432; SD3.5-Medium is
MMDiT-X (extra self-attn in early blocks — per-block sizes vary, which the
manifest records per shard anyway). Repos are LICENSE-GATED on the Hub, so no
network-marked index test — the layout is pinned from the model cards.

encode_prompt returns the 4-tuple (prompt_embeds, negative_prompt_embeds,
pooled_prompt_embeds, negative_pooled_prompt_embeds) — verified against
diffusers 0.39 source; pass negative_prompt via encode_kwargs so one call
produces all four. TEs: CLIP-L + OpenCLIP bigG + T5-XXL; T5 is droppable
(~9.5 GB, quality tradeoff) by passing text_encoder_3=None at load.
"""

from __future__ import annotations

from aircanvas.adapters.base import ModelAdapter, register


@register
class SD3Adapter(ModelAdapter):
    key = "sd3"
    model_classes = ("SD3Transformer2DModel",)
    expected_block_lists = ("transformer_blocks",)
    encode_prompt_outputs = (
        "prompt_embeds",
        "negative_prompt_embeds",
        "pooled_prompt_embeds",
        "negative_pooled_prompt_embeds",
    )
    text_tokens = 77 + 256  # CLIP window + T5 budget
    guidance_distilled = False
