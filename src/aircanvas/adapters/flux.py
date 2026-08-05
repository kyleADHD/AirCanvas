"""FLUX.1 dev/schnell adapter (M1): FluxTransformer2DModel.

19 double-stream blocks (`transformer_blocks`, ~340M/~680MB bf16 each) run
before 38 single-stream (`single_transformer_blocks`, ~140M/~280MB) — the
expected order pins the shard plan even when tensor names arrive sorted.
TEs: T5-XXL (4.7B) + CLIP-L. dev is guidance-distilled (no CFG batch
doubling); schnell is 4-step distilled -> budget solver should warn re:
streaming overhead. Tokens @1024^2: 4096 image + up to 512 text.

Block counts are deliberately NOT hard-validated: FLUX-architecture variants
(Flex, Chroma-adjacent) ship different depths and should still split.
"""

from __future__ import annotations

from aircanvas.adapters.base import ModelAdapter, register


@register
class FluxAdapter(ModelAdapter):
    key = "flux"
    model_classes = ("FluxTransformer2DModel",)
    expected_block_lists = ("transformer_blocks", "single_transformer_blocks")
    # FluxPipeline.encode_prompt -> (prompt_embeds, pooled_prompt_embeds, text_ids);
    # text_ids is rebuilt inside __call__ and is not an accepted kwarg.
    encode_prompt_outputs = ("prompt_embeds", "pooled_prompt_embeds", "text_ids")
    text_tokens = 512  # T5 max_sequence_length default
    guidance_distilled = True
