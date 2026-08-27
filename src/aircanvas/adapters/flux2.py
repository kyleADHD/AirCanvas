"""FLUX.2 klein adapter: Flux2Transformer2DModel.

A different model from FLUX.1 despite the name — different block counts,
different text encoder, different latent geometry — so it gets its own
adapter rather than an extra entry in FluxAdapter.model_classes. Without one
it would fall to GenericAdapter, which discovers the right block lists but
guesses `encode_prompt`'s return tuple and the token count, and the first is
what makes load-run-evict work at all.

Verified against the published configs and diffusers 0.40's own source:

- **Blocks.** `num_layers` double-stream (`transformer_blocks`) then
  `num_single_layers` single-stream (`single_transformer_blocks`). klein-4B is
  5 + 20 = 25; klein-9B is 8 + 24 = 32. Instantiating each config on meta and
  summing parameters reproduces the published checkpoint sizes to the byte
  (7.751 GB and 18.157 GB), so the shard plan below is the real one. Doubles
  are ~2x a single (491/245 MB on 4B, 872/436 MB on 9B), which is what sizes
  the slot pool.
- **encode_prompt** returns `(prompt_embeds, text_ids)`. `text_ids` is
  rebuilt inside `__call__` from the latent geometry and is not an accepted
  kwarg — same shape of answer as FLUX.1, different second element.
- **Text encoder** is a Qwen3 causal LM (8.0 GB bf16 on 4B, 16.4 GB on 9B),
  read at three hidden layers (`text_encoder_out_layers=(9, 18, 27)`). It is
  far too big to sit beside the DiT on a small card, which is exactly the
  load-run-evict case; on 6 GB it wants the CPU TE path.
- **CFG.** `do_classifier_free_guidance` is
  `guidance_scale > 1 and not config.is_distilled`, and both klein releases
  ship `is_distilled: true` — so no negative pass and no second forward.
  (The `-base-` variants are not distilled; the orchestrator's negative
  encode already covers them, and `negative_prompt_embeds` IS an accepted
  `__call__` kwarg, so nothing here needs to change for those.)
- **Latent geometry.** The FLUX.2 VAE has four `block_out_channels` (8x
  spatial) and 32 latent channels, and the pipeline packs 2x2 into the
  transformer's 128 in_channels — so tokens are (H/16) x (W/16), the base
  stride. 1024^2 = 4096 image tokens + up to 512 text.
"""

from __future__ import annotations

from aircanvas.adapters.base import ModelAdapter, register


@register
class Flux2Adapter(ModelAdapter):
    key = "flux2"
    model_classes = ("Flux2Transformer2DModel",)
    expected_block_lists = ("transformer_blocks", "single_transformer_blocks")
    encode_prompt_outputs = ("prompt_embeds", "text_ids")
    text_tokens = 512  # encode_prompt max_sequence_length default
    guidance_distilled = True  # klein is distilled: CFG is off, one forward per step
