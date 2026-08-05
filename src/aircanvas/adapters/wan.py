"""Wan 2.1 / 2.2 adapter (M6/M7): WanTransformer3DModel.

2.1-14B: 40 identical blocks (~350M, 700MB bf16 / 350MB fp8), hidden 5120.
2.2-A14B: TWO experts (transformer/ + transformer_2/), per-TIMESTEP switch at
boundary t_moe — expert_groups routes shard scheduling; only the active
expert streams. TE: UMT5-XXL 5.7B (11.4GB bf16) — evict after encode.
VAE: AutoencoderKLWan — NO upstream tiling; use runtime.vae aircanvas_wan
tiling. Same backbone serves SkyReels-V2/CausVid/VACE/Phantom.
Tokens: frames//4+1 x H/16 x W/16 (e.g. 720p x 81f = 75,600).
"""
