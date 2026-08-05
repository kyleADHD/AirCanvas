"""HunyuanVideo adapter (M7): HunyuanVideoTransformer3DModel.

12.8B: 20 dual-stream (`transformer_blocks`) + 40 single-stream
(`single_transformer_blocks`), hidden 3072. Guidance-distilled: NO CFG batch
doubling. TEs: Llava-Llama-3-8B (~15GB bf16 — the heaviest TE anywhere; must
load-run-evict or run on CPU) + CLIP-L. VAE: 3D causal, native enable_tiling.
Tokens @720p x 129f = 118,800 -> FFN transient ~2.9GB.
"""
