"""Qwen-Image adapter (M5): QwenImageTransformer2DModel.

20.4B: 60 IDENTICAL double-stream MMDiT blocks (~340M/~680MB bf16 each) — the
ideal uniform shard plan. TE: Qwen2.5-VL-7B (~16.6GB bf16) run as a VLM with a
task system prompt — must be group-offloaded or CPU-run, never co-resident.
VAE: Wan-2.1 family (16ch).
"""
