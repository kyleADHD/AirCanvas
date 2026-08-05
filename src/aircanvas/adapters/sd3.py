"""SD3 / SD3.5 adapter (M8): SD3Transformer2DModel.

SD3.5-Large: 8B, 38 MMDiT blocks, hidden 2432. Three TEs: CLIP-L + OpenCLIP
bigG + T5-XXL; T5 is DROPPABLE (quality tradeoff, saves ~9.5GB) — expose via
ComponentStrategy.droppable_encoders. SD3.5-Medium has MMDiT-X dual self-attn
in early blocks (blocks not perfectly uniform — plan per-block sizes from the
checkpoint, don't assume).
"""
