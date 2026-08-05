"""CogVideoX adapter (M8): CogVideoXTransformer3DModel.

2B (30 blocks, hidden 1920) / 5B (42 blocks, hidden 3072, 11.14GB bf16).
TE: T5-XXL. VAE: native tiling + slicing — best-behaved diffusers citizen;
serves as the correctness/CI target for the video path.
"""
