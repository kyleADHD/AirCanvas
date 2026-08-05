"""Text-encoder strategies: load-run-evict, group-offloaded, or CPU (M4).

Quirks handled here (RESEARCH.md §7):
- T5's forward does its own dtype casting — breaks naive fp8 storage; keep T5
  bf16 or special-case it.
- SD3.5's T5 is droppable entirely (quality tradeoff, saves ~9.5 GB).
- Qwen-Image's TE is a full Qwen2.5-VL run with a task system prompt.
"""

from __future__ import annotations
