"""M7: lazy per-timestep expert handover (Wan 2.2's dual-DiT pattern).

The mechanism under test: engine A streams the high-noise expert; the first
forward of the low-noise expert closes A (weights, pools, residents all
released) and builds B's engine inside the pre-hook — one-way, one-shot.
"""

from pathlib import Path
from types import SimpleNamespace

import torch
from tests.conftest import ToyDiT, write_checkpoint

from aircanvas.runtime.orchestrator import arm_expert_handover
from aircanvas.sharding.splitter import split_model
from aircanvas.streaming.engine import StreamingEngine


def _toy_manifest(tmp_path: Path, name: str, seed: int):
    torch.manual_seed(seed)
    reference = ToyDiT().eval()
    tensors = {k: v.detach().clone() for k, v in reference.state_dict().items()}
    checkpoint = write_checkpoint(
        tmp_path / name, tensors, sharded=False, class_name="ToyTransformer2DModel"
    )
    cache = tmp_path / f"{name}_cache"
    manifest = split_model(str(checkpoint), cache_dir=cache, compute_dtype=None)
    return reference, manifest, cache


def test_expert_handover_switches_engines(tmp_path: Path) -> None:
    ref_a, manifest_a, cache_a = _toy_manifest(tmp_path, "expert_a", seed=1)
    ref_b, manifest_b, cache_b = _toy_manifest(tmp_path, "expert_b", seed=2)
    x = torch.randn(2, 8, generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        expected_a, expected_b = ref_a(x), ref_b(x)

    with torch.device("meta"):
        model_a, model_b = ToyDiT().eval(), ToyDiT().eval()
    pipe = SimpleNamespace(transformer=model_a, transformer_2=model_b)

    engines = {
        "transformer": StreamingEngine(model_a, manifest_a, cache_a, device="cpu"),
    }
    hooks = arm_expert_handover(
        pipe,
        {"transformer_2": (manifest_b, cache_b)},
        engines,
        device=torch.device("cpu"),
    )
    try:
        with torch.no_grad():
            # high-noise phase: expert A streams, B untouched
            assert torch.equal(model_a(x), expected_a)
            assert all(p.is_meta for p in model_b.parameters())

            # boundary: B's first forward must close A and stream correctly
            assert torch.equal(model_b(x), expected_b)
        assert list(engines) == ["transformer_2"]
        assert all(p.is_meta for p in model_a.parameters()), "expert A must be fully released"

        with torch.no_grad():  # steady state on B, hook is one-shot and gone
            assert torch.equal(model_b(x), expected_b)
    finally:
        for h in hooks:
            h.remove()
        for eng in engines.values():
            eng.close()


def test_handover_skips_missing_modules(tmp_path: Path) -> None:
    _, manifest, cache = _toy_manifest(tmp_path, "solo", seed=3)
    pipe = SimpleNamespace(transformer=object())
    hooks = arm_expert_handover(
        pipe, {"transformer_2": (manifest, cache)}, {}, device=torch.device("cpu")
    )
    assert hooks == []
