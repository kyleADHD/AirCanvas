"""M1: generic block discovery + adapter resolution."""

import pytest

from aircanvas.adapters import AdapterError, GenericAdapter, resolve
from aircanvas.adapters.base import _discover_block_lists
from aircanvas.adapters.flux import FluxAdapter


def test_resolve_flux() -> None:
    assert isinstance(resolve("FluxTransformer2DModel"), FluxAdapter)


def test_resolve_unknown_falls_back_to_generic() -> None:
    adapter = resolve("SomeFutureTransformer3DModel")
    assert isinstance(adapter, GenericAdapter)
    assert adapter.key == "generic"


def test_discovery_outermost_index_only() -> None:
    names = [f"blocks.{i}.attn.to_out.0.weight" for i in range(5)]
    found = _discover_block_lists(names)
    assert list(found) == ["blocks"]
    assert set(found["blocks"]) == set(range(5))


def test_discovery_rejects_short_and_noncontiguous_lists() -> None:
    short = [f"tiny.{i}.w" for i in range(2)]
    gappy = [f"gappy.{i}.w" for i in (0, 1, 2, 4, 5)]
    ok = [f"blocks.{i}.w" for i in range(4)]
    found = _discover_block_lists(short + gappy + ok)
    assert list(found) == ["blocks"]


def test_generic_plan_residents(flux_tensors: dict) -> None:
    plan = GenericAdapter().block_plan(list(flux_tensors))
    assert plan.n_blocks == 10
    assert set(plan.resident_tensors) == {
        "x_embedder.weight",
        "norm_out.linear.weight",
        "proj_out.weight",
    }
    claimed = {n for ns in plan.tensors_by_block.values() for n in ns}
    assert claimed | set(plan.resident_tensors) == set(flux_tensors)


def test_flux_plan_order_is_execution_order(flux_tensors: dict) -> None:
    # Sorted names put single_transformer_blocks first alphabetically; the
    # flux adapter must still plan double-stream blocks first.
    plan = FluxAdapter().block_plan(sorted(flux_tensors))
    assert plan.block_lists == ("transformer_blocks", "single_transformer_blocks")
    assert plan.block_names[0] == "transformer_blocks.0"
    assert plan.block_names[-1] == "single_transformer_blocks.5"


def test_named_adapter_accepts_short_block_lists() -> None:
    # Tiny test checkpoints (e.g. hf-internal-testing/tiny-flux-pipe) have
    # fewer than MIN_BLOCKS blocks per list; a named adapter knows its lists
    # are block lists and must not be subject to the generic-discovery guard.
    names = [f"transformer_blocks.{i}.attn.to_q.weight" for i in range(2)] + [
        f"single_transformer_blocks.{i}.proj_mlp.weight" for i in range(2)
    ]
    plan = FluxAdapter().block_plan(names)
    assert plan.n_blocks == 4
    # Generic discovery on the same names finds nothing streamable.
    with pytest.raises(AdapterError):
        GenericAdapter().block_plan(names)


def test_flux_plan_missing_list_raises() -> None:
    names = [f"transformer_blocks.{i}.w" for i in range(4)]
    with pytest.raises(AdapterError, match="single_transformer_blocks"):
        FluxAdapter().block_plan(names)


def test_no_blocks_raises() -> None:
    with pytest.raises(AdapterError, match="No streamable block"):
        GenericAdapter().block_plan(["x_embedder.weight", "proj_out.weight"])
