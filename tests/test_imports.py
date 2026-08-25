"""M0 smoke test: the package and every stub module import cleanly on CPU."""

import importlib

import pytest

MODULES = [
    "aircanvas",
    "aircanvas.api",
    "aircanvas.cli",
    "aircanvas.config",
    "aircanvas.lora",
    "aircanvas.sharding.manifest",
    "aircanvas.sharding.splitter",
    "aircanvas.sharding.quant",
    "aircanvas.streaming.engine",
    "aircanvas.streaming.prefetch",
    "aircanvas.streaming.residency",
    "aircanvas.runtime.orchestrator",
    "aircanvas.runtime.text_encoders",
    "aircanvas.runtime.vae",
    "aircanvas.adapters.base",
    "aircanvas.utils.memory",
    "aircanvas.utils.hw",
]


@pytest.mark.parametrize("module", MODULES)
def test_imports(module: str) -> None:
    importlib.import_module(module)


def test_parse_size() -> None:
    from aircanvas.config import parse_size

    assert parse_size("6GB") == 6_000_000_000
    assert parse_size("512MB") == 512_000_000
    assert parse_size("2GiB") == 2 * 2**30
    assert parse_size(123) == 123
    with pytest.raises(ValueError):
        parse_size("lots")
