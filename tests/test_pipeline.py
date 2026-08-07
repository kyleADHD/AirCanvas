"""M4: AirPipeline — phase-aware orchestration end to end.

The unit tests here use fakes and stay offline. The `network`-marked test is
the real thing: `hf-internal-testing/tiny-flux-pipe` (~1.3 MB, public) goes
through split-on-first-use -> text encode + evict -> streamed DiT denoise ->
VAE decode -> PIL image, on whatever device is available. It is the only test
that proves the pieces compose against actual diffusers internals.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
from torch import nn

from aircanvas.adapters import GenericAdapter
from aircanvas.adapters.flux import FluxAdapter
from aircanvas.runtime import text_encoders as te
from aircanvas.runtime.orchestrator import force_execution_device
from aircanvas.runtime.vae import configure_vae, vae_on_demand

TINY_FLUX = "hf-internal-testing/tiny-flux-pipe"


# -- fakes -----------------------------------------------------------------


class FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(2, 2))


class FakeVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(2, 2))
        self.tiled = False
        self.sliced = False
        self.decoded_on: list[torch.device] = []

    def enable_tiling(self) -> None:
        self.tiled = True

    def enable_slicing(self) -> None:
        self.sliced = True

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        self.decoded_on.append(self.w.device)
        return latents


class FakePipeline:
    """Just enough diffusers surface for the placement logic under test."""

    def __init__(self) -> None:
        self.text_encoder = FakeEncoder()
        self.text_encoder_2 = FakeEncoder()
        self.tokenizer = object()  # not an nn.Module: must be ignored
        self.vae = FakeVAE()
        self.encode_calls: list[dict] = []

    @property
    def components(self) -> dict:
        return {
            "text_encoder": self.text_encoder,
            "text_encoder_2": self.text_encoder_2,
            "tokenizer": self.tokenizer,
            "vae": self.vae,
        }

    @property
    def _execution_device(self) -> torch.device:
        return torch.device("cpu")

    def encode_prompt(self, prompt, device=None, max_sequence_length=77):
        self.encode_calls.append(
            {
                "prompt": prompt,
                "device": device,
                "max_sequence_length": max_sequence_length,
                "encoder_device": self.text_encoder.w.device.type,
            }
        )
        n = len(prompt) if isinstance(prompt, list) else 1
        return (
            torch.ones(n, 4, 8),  # prompt_embeds
            torch.ones(n, 8),  # pooled_prompt_embeds
            torch.zeros(4, 3),  # text_ids — not an accepted __call__ kwarg
        )

    # Explicit signature, like every real diffusers pipeline: that is what lets
    # the orchestrator know `text_ids` and `negative_*` are not accepted here.
    def __call__(self, prompt_embeds=None, pooled_prompt_embeds=None, num_inference_steps=1):
        return {
            "prompt_embeds": prompt_embeds,
            "pooled": pooled_prompt_embeds,
            "num_inference_steps": num_inference_steps,
        }


# -- text-encoder phase ----------------------------------------------------


def test_text_encoder_discovery_skips_non_modules() -> None:
    assert te.text_encoder_names(FakePipeline()) == ("text_encoder", "text_encoder_2")


def test_encode_runs_on_device_then_evicts() -> None:
    pipe = FakePipeline()
    out = te.encode_and_evict(pipe, FluxAdapter(), device=torch.device("cpu"), prompt="a canvas")
    assert set(out.kwargs) == {"prompt_embeds", "pooled_prompt_embeds"}  # text_ids dropped
    assert out.encoders_run == ("text_encoder", "text_encoder_2")
    assert not out.cached
    for name in out.encoders_run:
        assert getattr(pipe, name).w.device.type == "cpu"


def test_encode_passes_only_supported_kwargs() -> None:
    pipe = FakePipeline()
    te.encode_and_evict(
        pipe,
        FluxAdapter(),
        device=torch.device("cpu"),
        prompt="x",
        encode_kwargs={"max_sequence_length": 16, "not_a_real_kwarg": 1},
    )
    call = pipe.encode_calls[0]
    assert call["max_sequence_length"] == 16 and "not_a_real_kwarg" not in call


def test_negative_prompt_is_dropped_when_unsupported() -> None:
    """FakePipeline.__call__ takes no negative_* kwargs, so they must not leak
    into the call — passing an unexpected kwarg would be a TypeError at run."""
    pipe = FakePipeline()
    out = te.encode_and_evict(
        pipe, FluxAdapter(), device=torch.device("cpu"), prompt="a", negative_prompt="b"
    )
    assert not any(k.startswith("negative_") for k in out.kwargs)
    assert len(pipe.encode_calls) == 2  # still encoded, just not accepted


def test_embedding_disk_cache_skips_the_encoders(tmp_path: Path) -> None:
    pipe = FakePipeline()
    kwargs = dict(device=torch.device("cpu"), prompt="cached prompt", cache_dir=tmp_path)
    first = te.encode_and_evict(pipe, FluxAdapter(), **kwargs)
    second = te.encode_and_evict(pipe, FluxAdapter(), **kwargs)
    assert not first.cached and second.cached
    assert len(pipe.encode_calls) == 1, "second call should not have touched the encoders"
    for name in first.kwargs:
        assert torch.equal(first.kwargs[name], second.kwargs[name])


def test_embedding_cache_key_separates_prompts(tmp_path: Path) -> None:
    pipe = FakePipeline()
    te.encode_and_evict(
        pipe, FluxAdapter(), device=torch.device("cpu"), prompt="one", cache_dir=tmp_path
    )
    out = te.encode_and_evict(
        pipe, FluxAdapter(), device=torch.device("cpu"), prompt="two", cache_dir=tmp_path
    )
    assert not out.cached and len(pipe.encode_calls) == 2


def test_unusable_encode_outputs_raise_a_pointed_error() -> None:
    class NoUsableOutputs(FakePipeline):
        def __call__(self, num_inference_steps=1):  # accepts none of the embed names
            return num_inference_steps

    with pytest.raises(RuntimeError, match="encode_prompt_outputs"):
        te.encode_and_evict(
            NoUsableOutputs(), GenericAdapter(), device=torch.device("cpu"), prompt="x"
        )


# -- VAE phase -------------------------------------------------------------


def test_configure_vae_enables_what_exists() -> None:
    vae = FakeVAE()
    assert configure_vae(vae) == ["tiling", "slicing"]
    assert vae.tiled and vae.sliced
    assert configure_vae(None) == []


def test_vae_on_demand_times_decode_and_restores_the_method() -> None:
    pipe = FakePipeline()
    original = type(pipe.vae).decode
    stats: dict[str, float] = {}
    with vae_on_demand(pipe, torch.device("cpu"), stats):
        assert pipe.vae.decode.__name__ == "timed_decode"
        pipe.vae.decode(torch.zeros(1))
    assert "decode_s" in stats and stats["decode_s"] >= 0.0
    assert pipe.vae.decode.__func__ is original  # instance shadow removed


def test_wan_vae_has_upstream_tiling() -> None:
    """The M6 custom-tiling plan was retired because diffusers 0.39 ships
    enable_tiling on AutoencoderKLWan — pin that fact so a diffusers downgrade
    or API rename resurfaces it loudly."""
    from diffusers import AutoencoderKLWan

    assert callable(getattr(AutoencoderKLWan, "enable_tiling", None))


# -- execution-device override --------------------------------------------


def test_force_execution_device_is_scoped() -> None:
    pipe = FakePipeline()
    original_cls = type(pipe)
    assert pipe._execution_device.type == "cpu"
    with force_execution_device(pipe, torch.device("meta")):
        assert pipe._execution_device.type == "meta"
        assert isinstance(pipe, original_cls)  # still the same pipeline, subclassed
    assert type(pipe) is original_cls
    assert pipe._execution_device.type == "cpu"


# -- the real thing --------------------------------------------------------


@pytest.mark.network
@pytest.mark.parametrize("compression", [None, "fp8"])
def test_tiny_flux_end_to_end(tmp_path: Path, compression) -> None:
    """split-on-first-use -> encode+evict -> streamed denoise -> decode -> image."""
    pytest.importorskip("diffusers")
    pytest.importorskip("accelerate")
    from aircanvas import AirPipeline

    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    dtype = "bfloat16" if cuda else "float32"

    pipe = AirPipeline.from_pretrained(
        TINY_FLUX,
        compression=compression,
        shard_cache=tmp_path / "shards",
        device=device,
        compute_dtype=dtype,
        max_resident_blocks=0,  # force the streamed path on a 2-block model
        cache_embeddings=False,
    )
    try:
        assert pipe.manifest.compression == compression
        assert pipe.manifest.model_class == "FluxTransformer2DModel"
        assert pipe.adapter.key == "flux"
        assert pipe.plan.resident_blocks == 0 and pipe.plan.streamed_blocks == 2
        # The shard cache lives outside the repo (CLAUDE.md hard rule); here we
        # pinned it to tmp_path, so just confirm nothing landed in the package.
        assert not list(Path(__file__).resolve().parents[1].glob("**/*.safetensors"))

        steps = 3
        result = pipe(
            "a tiny still life",
            num_inference_steps=steps,
            height=32,
            width=32,
            max_sequence_length=16,
            generator=torch.Generator(device="cpu").manual_seed(0),
        )
        image = result.images[0]
        assert image.size == (32, 32) and image.mode == "RGB"

        report = pipe.report(as_dict=True)
        phases = report["phases"]
        assert phases["encode_s"] > 0.0
        assert phases["denoise_s"] > 0.0
        assert phases["decode_s"] > 0.0
        assert phases["total_s"] >= phases["denoise_s"]
        assert phases["steps"] == steps

        engine = phases["engine"]
        assert engine["block_loads"] == steps * 2, "every block must stream every step"
        assert engine["prefetch_hits"] > 0, "prefetcher never armed"
        assert engine["bytes_loaded"] > 0
        assert report["plan"]["resident_blocks"] == 0
        assert report["compression"] == compression

        text = pipe.report()
        assert "encode" in text and "denoise" in text and "decode" in text
        assert "resident blocks" in text and "slot pool" in text

        # Every streamed block is back on meta after the run.
        for name, param in pipe.pipeline.transformer.named_parameters():
            if name.startswith(("transformer_blocks.", "single_transformer_blocks.")):
                assert param.is_meta, name
    finally:
        pipe.close()


@pytest.mark.network
def test_tiny_flux_fp8_matches_uncompressed_closely(tmp_path: Path) -> None:
    """fp8 shards must change the picture only within quant tolerance."""
    pytest.importorskip("diffusers")
    from aircanvas import AirPipeline

    images = {}
    for compression in (None, "fp8"):
        pipe = AirPipeline.from_pretrained(
            TINY_FLUX,
            compression=compression,
            shard_cache=tmp_path / f"shards-{compression}",
            device="cpu",
            compute_dtype="float32",
            max_resident_blocks=0,
            cache_embeddings=False,
        )
        try:
            out = pipe(
                "a tiny still life",
                num_inference_steps=2,
                height=32,
                width=32,
                max_sequence_length=16,
                output_type="latent",
                generator=torch.Generator(device="cpu").manual_seed(4),
            )
            images[compression] = out.images.float()
        finally:
            pipe.close()

    ref, approx = images[None], images["fp8"]
    rel = (approx - ref).abs().max() / ref.abs().max()
    assert rel < 0.25, f"fp8 latents drifted {rel:.3f} from the bf16 reference"


def test_meta_transformer_signature_is_stable() -> None:
    """Guards the one diffusers contact point orchestration depends on."""
    from aircanvas.runtime.orchestrator import meta_transformer

    params = list(inspect.signature(meta_transformer).parameters)
    assert params == ["model_cls", "config", "device"]


def test_resolve_te_device_policies(monkeypatch) -> None:
    """'auto' falls back to CPU when the encoders exceed free VRAM (FLUX's
    9.1 GB T5-XXL on a 6 GB card); explicit requests are always honoured."""

    class _P:
        components = {"text_encoder": None}
        text_encoder = nn.Linear(4, 4)

    pipe = _P()
    names = te.text_encoder_names(pipe)
    cpu, cuda = torch.device("cpu"), torch.device("cuda")

    assert te.resolve_te_device(pipe, names, cpu) == cpu  # non-cuda compute: as-is
    assert te.resolve_te_device(pipe, names, cuda, "cpu") == cpu  # explicit wins

    monkeypatch.setattr(te, "free_vram_bytes", lambda device=None: 10)
    assert te.resolve_te_device(pipe, names, cuda) == cpu  # too big -> CPU
    monkeypatch.setattr(te, "free_vram_bytes", lambda device=None: 1 << 30)
    assert te.resolve_te_device(pipe, names, cuda) == cuda  # fits -> device


def test_dispatched_encoders_are_never_moved() -> None:
    """accelerate device_map models manage their own placement — .to() on
    them raises. resolve_te_device must leave them alone and _move must skip
    them (the 11.4 GB UMT5-on-16GB-RAM path)."""

    class _Dispatched(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.hf_device_map = {"lin": "cpu"}

        def to(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError(".to() must not be called on a dispatched model")

    class _P:
        components = {"text_encoder": None}
        text_encoder = _Dispatched()

    pipe = _P()
    names = te.text_encoder_names(pipe)
    cuda = torch.device("cuda")
    assert te.resolve_te_device(pipe, names, cuda) == cuda
    te._move(pipe, names, "cpu")  # must not raise
