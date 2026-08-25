"""Verify GGUF-sourced shard caches and on-stream LoRA fusion on REAL models.

Usage (presets pin known-good public repos):

    python benchmarks/verify_gguf_lora.py flux2-9b            # FLUX.2-klein-9B, Q4_K_S
    python benchmarks/verify_gguf_lora.py wan-1.3b            # Wan 2.1 T2V 1.3B, Q8_0
    python benchmarks/verify_gguf_lora.py custom \
        --source <repo-or-dir> --gguf <file-or-repo:file> [--lora ...]

What it proves, per model:

1. **Split** — the quantized .gguf becomes an AirCanvas shard cache without
   the original checkpoint (only config.json is fetched). Prints download
   size vs the bf16 original.
2. **Fidelity gate** — every shard is loaded through the engine's own reader
   path (ShardHeader -> DecompressPlan, the exact code `StreamingEngine`
   binds from) and compared against an independent reference: the GGUF
   dequantized with gguf-py, cast, and pushed through the same quant
   round-trip the splitter used. Expected: BITWISE equal.
3. **LoRA gate** (when --lora is given) — the same shards with the overlay
   fused on load, compared against reference + a manually computed
   ``scale * up @ down``. Expected: BITWISE equal.
4. **--generate** (optional) — full `AirPipeline` generation from the GGUF
   cache (+ LoRA), producing an image/frames and `pipe.report()`. This
   downloads the pipeline's OTHER components too (klein-9B's text encoder
   is a 16.4 GB Qwen3 — on low-RAM boxes it is split and streamed, the
   Qwen-Image precedent).

Gates run fine on CPU; CUDA just makes them faster. Everything is resumable.
FLUX.2-klein-9B is a gated repo: accept the license on its Hub page and log
in (`hf auth login`) or pass --hf-token.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from aircanvas.adapters import resolve
from aircanvas.lora import LoraOverlay, load_lora_deltas
from aircanvas.sharding import quant
from aircanvas.sharding.gguf_source import (
    DiffusersGGUFCheckpoint,
    GGUFCheckpoint,
    plan_covers,
    resolve_gguf_path,
)
from aircanvas.sharding.splitter import shard_cache_dir, split_model
from aircanvas.streaming.prefetch import DecompressPlan, ShardHeader

PRESETS: dict[str, dict] = {
    "flux2-9b": {
        "source": "black-forest-labs/FLUX.2-klein-9B",
        "gguf": "unsloth/FLUX.2-klein-9B-GGUF:flux-2-klein-9b-Q4_K_S.gguf",
        "lora": "thedeoxen/refcontrol-FLUX.2-klein-9B-reference-pose-lora",
        "lora_weight": "refcontrol_v2_poses.safetensors",
        "steps": 4,
        "height": 512,
        "width": 512,
        "note": "gated repo: accept the license on the Hub first (hf auth login / --hf-token)",
    },
    "wan-1.3b": {
        "source": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "gguf": "samuelchristlie/Wan2.1-T2V-1.3B-GGUF:Wan2.1-T2V-1.3B-Q8_0.gguf",
        "lora": None,
        "lora_weight": None,
        "steps": 8,
        "height": 480,
        "width": 832,
        "frames": 33,
        "note": "",
    },
}


def _materialize_like_engine(
    path: Path, compression: str | None, compute_dtype: torch.dtype | None, device: torch.device
) -> dict[str, torch.Tensor]:
    """Mirror StreamingEngine._materialize: the byte-identical bind path."""
    header = ShardHeader.parse(path)
    buf = torch.empty(header.blob_size, dtype=torch.uint8)
    header.read_blob_into(buf)
    if compression is None:
        views = header.build_views(buf)
        return {name: view.to(device) for name, view in views.items()}
    assert compute_dtype is not None
    plan = DecompressPlan.build(header, compute_dtype)
    src = buf.to(device)
    dst = torch.empty(plan.out_bytes, dtype=torch.uint8, device=device)
    plan.run(src, dst)
    return plan.views(dst)


def _reference_checkpoint(gguf_path: Path, model_class: str, source: str, args):
    """Same source selection the splitter makes: direct reader, else diffusers."""
    ckpt = GGUFCheckpoint(gguf_path)
    adapter = resolve(model_class)
    names = ckpt.tensor_names()
    if plan_covers(adapter.block_plan(names), names):
        return ckpt, adapter
    ckpt = DiffusersGGUFCheckpoint(
        gguf_path, model_class, source, "transformer", None, args.hf_token, torch.bfloat16
    )
    return ckpt, adapter


def _reference_tensors(
    ckpt, names: tuple[str, ...], compression: str | None, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """gguf dequant -> compute dtype -> the SAME quant round-trip the cache
    stores. Independent of the engine path; agreement must be bitwise."""
    raw = ckpt.gather(names, dtype)
    if compression is None:
        return raw
    stored, meta = quant.compress_state_dict(raw, compression)
    return quant.decompress_state_dict(
        stored, compression=compression, compute_dtype=dtype, metadata=meta
    )


def run_gate(args) -> tuple[int, int, int]:
    device = torch.device(args.device)
    dtype = torch.bfloat16
    compression = None if args.compression == "none" else args.compression

    print(f"== Split from GGUF ({args.gguf}) ==")
    t0 = time.perf_counter()
    manifest = split_model(
        args.source,
        compression=compression,
        compute_dtype="bfloat16",
        gguf_file=args.gguf,
        hf_token=args.hf_token,
    )
    cache = shard_cache_dir(args.source, "transformer", compression, "bfloat16", args.gguf)
    gguf_path = resolve_gguf_path(args.gguf, hf_token=args.hf_token)
    split_s = time.perf_counter() - t0
    gguf_gb = gguf_path.stat().st_size / 1e9
    cache_gb = sum((cache / f).stat().st_size for f in manifest.shard_files()) / 1e9
    print(
        f"   {len(manifest.blocks)} blocks in {split_s:.0f}s | gguf {gguf_gb:.2f} GB "
        f"-> cache {cache_gb:.2f} GB ({manifest.compression or 'none'})"
    )

    overlay = LoraOverlay()
    deltas = {}
    if args.lora:
        overlay.load(
            args.lora,
            scale=args.lora_scale,
            weight_name=args.lora_weight or None,
            hf_token=args.hf_token,
        )
        overlay.materialize(device, dtype)
        deltas = load_lora_deltas(
            args.lora, weight_name=args.lora_weight or None, hf_token=args.hf_token
        )
        print(f"   LoRA: {args.lora} ({overlay.nbytes() / 1e6:.1f} MB resident)")

    print("== Fidelity gate (engine load path vs independent GGUF reference) ==")
    ckpt, _ = _reference_checkpoint(gguf_path, manifest.model_class, args.source, args)
    checked = fused = mismatched = 0
    files = [b.file for b in manifest.blocks]
    if args.max_blocks:
        files = files[: args.max_blocks]
    # The resident shard is stored uncompressed but goes through the same
    # bind-and-fuse path in the engine — gate it too.
    files.append(manifest.resident_file)
    for fname in files:
        resident = fname == manifest.resident_file
        shard_compression = None if resident else compression
        engine_side = _materialize_like_engine(cache / fname, shard_compression, dtype, device)
        names = tuple(engine_side)
        reference = _reference_tensors(ckpt, names, shard_compression, dtype)
        if args.lora:
            overlay.apply(engine_side)  # what the engine does after dequant
            for name, ref in reference.items():  # independent manual fuse
                d = deltas.get(name)
                if d is not None and ref.ndim == 2 and d.scale != 0.0:
                    up = d.up.to(device=device, dtype=dtype)
                    down = d.down.to(device=device, dtype=dtype)
                    reference[name] = ref.to(device).addmm_(up, down, alpha=d.scale)
                    fused += 1
        for name in names:
            checked += 1
            if not torch.equal(engine_side[name].to(device), reference[name].to(device)):
                mismatched += 1
                diff = (engine_side[name].float() - reference[name].to(device).float()).abs().max()
                print(f"   MISMATCH {fname}:{name} max|diff|={diff:.3e}")
    verdict = "BITWISE PASS" if mismatched == 0 else f"FAIL ({mismatched} tensors)"
    print(f"   {checked} tensors across {len(files)} shards, {fused} LoRA fusions -> {verdict}")
    return checked, fused, mismatched


def run_generate(args) -> None:
    from aircanvas.api import AirPipeline

    print("== Full generation from the GGUF cache ==")
    kwargs: dict = {}
    if args.lora:
        kwargs.update(lora=args.lora, lora_scale=args.lora_scale)
        if args.lora_weight:
            kwargs["lora_weight_name"] = args.lora_weight
    pipe = AirPipeline.from_pretrained(
        args.source,
        compression=None if args.compression == "none" else args.compression,
        gguf_file=args.gguf,
        hf_token=args.hf_token,
        **kwargs,
    )
    call: dict = {"num_inference_steps": args.steps, "height": args.height, "width": args.width}
    if args.frames:
        call["num_frames"] = args.frames
    result = pipe(args.prompt, **call)
    images = getattr(result, "images", None)
    if images:
        out = Path(args.out)
        images[0].save(out)
        print(f"   wrote {out}")
    print(pipe.report())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("preset", choices=[*PRESETS, "custom"])
    parser.add_argument("--source", help="pipeline repo id or local dir (custom preset)")
    parser.add_argument("--gguf", help="local .gguf or 'repo_id:filename' (custom preset)")
    parser.add_argument("--lora", default=None)
    parser.add_argument("--lora-weight", default=None)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--compression", default="fp8", choices=["none", "fp8", "nf4"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--max-blocks", type=int, default=0, help="gate only the first N shards")
    parser.add_argument("--generate", action="store_true", help="also run a full generation")
    parser.add_argument("--prompt", default="a watercolor fox reading a newspaper")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--out", default="verify_gguf_lora.png")
    args = parser.parse_args()

    preset = PRESETS.get(args.preset, {})
    for key in ("source", "gguf", "lora", "lora_weight", "steps", "height", "width", "frames"):
        if getattr(args, key, None) in (None, False) and key in preset:
            setattr(args, key, preset[key])
    if not args.source or not args.gguf:
        parser.error("custom preset needs --source and --gguf")
    if preset.get("note"):
        print(f"note: {preset['note']}")

    checked, fused, mismatched = run_gate(args)
    if args.generate and mismatched == 0:
        run_generate(args)

    print("\n== Paste-ready summary ==")
    print(
        f"| {args.preset} | {args.gguf.rsplit(':', 1)[-1]} | {args.compression} | "
        f"{checked} tensors bitwise-checked | {fused} LoRA fusions | "
        f"{'PASS' if mismatched == 0 else 'FAIL'} |"
    )
    return 0 if mismatched == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
