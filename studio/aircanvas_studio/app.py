"""FastAPI wrapper around AirPipeline.

Run from a machine with the GPU/RAM AirCanvas is for. The Grok preview uses a
browser demo of the same controls; this server is the live path.
"""

from __future__ import annotations

import io
import threading
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="AirCanvas Studio", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_lock = threading.Lock()
_pipe: Any = None
_loaded: dict[str, Any] = {}


class LoadBody(BaseModel):
    repo: str
    compression: Literal["none", "fp8", "nf4"] = "fp8"
    vram_budget: str = "auto"
    ram_budget: str = "auto"
    gguf_file: str | None = None
    hf_token: str | None = None
    prefetch: bool = True
    cache_embeddings: bool = True


class LoraBody(BaseModel):
    source: str
    scale: float = 1.0
    adapter_name: str | None = None
    weight_name: str | None = None


class GenerateBody(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = Field(20, ge=1, le=100)
    width: int = Field(1024, ge=256, le=2048)
    height: int = Field(1024, ge=256, le=2048)
    num_frames: int | None = None
    guidance_scale: float = 4.0
    seed: int | None = 13


@app.get("/api/health")
def health() -> dict[str, Any]:
    import torch

    return {
        "ok": True,
        "cuda": torch.cuda.is_available(),
        "loaded": _loaded or None,
    }


@app.get("/api/models")
def models() -> list[dict[str, Any]]:
    from aircanvas.cli import KNOWN_MODELS

    return [
        {
            "name": m.name,
            "repo": m.repo_id,
            "kind": m.kind,
            "n_blocks": m.n_blocks,
            "dit_bytes": m.dit_bytes,
            "steps": m.steps,
            "gated": m.gated,
            "note": m.note,
        }
        for m in KNOWN_MODELS
    ]


@app.post("/api/load")
def load(body: LoadBody) -> dict[str, Any]:
    from aircanvas import AirPipeline

    compression = None if body.compression == "none" else body.compression
    kwargs: dict[str, Any] = {
        "compression": compression,
        "vram_budget": body.vram_budget,
        "ram_budget": body.ram_budget,
        "prefetch": body.prefetch,
        "cache_embeddings": body.cache_embeddings,
        "hf_token": body.hf_token or None,
    }
    if body.gguf_file:
        kwargs["gguf_file"] = body.gguf_file
    with _lock:
        global _pipe
        if _pipe is not None:
            _pipe.close()
        _pipe = AirPipeline.from_pretrained(body.repo, **kwargs)
        _loaded.clear()
        _loaded.update({"repo": body.repo, "compression": body.compression})
        return {"ok": True, "report": _pipe.plan.describe()}


@app.post("/api/lora")
def load_lora(body: LoraBody) -> dict[str, str]:
    if _pipe is None:
        raise HTTPException(409, "Load a model first")
    name = _pipe.load_lora(
        body.source,
        scale=body.scale,
        adapter_name=body.adapter_name,
        weight_name=body.weight_name,
    )
    return {"adapter": name}


@app.delete("/api/lora")
def unload_lora(name: str | None = None) -> dict[str, bool]:
    if _pipe is None:
        raise HTTPException(409, "Load a model first")
    _pipe.unload_lora(name)
    return {"ok": True}


@app.post("/api/generate")
def generate(body: GenerateBody) -> dict[str, Any]:
    if _pipe is None:
        raise HTTPException(409, "Load a model first")
    import torch

    kwargs: dict[str, Any] = {
        "num_inference_steps": body.steps,
        "width": body.width,
        "height": body.height,
        "guidance_scale": body.guidance_scale,
    }
    if body.negative_prompt:
        kwargs["negative_prompt"] = body.negative_prompt
    if body.num_frames:
        kwargs["num_frames"] = body.num_frames
    if body.seed is not None:
        kwargs["generator"] = torch.Generator(device=_pipe.device).manual_seed(body.seed)

    with _lock:
        out = _pipe(body.prompt, **kwargs)
        report = _pipe.report(as_dict=True)

    payload: dict[str, Any] = {"report": report}
    images = getattr(out, "images", None)
    frames = getattr(out, "frames", None)
    if images:
        buf = io.BytesIO()
        images[0].save(buf, format="PNG")
        import base64

        payload["image_png_b64"] = base64.b64encode(buf.getvalue()).decode("ascii")
    if frames is not None:
        payload["n_frames"] = len(frames[0]) if frames else 0
    return payload
