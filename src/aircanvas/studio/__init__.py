"""AirCanvas Studio — the local desktop UI.

A single-page app served by the same Python process that owns the pipeline, so
telemetry travels in-process instead of over a socket to somewhere else. See
docs/STUDIO.md for the architecture and `aircanvas studio --help` to run it.

FastAPI and uvicorn are an optional extra (`pip install "aircanvas[studio]"`);
importing this package does not require them, so `aircanvas doctor` on a
minimal install stays unaffected.
"""

from __future__ import annotations

__all__ = ["create_app", "serve"]


def create_app(studio: object | None = None) -> object:
    """The FastAPI application. Imported lazily — see the module docstring."""
    from aircanvas.studio.server import Studio
    from aircanvas.studio.server import create_app as _create_app

    if studio is not None and not isinstance(studio, Studio):
        raise TypeError(f"Expected a Studio, got {type(studio).__name__}")
    return _create_app(studio)


def serve(**kwargs: object) -> None:
    """Run the Studio until interrupted. See `server.serve` for the options."""
    from aircanvas.studio.server import serve as _serve

    _serve(**kwargs)  # type: ignore[arg-type]
