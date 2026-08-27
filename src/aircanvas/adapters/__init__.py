"""Model-family adapters (see base.py for the discovery mechanism).

Importing this package registers all named adapters. `resolve()` maps a
diffusers model class name to an adapter, falling back to GenericAdapter.
"""

from aircanvas.adapters import cogvideox as _cogvideox  # noqa: F401  (registration side effect)
from aircanvas.adapters import flux as _flux  # noqa: F401  (registration side effect)
from aircanvas.adapters import flux2 as _flux2  # noqa: F401  (registration side effect)
from aircanvas.adapters import hunyuan_video as _hunyuan  # noqa: F401  (registration side effect)
from aircanvas.adapters import qwen_image as _qwen_image  # noqa: F401  (registration side effect)
from aircanvas.adapters import sd3 as _sd3  # noqa: F401  (registration side effect)
from aircanvas.adapters import wan as _wan  # noqa: F401  (registration side effect)
from aircanvas.adapters.base import (
    AdapterError,
    BlockPlan,
    GenericAdapter,
    ModelAdapter,
    resolve,
)

__all__ = ["AdapterError", "BlockPlan", "GenericAdapter", "ModelAdapter", "resolve"]
