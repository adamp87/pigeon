from __future__ import annotations

from pigeon.backends.base import InferenceEngine
from pigeon.backends.tflite import TFLiteEngine

__all__ = ["InferenceEngine", "TFLiteEngine"]

try:
    from pigeon.backends.edgetpu import EdgeTPUEngine, list_available_tpus  # noqa: F401

    __all__.extend(["EdgeTPUEngine", "list_available_tpus"])
except ImportError:
    pass
