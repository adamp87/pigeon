from __future__ import annotations

import collections
from typing import Any, cast

import cv2
import numpy as np

from pigeon.backends.base import InferenceEngine

ClassificationResult = collections.namedtuple("ClassificationResult", ["id", "score"])


class EdgeTPUEngine(InferenceEngine):
    """Google Coral Edge TPU inference engine using PyCoral."""

    interpreter: Any
    input_details: list[dict[str, Any]]
    output_details: list[dict[str, Any]]

    def __init__(self, model_path: str, device: str | None = None):
        try:
            from pycoral.adapters import classify, common
            from pycoral.utils.edgetpu import (
                load_edgetpu_delegate,
                make_interpreter,
                run_inference,
            )

            self.pycoral_classify = classify
            self.pycoral_common = common
            self.pycoral_run_inference = run_inference
        except (ImportError, AttributeError, SystemError, Exception) as e:
            raise ImportError(
                "PyCoral is not functional on this environment "
                "(requires Python <=3.9 and NumPy <2.0). "
                "For RPi5 or CPU inference, please use the LiteRT backend (`--backend tflite`)."
            ) from e

        self.model_path = model_path
        self.device = device

        if device:
            delegate = load_edgetpu_delegate({"device": device})
            self.interpreter = make_interpreter(model_path, delegate=delegate)
        else:
            self.interpreter = make_interpreter(model_path)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def get_input_shape(self) -> tuple[int, int]:
        shape = self.input_details[0]["shape"]
        return int(shape[1]), int(shape[2])

    def get_input_details(self) -> list[dict[str, Any]]:
        return self.input_details

    def get_output_details(self) -> list[dict[str, Any]]:
        return self.output_details

    def set_tensor(self, tensor_index: int, value: np.ndarray) -> None:
        self.interpreter.set_tensor(tensor_index, value)

    def get_tensor(self, tensor_index: int) -> np.ndarray:
        return cast(np.ndarray, self.interpreter.get_tensor(tensor_index))

    def invoke(self) -> None:
        self.interpreter.invoke()

    def classify(
        self, image: np.ndarray, top_k: int = 2, threshold: float = 0.5
    ) -> list[ClassificationResult]:
        target_h, target_w = self.get_input_shape()
        img_scaled = cv2.resize(image, (target_w, target_h))
        self.pycoral_run_inference(self.interpreter, img_scaled.tobytes())
        classes = self.pycoral_classify.get_classes(self.interpreter, top_k, threshold)
        return [ClassificationResult(id=c.id, score=float(c.score)) for c in classes]


def list_available_tpus() -> list[dict[str, Any]]:
    """Lists connected Coral Edge TPU devices."""
    try:
        from pycoral.utils.edgetpu import list_edge_tpus

        return cast(list[dict[str, Any]], list_edge_tpus())
    except (ImportError, AttributeError, SystemError, Exception):
        return []
