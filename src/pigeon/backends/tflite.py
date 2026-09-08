from __future__ import annotations

import collections
from typing import Any, cast

import cv2
import numpy as np

from pigeon.backends.base import InferenceEngine

# Try modern ai-edge-litert, then tflite_runtime, then tensorflow.lite
Interpreter: Any = None
try:
    from ai_edge_litert.interpreter import Interpreter  # type: ignore[no-redef]
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore[no-redef]
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter  # type: ignore[no-redef]
        except ImportError:
            Interpreter = None

ClassificationResult = collections.namedtuple("ClassificationResult", ["id", "score"])


class TFLiteEngine(InferenceEngine):
    """LiteRT / TFLite CPU inference engine."""

    interpreter: Any
    input_details: list[dict[str, Any]]
    output_details: list[dict[str, Any]]

    def __init__(self, model_path: str, num_threads: int = 4):
        if Interpreter is None:
            raise ImportError(
                "No LiteRT / TFLite interpreter available. "
                "Please install `ai-edge-litert` via `uv pip install ai-edge-litert` "
                "or install pigeon with `pip install -e '.[tflite]'`."
            )
        self.model_path = model_path
        self.interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def get_input_shape(self) -> tuple[int, int]:
        shape = self.input_details[0]["shape"]
        # [batch, height, width, channels]
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

        input_detail = self.input_details[0]
        output_detail = self.output_details[0]

        input_dtype = input_detail["dtype"]
        img_tensor: np.ndarray
        if input_dtype == np.float32:
            img_tensor = img_scaled.astype(np.float32) / 255.0
        elif input_dtype in (np.int8, np.uint8):
            # Quantize if scale is present
            quant_params = input_detail.get("quantization", (0.0, 0))
            if isinstance(quant_params, tuple) and quant_params[0] != 0.0:
                scale, zp = quant_params
                img_tensor = np.array(
                    (img_scaled.astype(np.float32) / 255.0) / scale + zp, dtype=input_dtype
                )
            else:
                img_tensor = img_scaled.astype(input_dtype)
        else:
            img_tensor = img_scaled

        img_tensor = np.expand_dims(img_tensor, axis=0)
        self.interpreter.set_tensor(input_detail["index"], img_tensor)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(output_detail["index"])[0]

        # Dequantize output if quantized
        quant_params = output_detail.get("quantization_parameters", {})
        if quant_params and "scales" in quant_params and len(quant_params["scales"]) > 0:
            scale, zp = quant_params["scales"][0], quant_params["zero_points"][0]
            output_data = (output_data.astype(np.float32) - zp) * scale
        elif (
            isinstance(output_detail.get("quantization"), tuple)
            and output_detail["quantization"][0] != 0.0
        ):
            scale, zp = output_detail["quantization"]
            output_data = (output_data.astype(np.float32) - zp) * scale

        top_indices = np.argsort(output_data)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            score = float(output_data[idx])
            if score >= threshold:
                results.append(ClassificationResult(id=int(idx), score=score))
        return results
