from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class InferenceEngine(ABC):
    """Abstract base class for inference backends (LiteRT / PyCoral / CPU)."""

    @abstractmethod
    def get_input_shape(self) -> tuple[int, int]:
        """Returns input (height, width)."""
        pass

    @abstractmethod
    def get_input_details(self) -> list[dict]:
        """Returns input details list."""
        pass

    @abstractmethod
    def get_output_details(self) -> list[dict]:
        """Returns output details list."""
        pass

    @abstractmethod
    def set_tensor(self, tensor_index: int, value: np.ndarray) -> None:
        """Sets tensor data on input."""
        pass

    @abstractmethod
    def get_tensor(self, tensor_index: int) -> np.ndarray:
        """Gets tensor data from output."""
        pass

    @abstractmethod
    def invoke(self) -> None:
        """Invokes model inference."""
        pass

    @abstractmethod
    def classify(self, image: np.ndarray, top_k: int = 2, threshold: float = 0.5) -> list[Any]:
        """Classifies image crop and returns list of results (id, score)."""
        pass
