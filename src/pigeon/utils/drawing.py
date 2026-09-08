from __future__ import annotations

import cv2
import numpy as np


def draw_debug_info(img: np.ndarray, frame_idx: int, proc_time: float) -> None:
    """Draws frame index and processing time on the top-left of the image."""
    cv2.putText(
        img,
        f"frame_idx: {frame_idx}",
        (10, 30),
        cv2.FONT_HERSHEY_DUPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"proc_time: {round(proc_time * 1000, 1)} ms",
        (10, 50),
        cv2.FONT_HERSHEY_DUPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_detection(
    frame: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    label: str,
    color: tuple = (0, 255, 0),
    text_color: tuple = (0, 0, 255),
) -> np.ndarray:
    """Draws bounding box and class label on the frame."""
    frame = cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
    label_y = max(y0 + 25, 25)
    frame = cv2.putText(
        frame,
        label,
        (x0, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        text_color,
        2,
        cv2.LINE_AA,
    )
    return frame
