from __future__ import annotations

import logging as log
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import cv2
import numpy as np

from pigeon.backends.base import InferenceEngine
from pigeon.models.letterbox import letterbox
from pigeon.models.ssd import decode_ssd_predictions
from pigeon.models.yolo import Object, yolo26, yolov7, yolov8
from pigeon.utils.drawing import draw_debug_info, draw_detection


def default_pigeon_action() -> None:
    """Default callback when a pigeon is detected."""
    log.info("PIGEON DETECTED!")


@dataclass
class FrameTiming:
    read_ms: float = 0.0
    preprocess_ms: float = 0.0
    detection_ms: float = 0.0
    classification_ms: float = 0.0
    crops_classified: int = 0
    draw_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class PipelineMetrics:
    total_frames: int = 0
    pigeons_detected: int = 0
    total_crops_classified: int = 0
    timings: list[FrameTiming] = field(default_factory=list)

    @property
    def avg_read_ms(self) -> float:
        return float(np.mean([t.read_ms for t in self.timings])) if self.timings else 0.0

    @property
    def avg_preprocess_ms(self) -> float:
        return float(np.mean([t.preprocess_ms for t in self.timings])) if self.timings else 0.0

    @property
    def avg_detection_ms(self) -> float:
        return float(np.mean([t.detection_ms for t in self.timings])) if self.timings else 0.0

    @property
    def avg_classification_ms(self) -> float:
        return float(np.mean([t.classification_ms for t in self.timings])) if self.timings else 0.0

    @property
    def avg_per_crop_classification_ms(self) -> float:
        total_cls_ms = sum(t.classification_ms for t in self.timings)
        return (
            (total_cls_ms / self.total_crops_classified) if self.total_crops_classified > 0 else 0.0
        )

    @property
    def avg_draw_ms(self) -> float:
        return float(np.mean([t.draw_ms for t in self.timings])) if self.timings else 0.0

    @property
    def avg_inference_ms(self) -> float:
        """Pure neural network inference latency (Detection + Classification)."""
        return self.avg_detection_ms + self.avg_classification_ms

    @property
    def avg_processing_ms(self) -> float:
        """Frame processing latency excluding disk/video read I/O."""
        return (
            self.avg_preprocess_ms
            + self.avg_detection_ms
            + self.avg_classification_ms
            + self.avg_draw_ms
        )

    @property
    def avg_total_ms(self) -> float:
        """Full end-to-end latency including disk/stream read."""
        return float(np.mean([t.total_ms for t in self.timings])) if self.timings else 0.0

    @property
    def inference_fps(self) -> float:
        """Maximum inference throughput capability."""
        return (1000.0 / self.avg_inference_ms) if self.avg_inference_ms > 0 else 0.0

    @property
    def processing_fps(self) -> float:
        """Throughput excluding disk I/O (live camera speed)."""
        return (1000.0 / self.avg_processing_ms) if self.avg_processing_ms > 0 else 0.0

    @property
    def fps(self) -> float:
        """End-to-end throughput including reading."""
        return (1000.0 / self.avg_total_ms) if self.avg_total_ms > 0 else 0.0

    def summary(self) -> str:
        lines = [
            "=" * 56,
            "              PIPELINE BENCHMARK SUMMARY                ",
            "=" * 56,
            f" Total frames processed:    {self.total_frames:6d}",
            f" Pigeon events detected:    {self.pigeons_detected:6d}",
            f" Total crops classified:    {self.total_crops_classified:6d}",
            "-" * 56,
            " Stage Latency Breakdown (Avg / Frame):",
            f"   - Read & Decode:          {self.avg_read_ms:7.2f} ms",
            f"   - Preprocessing:          {self.avg_preprocess_ms:7.2f} ms",
            f"   - Object Detection:       {self.avg_detection_ms:7.2f} ms",
            (
                f"   - Classification:         {self.avg_classification_ms:7.2f} ms  "
                f"({self.avg_per_crop_classification_ms:.2f} ms/crop)"
            ),
            f"   - Drawing & Overlay:      {self.avg_draw_ms:7.2f} ms",
            "-" * 56,
            f" Pure Inference Latency:     {self.avg_inference_ms:7.2f} ms  (Det + Cls)",
            f" Processing Latency:         {self.avg_processing_ms:7.2f} ms  (Excl. Disk Read)",
            f" Total Frame Latency:        {self.avg_total_ms:7.2f} ms  (Incl. Disk Read)",
            "-" * 56,
            f" Pure Inference Speed:       {self.inference_fps:7.2f} FPS (Max TPU/CPU capability)",
            f" Live Stream Throughput:     {self.processing_fps:7.2f} FPS (Camera / In-Memory)",
            f" End-to-End Throughput:      {self.fps:7.2f} FPS (Disk I/O bounded)",
            "=" * 56,
        ]
        return "\n".join(lines)


class VideoReader:
    """Reads frames from video file, camera, single image, or image directory with looping."""

    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")

    MAX_IMAGE_DIM = 1280

    def __init__(
        self,
        source: str | int,
        loop: bool = False,
        default_image_fps: float = 1.0,
    ):
        self.source = source
        self.loop = loop
        self.default_image_fps = default_image_fps
        self.is_image_directory = False
        self.is_single_image = False
        self.image_files: list[str] = []
        self.cap: cv2.VideoCapture | None = None
        self.width = 0
        self.height = 0
        self.fps = default_image_fps

        self._init_source()

    def _resize_if_needed(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if max(h, w) > self.MAX_IMAGE_DIM:
            scale = self.MAX_IMAGE_DIM / float(max(h, w))
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img

    def _init_source(self) -> None:
        if isinstance(self.source, str) and os.path.isdir(self.source):
            self.is_image_directory = True
            entries = sorted(os.listdir(self.source))
            self.image_files = [
                os.path.join(self.source, f)
                for f in entries
                if f.lower().endswith(self.IMAGE_EXTENSIONS)
            ]
            if not self.image_files:
                raise ValueError(f"No valid image files found in directory: {self.source}")
            first_img = cv2.imread(self.image_files[0])
            if first_img is None:
                raise ValueError(f"Failed to read first image: {self.image_files[0]}")
            first_img = self._resize_if_needed(first_img)
            self.height, self.width = first_img.shape[:2]
            self.fps = self.default_image_fps
        elif (
            isinstance(self.source, str)
            and os.path.isfile(self.source)
            and self.source.lower().endswith(self.IMAGE_EXTENSIONS)
        ):
            self.is_single_image = True
            self.image_files = [self.source]
            first_img = cv2.imread(self.source)
            if first_img is None:
                raise ValueError(f"Failed to read image file: {self.source}")
            first_img = self._resize_if_needed(first_img)
            self.height, self.width = first_img.shape[:2]
            self.fps = self.default_image_fps
        else:
            cap_source = int(self.source) if str(self.source).isdigit() else self.source
            self.cap = cv2.VideoCapture(cap_source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open video source: {self.source}")
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps = cap_fps if cap_fps and cap_fps > 0 else 25.0

    def read_frames(self, max_frames: int | None = None) -> Iterator[tuple[int, np.ndarray, float]]:
        """Yields (frame_idx, frame_bgr, read_time_ms)."""
        frame_idx = 0

        if self.is_image_directory or self.is_single_image:
            while True:
                for img_path in self.image_files:
                    if max_frames is not None and frame_idx >= max_frames:
                        return
                    t0 = time.perf_counter()
                    frame = cv2.imread(img_path)
                    if frame is not None:
                        frame = self._resize_if_needed(frame)
                    read_ms = (time.perf_counter() - t0) * 1000.0
                    if frame is None:
                        continue
                    if frame.shape[0] != self.height or frame.shape[1] != self.width:
                        frame = cv2.resize(frame, (self.width, self.height))
                    yield frame_idx, frame, read_ms
                    frame_idx += 1

                if not self.loop or self.is_single_image:
                    break
        else:
            assert self.cap is not None
            while True:
                if max_frames is not None and frame_idx >= max_frames:
                    break
                t0 = time.perf_counter()
                ret, frame = self.cap.read()
                read_ms = (time.perf_counter() - t0) * 1000.0
                if not ret:
                    if self.loop:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        t0 = time.perf_counter()
                        ret, frame = self.cap.read()
                        read_ms = (time.perf_counter() - t0) * 1000.0
                        if not ret:
                            break
                    else:
                        break

                yield frame_idx, frame, read_ms
                frame_idx += 1

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()


class PigeonPipeline:
    """Unified detection and classification pipeline."""

    def __init__(
        self,
        engine_detect: InferenceEngine,
        engine_pigeon: InferenceEngine,
        labels_coco: dict[int, str],
        labels_bird: dict[int, str],
        detector_type: str = "ssd",
        score_threshold: float = 0.5,
        top_k: int = 2,
        on_pigeon_detected: Callable[[], None] | None = None,
    ):
        self.engine_detect = engine_detect
        self.engine_pigeon = engine_pigeon
        self.labels_coco = labels_coco
        self.labels_bird = labels_bird
        self.detector_type = detector_type.lower()
        self.score_threshold = score_threshold
        self.top_k = top_k
        self.on_pigeon_detected = on_pigeon_detected or default_pigeon_action

        det_h, det_w = self.engine_detect.get_input_shape()
        self.det_size = (det_w, det_h)
        self.metrics = PipelineMetrics()

    def detect_objects(
        self, frame_rgb: np.ndarray, original_size: tuple
    ) -> tuple[list[Object], float, float]:
        """Runs detection on frame and returns (objs, preprocess_ms, detect_ms)."""
        orig_w, orig_h = original_size
        det_w, det_h = self.det_size

        t0 = time.perf_counter()
        if self.detector_type == "ssd":
            img_det = cv2.resize(frame_rgb, self.det_size)
            input_detail = self.engine_detect.get_input_details()[0]
            input_dtype = input_detail["dtype"]
            if input_dtype in (np.int8, np.uint8):
                img_input = img_det[None, ...]
            else:
                img_input = (img_det.astype(np.float32) / 255.0)[None, ...]
            self.engine_detect.set_tensor(input_detail["index"], img_input)
            t_pre = time.perf_counter()
            self.engine_detect.invoke()
            t_inv = time.perf_counter()
            objs = decode_ssd_predictions(
                self.engine_detect,
                score_threshold=self.score_threshold,
                det_scale=(orig_w, orig_h),
            )
            preprocess_ms = (t_pre - t0) * 1000.0
            detect_ms = (t_inv - t_pre + (time.perf_counter() - t_inv)) * 1000.0
            return objs, preprocess_ms, detect_ms

        elif self.detector_type == "yolo26":
            img_det, ratio, pad = letterbox(frame_rgb, (det_h, det_w), auto=False)
            t_pre = time.perf_counter()
            objs = yolo26(img_det, self.engine_detect, self.score_threshold, ratio, pad)
            t_inv = time.perf_counter()
            preprocess_ms = (t_pre - t0) * 1000.0
            detect_ms = (t_inv - t_pre) * 1000.0
            return objs, preprocess_ms, detect_ms

        elif self.detector_type in ("yolov7", "yolov8"):
            img_det = cv2.resize(frame_rgb, self.det_size)
            t_pre = time.perf_counter()
            if self.detector_type == "yolov7":
                objs = yolov7(img_det, self.engine_detect, self.score_threshold)
            else:
                objs = yolov8(img_det, self.engine_detect, self.score_threshold)
            t_inv = time.perf_counter()
            scale_x, scale_y = orig_w / det_w, orig_h / det_h
            scaled_objs = []
            for o in objs:
                scaled_objs.append(
                    Object(
                        id=o.id,
                        score=o.score,
                        bbox=type(o.bbox)(
                            xmin=o.bbox.xmin * scale_x,
                            ymin=o.bbox.ymin * scale_y,
                            xmax=o.bbox.xmax * scale_x,
                            ymax=o.bbox.ymax * scale_y,
                        ),
                    )
                )
            preprocess_ms = (t_pre - t0) * 1000.0
            detect_ms = (t_inv - t_pre + (time.perf_counter() - t_inv)) * 1000.0
            return scaled_objs, preprocess_ms, detect_ms
        else:
            raise ValueError(f"Unknown detector type: {self.detector_type}")

    def process_frame(
        self, frame_bgr: np.ndarray, frame_idx: int = 0
    ) -> tuple[np.ndarray, FrameTiming]:
        """Processes a single frame and returns (annotated_frame, timing_metrics)."""
        t_start = time.perf_counter()
        orig_h, orig_w = frame_bgr.shape[:2]

        t_conv = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        color_conv_ms = (time.perf_counter() - t_conv) * 1000.0

        objs, preprocess_ms, detect_ms = self.detect_objects(frame_rgb, (orig_w, orig_h))
        preprocess_ms += color_conv_ms

        classify_ms = 0.0
        draw_ms = 0.0

        crops_classified = 0
        for obj in objs:
            label_name = self.labels_coco.get(obj.id, str(obj.id))
            if label_name != "bird":
                continue

            x0 = max(0, int(obj.bbox.xmin))
            y0 = max(0, int(obj.bbox.ymin))
            x1 = min(orig_w, int(obj.bbox.xmax))
            y1 = min(orig_h, int(obj.bbox.ymax))

            img_bird = frame_rgb[y0:y1, x0:x1, :]
            if img_bird.size == 0:
                continue

            t_cls0 = time.perf_counter()
            bird_classes = self.engine_pigeon.classify(
                img_bird, top_k=self.top_k, threshold=self.score_threshold
            )
            classify_ms += (time.perf_counter() - t_cls0) * 1000.0
            crops_classified += 1

            if not bird_classes:
                continue

            bird_top_1 = self.labels_bird.get(bird_classes[0].id, str(bird_classes[0].id))
            if "Dove" in bird_top_1 or "Pigeon" in bird_top_1:
                self.metrics.pigeons_detected += 1
                self.on_pigeon_detected()

            t_draw0 = time.perf_counter()
            draw_detection(frame_bgr, x0, y0, x1, y1, bird_top_1)
            draw_ms += (time.perf_counter() - t_draw0) * 1000.0

            for c in bird_classes:
                bird_label = self.labels_bird.get(c.id, str(c.id))
                log.debug(f"Frame {frame_idx:06d}: {bird_label} ({c.score:.3f})")

        t_end = time.perf_counter()
        proc_time = t_end - t_start

        t_draw0 = time.perf_counter()
        draw_debug_info(frame_bgr, frame_idx, proc_time)
        draw_ms += (time.perf_counter() - t_draw0) * 1000.0

        total_ms = (time.perf_counter() - t_start) * 1000.0

        timing = FrameTiming(
            read_ms=0.0,
            preprocess_ms=preprocess_ms,
            detection_ms=detect_ms,
            classification_ms=classify_ms,
            crops_classified=crops_classified,
            draw_ms=draw_ms,
            total_ms=total_ms,
        )
        return frame_bgr, timing

    def run(
        self,
        input_source: str | int,
        output_path: str | None = None,
        loop: bool = False,
        fps: float | None = None,
        max_frames: int | None = None,
    ) -> PipelineMetrics:
        """Runs pipeline on video stream, directory of images, or file."""
        self.metrics = PipelineMetrics()
        default_fps = 1.0 if fps is None else fps
        reader = VideoReader(input_source, loop=loop, default_image_fps=default_fps)

        vid_width = reader.width
        vid_height = reader.height
        cap_fps = fps if fps is not None else reader.fps

        file_writer = None
        if output_path:
            out_dir = os.path.dirname(os.path.abspath(output_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
            file_writer = cv2.VideoWriter(output_path, fourcc, cap_fps, (vid_width, vid_height))

        try:
            for frame_idx, frame, read_ms in reader.read_frames(max_frames=max_frames):
                processed_frame, timing = self.process_frame(frame, frame_idx)
                timing.read_ms = read_ms
                timing.total_ms += read_ms
                self.metrics.timings.append(timing)
                self.metrics.total_frames += 1
                self.metrics.total_crops_classified += timing.crops_classified

                if file_writer is not None:
                    file_writer.write(processed_frame)
        finally:
            reader.release()
            if file_writer is not None:
                file_writer.release()

        return self.metrics
