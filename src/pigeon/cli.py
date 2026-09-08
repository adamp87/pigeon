from __future__ import annotations

import argparse
import logging as log
import os

from pigeon.backends.tflite import TFLiteEngine
from pigeon.pipeline import PigeonPipeline
from pigeon.utils.labels import read_label_file


def find_file(relative_paths):
    for p in relative_paths:
        if os.path.exists(p):
            return p
    return relative_paths[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pigeon Detection & Repellent Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path of video/image or camera index / URL to process"
    )
    parser.add_argument(
        "-o", "--output", help="File path for the result video/image with annotations"
    )
    parser.add_argument(
        "-v", "--verbose", type=int, default=1, help="Verbosity: 0 (warnings), 1 (info), 2 (debug)"
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.5, help="Score threshold for detections"
    )
    parser.add_argument(
        "-k", "--top_k", type=int, default=2, help="Max number of classification results"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output FPS (default: 1.0 for image folders, source fps for video)",
    )
    parser.add_argument(
        "--loop", action="store_true", help="Loop input video or image directory sequence"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Maximum number of frames to process"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Print detailed latency benchmark report upon completion",
    )
    parser.add_argument(
        "-b",
        "--backend",
        choices=["auto", "edgetpu", "tflite"],
        default="auto",
        help="Inference backend: 'edgetpu' (PyCoral), 'tflite' (LiteRT / CPU), or 'auto'",
    )
    parser.add_argument(
        "-d",
        "--detector",
        choices=["ssd", "yolo26", "yolov7", "yolov8"],
        default="ssd",
        help="Object detector architecture",
    )
    parser.add_argument("--model-detect", help="Path to object detection model (.tflite)")
    parser.add_argument("--model-pigeon", help="Path to bird classification model (.tflite)")
    parser.add_argument(
        "--labels-coco",
        default=find_file(["data/models/coco_labels.txt", "models/coco_labels.txt"]),
        help="Path to COCO labels",
    )
    parser.add_argument(
        "--labels-bird",
        default=find_file(["data/models/inat_bird_labels.txt", "models/inat_bird_labels.txt"]),
        help="Path to bird labels",
    )
    parser.add_argument(
        "--tpu-0", default=None, help="Coral TPU device path for detection (e.g. :0)"
    )
    parser.add_argument(
        "--tpu-1", default=None, help="Coral TPU device path for classification (e.g. :1)"
    )
    return parser.parse_args()


def resolve_backend(requested_backend: str):
    if requested_backend == "edgetpu":
        from pigeon.backends.edgetpu import EdgeTPUEngine, list_available_tpus

        return "edgetpu", EdgeTPUEngine, list_available_tpus
    elif requested_backend == "tflite":
        return "tflite", TFLiteEngine, None
    else:  # auto
        try:
            from pigeon.backends.edgetpu import EdgeTPUEngine, list_available_tpus

            tpus = list_available_tpus()
            if len(tpus) > 0:
                log.info(f"Found {len(tpus)} Coral TPU device(s). Using EdgeTPU backend.")
                return "edgetpu", EdgeTPUEngine, list_available_tpus
        except Exception:
            pass
        log.info("Using LiteRT / CPU backend.")
        return "tflite", TFLiteEngine, None


def main():
    args = parse_args()

    if args.verbose == 0:
        log.root.setLevel(log.WARNING)
    elif args.verbose == 1:
        log.root.setLevel(log.INFO)
    else:
        log.root.setLevel(log.DEBUG)

    backend_type, engine_cls, list_tpus_fn = resolve_backend(args.backend)

    # Set default models based on detector and backend
    if not args.model_detect:
        if backend_type == "edgetpu":
            if args.detector == "ssd":
                args.model_detect = find_file(
                    [
                        "data/models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite",
                        "models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite",
                    ]
                )
            elif args.detector == "yolov7":
                args.model_detect = find_file(
                    [
                        "data/models/yolov7tiny_relu6.tflite",
                        "models/yolov7tiny_relu6.tflite",
                    ]
                )
            elif args.detector == "yolov8":
                args.model_detect = find_file(
                    [
                        "data/models/yolov8n_relu6.tflite",
                        "models/yolov8n_relu6.tflite",
                    ]
                )
            elif args.detector == "yolo26":
                args.model_detect = find_file(
                    [
                        "data/models/yolo26n-det-int8.tflite",
                        "models/yolo26n-det-int8.tflite",
                    ]
                )
        else:
            if args.detector == "yolo26":
                args.model_detect = find_file(
                    [
                        "data/models/yolo26n-det-int8.tflite",
                        "models/yolo26n-det-int8.tflite",
                    ]
                )
            elif args.detector == "ssd":
                args.model_detect = find_file(
                    [
                        "data/models/ssdlite_mobiledet_coco_qat_postprocess.tflite",
                        "models/ssdlite_mobiledet_coco_qat_postprocess.tflite",
                        "data/models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite",
                    ]
                )
            elif args.detector == "yolov7":
                args.model_detect = find_file(
                    [
                        "data/models/yolov7tiny_relu6.tflite",
                        "models/yolov7tiny_relu6.tflite",
                    ]
                )
            elif args.detector == "yolov8":
                args.model_detect = find_file(
                    [
                        "data/models/yolov8n_relu6.tflite",
                        "models/yolov8n_relu6.tflite",
                    ]
                )

    if not args.model_pigeon:
        if backend_type == "edgetpu":
            args.model_pigeon = find_file(
                [
                    "data/models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite",
                    "models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite",
                ]
            )
        else:
            args.model_pigeon = find_file(
                [
                    "data/models/mobilenet_v2_1.0_224_inat_bird_quant.tflite",
                    "models/mobilenet_v2_1.0_224_inat_bird_quant.tflite",
                    "data/models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite",
                ]
            )

    # Instantiate engines
    if backend_type == "edgetpu":
        tpus = list_tpus_fn()
        tpu_0 = args.tpu_0 or (":0" if len(tpus) > 0 else None)
        tpu_1 = args.tpu_1 or (":1" if len(tpus) > 1 else tpu_0)
        log.info(f"EdgeTPU detect device: {tpu_0}, classify device: {tpu_1}")
        engine_detect = engine_cls(args.model_detect, device=tpu_0)
        engine_pigeon = engine_cls(args.model_pigeon, device=tpu_1)
    else:
        engine_detect = engine_cls(args.model_detect)
        engine_pigeon = engine_cls(args.model_pigeon)

    labels_coco = read_label_file(args.labels_coco)
    labels_bird = read_label_file(args.labels_bird)

    pipeline = PigeonPipeline(
        engine_detect=engine_detect,
        engine_pigeon=engine_pigeon,
        labels_coco=labels_coco,
        labels_bird=labels_bird,
        detector_type=args.detector,
        score_threshold=args.threshold,
        top_k=args.top_k,
    )

    metrics = pipeline.run(
        args.input,
        output_path=args.output,
        loop=args.loop,
        fps=args.fps,
        max_frames=args.max_frames,
    )

    if args.benchmark or args.verbose >= 1:
        print(metrics.summary())


if __name__ == "__main__":
    main()
