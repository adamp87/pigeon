from __future__ import annotations

import os
import unittest

import cv2

from pigeon.backends.edgetpu import EdgeTPUEngine, list_available_tpus
from pigeon.backends.tflite import TFLiteEngine
from pigeon.pipeline import PigeonPipeline
from pigeon.utils.labels import read_label_file


class TestBenchmark(unittest.TestCase):
    def setUp(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.test_image_dir = os.path.join(self.project_root, "data", "testdata", "pigeons")
        self.coco_labels_path = os.path.join(self.project_root, "data", "models", "coco_labels.txt")
        self.bird_labels_path = os.path.join(
            self.project_root, "data", "models", "inat_bird_labels.txt"
        )
        self.detect_model_cpu = os.path.join(
            self.project_root, "data", "models", "ssdlite_mobiledet_coco_qat_postprocess.tflite"
        )
        self.bird_model_cpu = os.path.join(
            self.project_root, "data", "models", "mobilenet_v2_1.0_224_inat_bird_quant.tflite"
        )
        self.detect_model_tpu = os.path.join(
            self.project_root,
            "data",
            "models",
            "ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite",
        )
        self.bird_model_tpu = os.path.join(
            self.project_root,
            "data",
            "models",
            "mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite",
        )

        if not os.path.isdir(self.test_image_dir) or not os.listdir(self.test_image_dir):
            self.skipTest("Test images not found in data/testdata/pigeons.")

        # Warm up filesystem cache so all benchmarks run under identical conditions
        for fname in os.listdir(self.test_image_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                _ = cv2.imread(os.path.join(self.test_image_dir, fname))

    def test_pipeline_benchmark_cpu(self):
        """Tests and benchmarks pipeline execution without TPU (using CPU / LiteRT engine)."""
        detect_model = (
            self.detect_model_cpu
            if os.path.exists(self.detect_model_cpu)
            else self.detect_model_tpu
        )
        bird_model = (
            self.bird_model_cpu if os.path.exists(self.bird_model_cpu) else self.bird_model_tpu
        )

        if not os.path.exists(detect_model) or not os.path.exists(bird_model):
            self.skipTest("CPU TFLite models not found in data/models.")

        engine_detect = TFLiteEngine(detect_model)
        engine_pigeon = TFLiteEngine(bird_model)
        labels_coco = read_label_file(self.coco_labels_path)
        labels_bird = read_label_file(self.bird_labels_path)

        pipeline = PigeonPipeline(
            engine_detect=engine_detect,
            engine_pigeon=engine_pigeon,
            labels_coco=labels_coco,
            labels_bird=labels_bird,
            detector_type="ssd",
            score_threshold=0.4,
            top_k=2,
        )

        max_benchmark_frames = 10
        metrics = pipeline.run(
            input_source=self.test_image_dir,
            output_path=None,
            loop=True,
            max_frames=max_benchmark_frames,
        )

        print("\n--- CPU Pipeline Benchmark Summary ---")
        print(metrics.summary())

        self.assertEqual(metrics.total_frames, max_benchmark_frames)
        self.assertGreaterEqual(metrics.pigeons_detected, 1, "Expected at least 1 pigeon detection")
        self.assertGreater(metrics.avg_read_ms, 0.0)
        self.assertGreater(metrics.avg_preprocess_ms, 0.0)
        self.assertGreater(metrics.avg_detection_ms, 0.0)
        self.assertGreater(metrics.avg_classification_ms, 0.0)
        self.assertGreater(metrics.avg_total_ms, 0.0)
        self.assertGreater(metrics.fps, 0.0)

    def test_pipeline_benchmark_edgetpu_single(self):
        """Tests and benchmarks on a single Coral Edge TPU (TPU detector + CPU classifier)."""

        tpus = list_available_tpus()
        if not tpus:
            self.skipTest("No Coral Edge TPU hardware devices detected.")

        if not os.path.exists(self.detect_model_tpu):
            self.skipTest("Edge TPU model not found in data/models.")

        bird_model = (
            self.bird_model_cpu if os.path.exists(self.bird_model_cpu) else self.bird_model_tpu
        )

        try:
            engine_detect = EdgeTPUEngine(self.detect_model_tpu, device=":0")
            engine_pigeon = TFLiteEngine(bird_model)
        except Exception as e:
            self.skipTest(f"Failed to initialize Edge TPU engine: {e}")

        labels_coco = read_label_file(self.coco_labels_path)
        labels_bird = read_label_file(self.bird_labels_path)

        pipeline = PigeonPipeline(
            engine_detect=engine_detect,
            engine_pigeon=engine_pigeon,
            labels_coco=labels_coco,
            labels_bird=labels_bird,
            detector_type="ssd",
            score_threshold=0.4,
            top_k=2,
        )

        max_benchmark_frames = 10
        metrics = pipeline.run(
            input_source=self.test_image_dir,
            output_path=None,
            loop=True,
            max_frames=max_benchmark_frames,
        )

        print("\n--- Single Edge TPU (:0 Detect + CPU Classify) Benchmark Summary ---")
        print(metrics.summary())

        self.assertEqual(metrics.total_frames, max_benchmark_frames)
        self.assertGreaterEqual(metrics.pigeons_detected, 1, "Expected at least 1 pigeon detection")
        self.assertGreater(metrics.avg_detection_ms, 0.0)
        self.assertGreater(metrics.avg_classification_ms, 0.0)
        self.assertGreater(metrics.fps, 0.0)

    def test_pipeline_benchmark_edgetpu_dual(self):
        """Tests and benchmarks distributing detection to TPU 0 and classification to TPU 1."""

        tpus = list_available_tpus()
        if len(tpus) < 2:
            self.skipTest(f"Dual Edge TPU test requires >= 2 Coral TPUs, found {len(tpus)}.")

        if not os.path.exists(self.detect_model_tpu) or not os.path.exists(self.bird_model_tpu):
            self.skipTest("Edge TPU models not found in data/models.")

        try:
            engine_detect = EdgeTPUEngine(self.detect_model_tpu, device=":0")
            engine_pigeon = EdgeTPUEngine(self.bird_model_tpu, device=":1")
        except Exception as e:
            self.skipTest(f"Failed to initialize Dual Edge TPU engines: {e}")

        labels_coco = read_label_file(self.coco_labels_path)
        labels_bird = read_label_file(self.bird_labels_path)

        pipeline = PigeonPipeline(
            engine_detect=engine_detect,
            engine_pigeon=engine_pigeon,
            labels_coco=labels_coco,
            labels_bird=labels_bird,
            detector_type="ssd",
            score_threshold=0.4,
            top_k=2,
        )

        max_benchmark_frames = 10
        metrics = pipeline.run(
            input_source=self.test_image_dir,
            output_path=None,
            loop=True,
            max_frames=max_benchmark_frames,
        )

        print("\n--- Dual Edge TPU (:0 Detect, :1 Classify) Benchmark Summary ---")
        print(metrics.summary())

        self.assertEqual(metrics.total_frames, max_benchmark_frames)
        self.assertGreaterEqual(metrics.pigeons_detected, 1, "Expected at least 1 pigeon detection")
        self.assertGreater(metrics.avg_detection_ms, 0.0)
        self.assertGreater(metrics.avg_classification_ms, 0.0)
        self.assertGreater(metrics.fps, 0.0)


if __name__ == "__main__":
    unittest.main()
