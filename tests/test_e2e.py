from __future__ import annotations

import os
import subprocess
import sys
import unittest

import cv2

from pigeon.backends.edgetpu import list_available_tpus


class TestE2E(unittest.TestCase):
    def setUp(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.test_image_dir = os.path.join(self.project_root, "data", "testdata", "pigeons")
        self.output_dir = os.path.join(self.project_root, "data", "testdata", "output")

        if not os.path.isdir(self.test_image_dir) or not os.listdir(self.test_image_dir):
            self.skipTest(f"Test images not found in {self.test_image_dir}. Run download.sh first.")

        self.test_images = [
            f
            for f in os.listdir(self.test_image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not self.test_images:
            self.skipTest("No supported images found in testdata directory.")

        os.makedirs(self.output_dir, exist_ok=True)

    def test_pipeline_cpu_on_testdata_directory(self):
        """Tests end-to-end video processing and output generation using CPU / LiteRT."""
        output_path = os.path.join(self.output_dir, "processed_pigeons_cpu.mp4")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass

        command = [
            sys.executable,
            "-m",
            "pigeon.cli",
            "-i",
            self.test_image_dir,
            "-o",
            output_path,
            "--backend",
            "tflite",
            "--fps",
            "1.0",
            "-v",
            "0",
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(self.project_root, "src")

        result = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=120, check=False
        )
        self.assertEqual(
            result.returncode,
            0,
            f"CPU CLI pipeline run failed: {result.stderr}",
        )

        self.assertTrue(
            os.path.exists(output_path), f"Output video was not created at {output_path}"
        )
        self.assertGreater(os.path.getsize(output_path), 0, "Output video file is empty")

        cap = cv2.VideoCapture(output_path)
        self.assertTrue(cap.isOpened(), "Could not open generated output video with OpenCV")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        self.assertEqual(
            frame_count,
            len(self.test_images),
            f"Expected {len(self.test_images)} frames in video, got {frame_count}",
        )
        self.assertAlmostEqual(fps, 1.0, delta=0.1, msg=f"Expected 1.0 FPS, got {fps}")

    def test_pipeline_edgetpu_dual_on_testdata_directory(self):
        """Tests end-to-end video processing and output generation using dual Coral Edge TPUs."""
        tpus = list_available_tpus()
        if not tpus:
            self.skipTest("No Coral Edge TPU hardware detected.")

        tpu_0 = ":0"
        tpu_1 = ":1" if len(tpus) > 1 else ":0"

        output_path = os.path.join(self.output_dir, "processed_pigeons_edgetpu.mp4")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass

        command = [
            sys.executable,
            "-m",
            "pigeon.cli",
            "-i",
            self.test_image_dir,
            "-o",
            output_path,
            "--backend",
            "edgetpu",
            "--tpu-0",
            tpu_0,
            "--tpu-1",
            tpu_1,
            "--fps",
            "1.0",
            "-v",
            "0",
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(self.project_root, "src")

        result = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=120, check=False
        )
        self.assertEqual(
            result.returncode,
            0,
            f"Edge TPU CLI pipeline run failed: {result.stderr}",
        )

        self.assertTrue(
            os.path.exists(output_path), f"Output video was not created at {output_path}"
        )
        self.assertGreater(os.path.getsize(output_path), 0, "Output video file is empty")

        cap = cv2.VideoCapture(output_path)
        self.assertTrue(cap.isOpened(), "Could not open generated output video with OpenCV")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        self.assertEqual(
            frame_count,
            len(self.test_images),
            f"Expected {len(self.test_images)} frames in video, got {frame_count}",
        )
        self.assertAlmostEqual(fps, 1.0, delta=0.1, msg=f"Expected 1.0 FPS, got {fps}")


if __name__ == "__main__":
    unittest.main()
