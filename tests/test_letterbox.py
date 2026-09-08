from __future__ import annotations

import unittest

import numpy as np

from pigeon.models.letterbox import letterbox


class TestLetterbox(unittest.TestCase):
    def test_letterbox_square(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        out, ratio, pad = letterbox(img, new_shape=(320, 320), auto=False)
        self.assertEqual(out.shape, (320, 320, 3))
        self.assertEqual(ratio, 0.5)
        self.assertEqual(pad[0], 0.0)
        self.assertEqual(pad[1], 40.0)

    def test_letterbox_tall(self):
        img = np.zeros((640, 320, 3), dtype=np.uint8)
        out, ratio, pad = letterbox(img, new_shape=(320, 320), auto=False)
        self.assertEqual(out.shape, (320, 320, 3))
        self.assertEqual(ratio, 0.5)
        self.assertEqual(pad[0], 80.0)
        self.assertEqual(pad[1], 0.0)


if __name__ == "__main__":
    unittest.main()
