from __future__ import annotations

import unittest

from pigeon.models.yolo import BBox, Object, coco80_to_coco91_class


class TestModels(unittest.TestCase):
    def test_coco_mapping(self):
        # 0 in COCO 80 is person (index 1 in COCO 91)
        self.assertEqual(coco80_to_coco91_class(0), 1)
        # 14 in COCO 80 is bird (index 16 in COCO 91)
        self.assertEqual(coco80_to_coco91_class(14), 16)

    def test_bbox_structure(self):
        bbox = BBox(xmin=10.0, ymin=20.0, xmax=50.0, ymax=60.0)
        obj = Object(id=15, score=0.95, bbox=bbox)
        self.assertEqual(obj.id, 15)
        self.assertEqual(obj.score, 0.95)
        self.assertEqual(obj.bbox.xmin, 10.0)
        self.assertEqual(obj.bbox.ymax, 60.0)


if __name__ == "__main__":
    unittest.main()
