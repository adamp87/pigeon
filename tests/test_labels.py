from __future__ import annotations

import tempfile
import unittest

from pigeon.utils.labels import read_label_file


class TestLabels(unittest.TestCase):
    def test_read_label_file_plain(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("background\nperson\nbird\ncat\n")
            f.flush()
            labels = read_label_file(f.name)
            self.assertEqual(labels[0], "background")
            self.assertEqual(labels[1], "person")
            self.assertEqual(labels[2], "bird")
            self.assertEqual(labels[3], "cat")

    def test_read_label_file_indexed(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("0 background\n1 person\n2 bird\n")
            f.flush()
            labels = read_label_file(f.name)
            self.assertEqual(labels[0], "background")
            self.assertEqual(labels[1], "person")
            self.assertEqual(labels[2], "bird")


if __name__ == "__main__":
    unittest.main()
