from __future__ import annotations

import os


def read_label_file(file_path: str) -> dict[int, str]:
    """Reads label file and returns mapping from index to class name."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Label file not found: {file_path}")
    labels = {}
    with open(file_path, encoding="utf-8") as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip()
            if not line:
                continue
            # Some Coral label files have format "0 bird" while others are just "bird" per line
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                labels[int(parts[0])] = parts[1].strip()
            else:
                labels[i] = line
    return labels
