"""Shared fixtures: synthetic detections and tiny generated media files."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import Detection  # noqa: E402  (needs ROOT on sys.path first)


@pytest.fixture
def detection_factory() -> Callable[..., Detection]:
    """Factory building a :class:`~utils.Detection` with a top-left box."""

    def _make(
        x: float = 100.0,
        y: float = 100.0,
        w: float = 40.0,
        h: float = 80.0,
        score: float = 0.9,
        class_id: int = 0,
        class_name: str = "person",
    ) -> Detection:
        tlbr = np.array([x, y, x + w, y + h], dtype=np.float32)
        return Detection(tlbr=tlbr, score=score, class_id=class_id, class_name=class_name)

    return _make


@pytest.fixture
def tiny_video(tmp_path: Path) -> Path:
    """A 10-frame 64x48 MP4 with a moving square, for fast end-to-end runs."""
    path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    assert writer.isOpened(), "OpenCV could not open a writer for the fixture"
    for i in range(10):
        frame = np.full((48, 64, 3), 30, dtype=np.uint8)
        cv2.rectangle(frame, (5 + 5 * i, 14), (25 + 5 * i, 44), (220, 220, 220), -1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def tiny_image(tmp_path: Path) -> Path:
    """A 64x48 PNG with a bright rectangle, for image-source tests."""
    path = tmp_path / "tiny.png"
    img = np.full((48, 64, 3), 40, dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (50, 40), (230, 230, 230), -1)
    assert cv2.imwrite(str(path), img)
    return path
