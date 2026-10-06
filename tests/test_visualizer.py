"""Tests for annotation drawing, CSV output and video writer helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from config import VisualizationConfig
from utils import TracksCsvWriter, Visualizer, open_writer, release_writer


class TestTracksCsvWriter:
    """The CSV writer emits the documented schema, one row per track."""

    def test_header_and_rows(self, tmp_path: Path, detection_factory) -> None:
        path = tmp_path / "out" / "tracking.csv"
        writer = TracksCsvWriter(path)
        writer.write(0, [])  # header only
        assert writer.row_count == 0

        from utils import BYTETracker
        from config import TrackerConfig

        tracker = BYTETracker(TrackerConfig())
        for frame_id in range(3):
            det = detection_factory(x=100.0 + 4.0 * frame_id)
            writer.write(frame_id, tracker.update([det], frame_id=frame_id))
        writer.close()

        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        assert rows[0] == [
            "frame",
            "track_id",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "age",
        ]
        assert len(rows) == 1 + writer.row_count
        assert writer.row_count >= 3
        assert all(len(row) == 10 for row in rows)

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        writer = TracksCsvWriter(tmp_path / "t.csv")
        writer.write(0, [])
        writer.close()
        writer.close()  # must not raise
        assert writer._handle is None

    def test_context_manager_closes(self, tmp_path: Path) -> None:
        path = tmp_path / "ctx.csv"
        with TracksCsvWriter(path) as writer:
            writer.write(0, [])
        assert path.exists()
        assert writer._handle is None


class TestVisualizer:
    """Drawing keeps frame geometry intact and colours are stable."""

    def test_draw_preserves_shape(self, detection_factory) -> None:
        from utils import BYTETracker
        from config import TrackerConfig

        frame = np.full((240, 320, 3), 50, dtype=np.uint8)
        tracker = BYTETracker(TrackerConfig())
        tracks = tracker.update([detection_factory()], frame_id=0)
        out = Visualizer().draw(frame, tracks, fps=12.5)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_draw_changes_pixels_when_tracks_exist(self, detection_factory) -> None:
        """Annotations must actually land on the frame.

        ``draw()`` mutates its input when trails are disabled
        (``canvas = frame`` in visualizer), so the pristine copy must be
        taken before the call.
        """
        from utils import BYTETracker
        from config import TrackerConfig

        frame = np.full((240, 320, 3), 50, dtype=np.uint8)
        pristine = frame.copy()
        tracker = BYTETracker(TrackerConfig())
        tracks = tracker.update([detection_factory()], frame_id=0)
        assert tracks, "fixture produced no tracks to draw"
        out = Visualizer().draw(frame, tracks, fps=0.0)
        assert not np.array_equal(out, pristine)

    def test_track_color_deterministic(self) -> None:
        first = Visualizer.track_color(7)
        second = Visualizer.track_color(7)
        assert first == second
        assert first != Visualizer.track_color(8)

    def test_color_for_caches(self) -> None:
        vis = Visualizer()
        assert vis.color_for(3) == vis.color_for(3)


class TestOpenWriter:
    """Codec fallback produces a playable file."""

    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "clip.mp4"
        writer = open_writer(path, fps=10.0, frame_size=(64, 48))
        assert writer is not None
        frame = np.full((48, 64, 3), 120, dtype=np.uint8)
        for _ in range(5):
            writer.write(frame)
        release_writer(writer)
        assert path.exists() and path.stat().st_size > 0

        cap = cv2.VideoCapture(str(path))
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
        cap.release()

    def test_invalid_fps_clamped(self, tmp_path: Path) -> None:
        """A zero FPS must be repaired rather than rejected."""
        writer = open_writer(tmp_path / "z.mp4", fps=0.0, frame_size=(64, 48))
        assert writer is not None
        release_writer(writer)

    def test_release_none_is_safe(self) -> None:
        release_writer(None)  # must not raise


class TestVisualizerConfig:
    """Rendering config validation is enforced at construction."""

    def test_bad_thickness_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="thickness"):
            Visualizer(VisualizationConfig(thickness=0))
