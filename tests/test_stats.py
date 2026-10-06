"""Tests for the run-statistics collector (:mod:`utils.stats`)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from utils.stats import RunStats


def _make_stats() -> RunStats:
    return RunStats(
        source="clip.mp4",
        source_kind="video",
        device="cpu",
        model="yolov8n",
        imgsz=480,
    )


class TestRecording:
    """Per-frame accumulation of detections and track ids."""

    def test_counts_frames_and_detections(self) -> None:
        stats = _make_stats()
        det = SimpleNamespace(class_name="person")
        track = SimpleNamespace(track_id=1, last_class_name="person")
        for _ in range(5):
            stats.record([det], [track])
        assert stats.frames == 5
        assert stats.detections_total == 5
        assert stats.unique_track_ids == 1

    def test_per_class_breakdown(self) -> None:
        stats = _make_stats()
        stats.record(
            [SimpleNamespace(class_name="person"), SimpleNamespace(class_name="car")],
            [SimpleNamespace(track_id=1, last_class_name="person")],
        )
        stats.record(
            [SimpleNamespace(class_name="car")],
            [SimpleNamespace(track_id=2, last_class_name="car")],
        )
        data = stats.to_dict()
        assert data["per_class"]["person"] == {"detections": 1, "unique_track_ids": 1}
        assert data["per_class"]["car"] == {"detections": 2, "unique_track_ids": 1}
        assert data["unique_track_ids"] == 2

    def test_unknown_class_name_falls_back(self) -> None:
        stats = _make_stats()
        stats.record([SimpleNamespace()], [])
        assert stats.to_dict()["per_class"]["unknown"]["detections"] == 1

    def test_empty_run(self) -> None:
        data = _make_stats().to_dict()
        assert data["frames"] == 0
        assert data["unique_track_ids"] == 0
        assert data["per_class"] == {}


class TestSummaryAndWrite:
    """``set_summary`` stamps the run; ``write`` emits valid JSON."""

    def test_set_summary_fills_fields(self) -> None:
        stats = _make_stats()
        stats.set_summary(
            wall_seconds=1.5,
            avg_fps=20.0,
            mean_detect_ms=40.0,
            mean_track_ms=1.0,
            mean_draw_ms=3.0,
            outputs={"video": Path("/tmp/out.mp4"), "csv": None},
        )
        data = stats.to_dict()
        assert data["wall_seconds"] == 1.5
        assert data["avg_fps"] == 20.0
        assert data["finished_at"]
        # None entries are dropped; remaining paths are stringified.
        assert data["outputs"] == {"video": str(Path("/tmp/out.mp4"))}

    def test_write_produces_readable_json(self, tmp_path: Path) -> None:
        stats = _make_stats()
        stats.record([SimpleNamespace(class_name="bus")], [])
        out = stats.write(tmp_path / "nested" / "run_stats.json")
        assert out.exists()
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["source"] == "clip.mp4"
        assert loaded["model"] == "yolov8n"
        assert loaded["per_class"]["bus"]["detections"] == 1
        assert loaded["started_at"] and loaded["finished_at"] == ""
