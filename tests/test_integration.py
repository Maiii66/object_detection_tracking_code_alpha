"""End-to-end pipeline tests: real model, tiny generated inputs.

Skipped automatically when ``models/yolov8n.pt`` is absent (it is gitignored,
so fresh clones must download weights or run with network access first).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config import AppConfig, DetectorConfig, OutputConfig
from main import run

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "yolov8n.pt"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not MODEL.exists(), reason="models/yolov8n.pt not present"),
]


def _cfg(source: Path, out_dir: Path) -> AppConfig:
    """Build a validated config pointed at a temp output directory."""
    cfg = AppConfig(
        source=str(source),
        detector=DetectorConfig(imgsz=320),
        output=OutputConfig(output_dir=out_dir),
    )
    return cfg.validate()


class TestEndToEnd:
    """``run()`` returns 0 and writes the documented artefacts."""

    def test_video_source(self, tiny_video: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        code = run(_cfg(tiny_video, out_dir), display=False)
        assert code == 0
        produced = out_dir / "tiny_tracked.mp4"
        assert produced.exists() and produced.stat().st_size > 0
        assert (out_dir / "tracking.csv").exists()

    def test_image_source_writes_png(self, tiny_image: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        code = run(_cfg(tiny_image, out_dir), display=False)
        assert code == 0
        png = out_dir / "result.png"
        assert png.exists() and png.stat().st_size > 0

    def test_no_video_flag_suppresses_mp4(self, tiny_video: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        cfg = _cfg(tiny_video, out_dir)
        cfg.output.save_video = False
        code = run(cfg, display=False)
        assert code == 0
        assert not list(out_dir.glob("*.mp4"))
        assert (out_dir / "tracking.csv").exists()

    def test_stats_and_profile_written(self, tiny_video: Path, tmp_path: Path) -> None:
        """``profile=True`` writes both artefacts with plausible content."""
        import json

        out_dir = tmp_path / "out"
        code = run(_cfg(tiny_video, out_dir), display=False, profile=True)
        assert code == 0

        report = json.loads((out_dir / "run_stats.json").read_text(encoding="utf-8"))
        assert report["frames"] == 10
        assert report["source_kind"] == "video"
        assert report["outputs"]["video"].endswith("tiny_tracked.mp4")
        assert report["finished_at"]

        pstats_file = out_dir / "profile.pstats"
        assert pstats_file.exists() and pstats_file.stat().st_size > 0

    def test_no_stats_flag_suppresses_json(self, tiny_video: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        cfg = _cfg(tiny_video, out_dir)
        cfg.output.save_stats = False
        code = run(cfg, display=False)
        assert code == 0
        assert not (out_dir / "run_stats.json").exists()

    def test_bad_source_returns_error_code(self, tmp_path: Path) -> None:
        cfg = AppConfig(source=str(tmp_path / "missing.mp4"))
        cfg.output.output_dir = tmp_path / "out"
        assert run(cfg, display=False) == 1


def test_unreadable_source_returns_error_code(tmp_path: Path) -> None:
    """An unreadable/missing source exits with code 1, not a traceback."""
    cfg = AppConfig(source=str(tmp_path / "any.mp4"))
    cfg.output.output_dir = tmp_path / "out"
    assert run(cfg, display=False) == 1
