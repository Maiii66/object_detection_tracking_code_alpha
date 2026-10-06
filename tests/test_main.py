"""Tests for CLI helpers and output-path resolution in ``main.py``."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from config import AppConfig, OutputConfig, build_parser
from main import (
    _extract_class_tokens,
    _resolve_output_paths,
    _sanitised_fps,
    build_cli,
    open_source,
)


class TestExtractClassTokens:
    """``--classes`` accepts names, ids and ``=`` syntax."""

    def test_names_pulled_out(self) -> None:
        cleaned, tokens = _extract_class_tokens(["--source", "a.mp4", "--classes", "person", "car"])
        assert cleaned == ["--source", "a.mp4"]
        assert tokens == ["person", "car"]

    def test_equals_syntax(self) -> None:
        cleaned, tokens = _extract_class_tokens(["--classes=person", "--source", "x"])
        assert tokens == ["person"]
        assert cleaned == ["--source", "x"]

    def test_absent_option(self) -> None:
        cleaned, tokens = _extract_class_tokens(["--source", "x"])
        assert cleaned == ["--source", "x"]
        assert tokens == []

    def test_stops_at_next_flag(self) -> None:
        cleaned, tokens = _extract_class_tokens(["--classes", "0", "--imgsz", "320"])
        assert tokens == ["0"]
        assert cleaned == ["--imgsz", "320"]


class TestSanitisedFps:
    """Bogus container FPS values fall back to 25."""

    def test_valid_passthrough(self) -> None:
        assert _sanitised_fps(30.0) == 30.0

    @pytest.mark.parametrize("bad", [float("nan"), 0.0, -5.0])
    def test_invalid_falls_back(self, bad: float) -> None:
        assert _sanitised_fps(bad) == 25.0
        assert not math.isnan(_sanitised_fps(bad))


class TestResolveOutputPaths:
    """Naming rules for the annotated video and the tracking CSV."""

    def _cfg(self, **kwargs) -> AppConfig:
        return AppConfig(**kwargs)

    def test_video_names_output_after_source(self) -> None:
        cfg = self._cfg(source="clip.mp4")
        video, csv = _resolve_output_paths(cfg, None, "video")
        assert video is not None and video.name == "clip_tracked.mp4"
        assert csv.name == "tracking.csv"

    def test_image_still_resolves_video_path(self) -> None:
        """``run()`` skips the writer for images; the stem is still computed."""
        cfg = self._cfg(source="photo.jpg")
        video, _ = _resolve_output_paths(cfg, None, "image")
        assert video is not None and video.name == "photo_tracked.mp4"

    def test_no_video_returns_none(self) -> None:
        cfg = self._cfg(source="clip.mp4", output=OutputConfig(save_video=False))
        video, csv = _resolve_output_paths(cfg, None, "video")
        assert video is None
        assert csv.name == "tracking.csv"

    def test_output_file_with_suffix(self, tmp_path: Path) -> None:
        cfg = self._cfg(source="clip.mp4")
        target = tmp_path / "custom.mp4"
        video, csv = _resolve_output_paths(cfg, str(target), "video")
        assert video == target
        assert csv == tmp_path / "tracking.csv"
        assert cfg.output.output_dir == tmp_path

    def test_output_file_without_suffix_is_directory(self, tmp_path: Path) -> None:
        cfg = self._cfg(source="clip.mp4")
        target = tmp_path / "results"
        video, csv = _resolve_output_paths(cfg, str(target), "video")
        assert video == target / "result.mp4"
        assert csv == target / "tracking.csv"

    def test_unknown_source_uses_result_stem(self) -> None:
        cfg = self._cfg(source="0")
        video, _ = _resolve_output_paths(cfg, None, "webcam")
        assert video is not None and video.name == "result.mp4"


class TestBuildCli:
    """Presentation-only flags parse and land on the right attributes."""

    def test_no_display_flag(self) -> None:
        parser, tokens = build_cli(["--source", "a.mp4", "--no-display"])
        args = parser.parse_args(["--source", "a.mp4", "--no-display"])
        assert args.display is False
        assert tokens == []

    def test_output_flag(self) -> None:
        parser, _ = build_cli(["--output", "out.mp4"])
        args = parser.parse_args(["--output", "out.mp4"])
        assert args.output_file == "out.mp4"

    def test_classes_tokens_extracted_before_argparse(self) -> None:
        _, tokens = build_cli(["--classes", "person", "car"])
        assert tokens == ["person", "car"]

    def test_profile_flag_defaults_off(self) -> None:
        parser, _ = build_cli([])
        assert parser.parse_args([]).profile is False
        parser2, _ = build_cli(["--profile"])
        assert parser2.parse_args(["--profile"]).profile is True

    def test_no_stats_flag(self) -> None:
        """``--no-stats`` lands on ``save_stats`` for the config mapper."""
        parser, _ = build_cli(["--no-stats"])
        args = parser.parse_args(["--no-stats"])
        assert args.save_stats is False

    def test_base_parser_rejects_unknown_flag(self) -> None:
        parser, _ = build_cli([])
        with pytest.raises(SystemExit):
            parser.parse_args(["--bogus"])


class TestOpenSource:
    """Sources open through ``cv2.VideoCapture`` with sane metadata."""

    def test_video_file(self, tiny_video: Path) -> None:
        cfg = AppConfig(source=str(tiny_video))
        capture, fps, (width, height) = open_source(cfg)
        try:
            assert capture.isOpened()
            assert fps == pytest.approx(10.0)
            assert (width, height) == (64, 48)
        finally:
            capture.release()

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        cfg = AppConfig(source=str(tmp_path / "missing.mp4"))
        with pytest.raises((FileNotFoundError, ValueError)):
            open_source(cfg)


def test_base_parser_defaults_none() -> None:
    """``build_parser`` keeps every default ``None`` (YAML interop)."""
    args = build_parser().parse_args([])
    assert all(v is None for k, v in vars(args).items() if k != "config")
