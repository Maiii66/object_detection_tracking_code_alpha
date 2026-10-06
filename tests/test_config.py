"""Tests for configuration classification, validation and YAML/CLI merging."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from config import AppConfig, DetectorConfig, TrackerConfig, build_parser


class TestSourceClassification:
    """``AppConfig`` must classify each source kind correctly."""

    @pytest.mark.parametrize(
        ("source", "kind"),
        [
            ("0", "webcam"),
            ("rtsp://cam/stream", "stream"),
            ("http://cam/stream", "stream"),
            ("tcp://10.0.0.5:554/feed", "stream"),
            ("clip.mp4", "video"),
            ("photo.jpg", "image"),
        ],
    )
    def test_source_kind(self, source: str, kind: str) -> None:
        """Each source maps to exactly one of webcam/stream/video/image."""
        assert AppConfig(source=source).source_kind == kind

    def test_is_url_only_for_stream_schemes(self) -> None:
        """A plain file path is never treated as a URL."""
        cfg = AppConfig(source="clip.mp4")
        assert not cfg.is_url
        assert AppConfig(source="rtsp://x/y").is_url

    def test_is_image_requires_known_extension(self) -> None:
        """``.png`` counts as an image; ``.txt`` falls through to video."""
        assert AppConfig(source="a.png").is_image
        assert not AppConfig(source="a.txt").is_image


class TestValidation:
    """``validate()`` must reject bad values with a clear ``ValueError``."""

    def test_empty_source_rejected(self) -> None:
        with pytest.raises(ValueError, match="source must not be empty"):
            AppConfig(source="  ").validate()

    def test_missing_file_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="source file not found"):
            AppConfig(source=str(tmp_path / "nope.mp4")).validate()

    def test_existing_file_accepted(self, tiny_video: Path) -> None:
        cfg = AppConfig(source=str(tiny_video))
        cfg.validate()  # must not raise

    @pytest.mark.parametrize("conf", [0.0, 1.0, -0.5])
    def test_conf_out_of_range(self, conf: float) -> None:
        with pytest.raises(ValueError, match="detector.conf"):
            DetectorConfig(conf=conf).validate()

    @pytest.mark.parametrize("imgsz", [100, 0, -64])
    def test_imgsz_must_be_multiple_of_32(self, imgsz: int) -> None:
        with pytest.raises(ValueError, match="imgsz"):
            DetectorConfig(imgsz=imgsz).validate()

    def test_half_forced_off_on_cpu(self) -> None:
        """``half=True`` is coerced to False rather than failing."""
        cfg = DetectorConfig(half=True)
        cfg.validate()
        assert cfg.half is False

    def test_tracker_threshold_ordering(self) -> None:
        with pytest.raises(ValueError, match="track_low_thresh"):
            TrackerConfig(track_low_thresh=0.9, track_high_thresh=0.2).validate()

    def test_negative_track_buffer(self) -> None:
        with pytest.raises(ValueError, match="track_buffer"):
            TrackerConfig(track_buffer=0).validate()


class TestYamlAndCli:
    """YAML layers defaults, CLI wins on conflict."""

    def test_yaml_override(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "c.yaml"
        cfg_file.write_text("detector:\n  imgsz: 320\n", encoding="utf-8")
        cfg = AppConfig().merged_with_yaml(cfg_file)
        assert cfg.detector.imgsz == 320

    def test_unknown_yaml_key_rejected(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "c.yaml"
        cfg_file.write_text("detecotr:\n  imgsz: 320\n", encoding="utf-8")
        with pytest.raises(ValueError, match="detecotr"):
            AppConfig().merged_with_yaml(cfg_file)

    def test_cli_beats_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "c.yaml"
        cfg_file.write_text("source: from_yaml.mp4\n", encoding="utf-8")
        parser = build_parser()
        args = parser.parse_args(["--config", str(cfg_file), "--source", "cli.mp4"])
        cfg = AppConfig.from_args(args)
        assert cfg.source == "cli.mp4"

    def test_parser_defaults_are_none(self) -> None:
        """Defaults must stay ``None`` so YAML values are not clobbered."""
        args = build_parser().parse_args([])
        unset = [v for k, v in vars(args).items() if k != "config"]
        assert all(v is None for v in unset)

    def test_from_args_missing_config_file(self) -> None:
        args = argparse.Namespace(config=Path("does_not_exist.yaml"), source=None)
        with pytest.raises(FileNotFoundError):
            AppConfig.from_args(args)
