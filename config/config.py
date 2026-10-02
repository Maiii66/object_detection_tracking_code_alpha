"""Central configuration for the detection + tracking pipeline.

Design rules:
  * Every magic number in the project lives here. No literals in other modules.
  * This module is standalone-testable: it imports only the standard library
    plus PyYAML. ``torch`` is imported lazily so config validation works on a
    machine with no ML stack installed.
  * Precedence is: dataclass defaults < YAML file < CLI flags.

Typical use::

    from config.config import AppConfig, build_parser

    cfg = AppConfig.from_args(build_parser().parse_args())
    cfg.validate()
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is a hard requirement
    yaml = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

# 95% chi-square threshold for a 4-DOF measurement vector. This was a local
# constant in the original SORT/ByteTrack code, NOT a scipy export --
# ``scipy.stats.chi2inv95`` does not exist and raises ImportError.
# Verified equal to ``scipy.stats.chi2.ppf(0.95, 4) == 9.4877``.
CHI2INV95 = 9.4877

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The dev machine is an i5-1235U: 10 physical / 12 logical cores. Six threads
# leaves headroom for the OS and the video decode thread. Tune with --threads.
DEFAULT_THREADS = 6

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
URL_SCHEMES = ("rtsp://", "rtmp://", "http://", "https://", "tcp://")


def detect_device(requested: str = "auto") -> str:
    """Resolve a device string to something PyTorch understands.

    ``auto`` prefers CUDA and falls back to CPU. Any other value is returned
    unchanged so the caller keeps control. When torch is missing we cannot
    probe, so we warn and force CPU.
    """
    if requested and requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        LOGGER.warning("torch is not importable; assuming device='cpu'")
        return "cpu"
    return "0" if torch.cuda.is_available() else "cpu"


def resolve_threads(requested: Optional[int]) -> int:
    """Clamp the requested torch thread count into a sane range.

    Falls back to :data:`DEFAULT_THREADS` when unspecified. Oversubscribing
    hurts inference throughput, so the value is always clamped to
    ``[1, os.cpu_count()]``.
    """
    available = os.cpu_count() or 1
    count = DEFAULT_THREADS if requested is None else requested
    if count < 1:
        raise ValueError(f"num_threads must be >= 1, got {count}")
    if count > available:
        LOGGER.warning(
            "num_threads=%d exceeds %d logical CPUs; clamping to %d",
            count, available, available,
        )
        count = available
    return count


# --------------------------------------------------------------------------- #
# Sub-configs
# --------------------------------------------------------------------------- #
@dataclass
class DetectorConfig:
    """YOLO inference settings."""

    model: str = "yolov8n"
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 480
    half: bool = False
    classes: Optional[List[int]] = None
    max_det: int = 100

    def validate(self) -> None:
        if not 0.0 < self.conf < 1.0:
            raise ValueError(f"detector.conf must be in (0, 1), got {self.conf}")
        if not 0.0 < self.iou < 1.0:
            raise ValueError(f"detector.iou must be in (0, 1), got {self.iou}")
        if self.imgsz <= 0 or self.imgsz % 32 != 0:
            raise ValueError(
                f"detector.imgsz must be a positive multiple of 32, got {self.imgsz}"
            )
        if self.half:
            LOGGER.warning("detector.half=True is unreliable on CPU; forcing False")
            self.half = False
        if self.classes is not None:
            bad = [c for c in self.classes if not isinstance(c, int) or c < 0]
            if bad:
                raise ValueError(
                    f"detector.classes must be non-negative ints, got {bad}"
                )
            if len(self.classes) > 80:
                raise ValueError(
                    f"detector.classes has {len(self.classes)} ids; COCO has 80"
                )


@dataclass
class TrackerConfig:
    """ByteTrack association settings."""

    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer: int = 30
    match_thresh: float = 0.8
    frame_rate: int = 30

    def validate(self) -> None:
        if not 0.0 < self.track_low_thresh < self.track_high_thresh < 1.0:
            raise ValueError(
                "expected 0 < track_low_thresh < track_high_thresh < 1, got "
                f"low={self.track_low_thresh}, high={self.track_high_thresh}"
            )
        if not 0.0 < self.new_track_thresh < 1.0:
            raise ValueError(
                f"tracker.new_track_thresh must be in (0, 1), got {self.new_track_thresh}"
            )
        if not 0.0 < self.match_thresh <= 1.0:
            raise ValueError(
                f"tracker.match_thresh must be in (0, 1], got {self.match_thresh}"
            )
        if self.track_buffer < 1:
            raise ValueError(f"tracker.track_buffer must be >= 1, got {self.track_buffer}")
        if self.frame_rate < 1:
            raise ValueError(f"tracker.frame_rate must be >= 1, got {self.frame_rate}")


@dataclass
class VisualizationConfig:
    """Rendering settings for :mod:`utils.visualizer`."""

    show_fps: bool = True
    show_confidence: bool = True
    draw_trails: bool = False
    trail_length: int = 30
    thickness: int = 2
    font_scale: float = 0.5

    def validate(self) -> None:
        if self.thickness < 1:
            raise ValueError(f"visualization.thickness must be >= 1, got {self.thickness}")
        if self.font_scale <= 0:
            raise ValueError(
                f"visualization.font_scale must be > 0, got {self.font_scale}"
            )
        if self.trail_length < 1:
            raise ValueError(
                f"visualization.trail_length must be >= 1, got {self.trail_length}"
            )


@dataclass
class OutputConfig:
    """Where results are written."""

    save_video: bool = True
    save_csv: bool = True
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "output")
    codec: str = "mp4v"

    def validate(self) -> None:
        if not str(self.codec).strip():
            raise ValueError("output.codec must not be empty")
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(
                f"cannot create output dir {self.output_dir}: {exc}"
            ) from exc


@dataclass
class AppConfig:
    """Top-level configuration handed to every component."""

    source: str = "0"
    device: str = "auto"
    num_threads: int = field(default_factory=lambda: resolve_threads(None))
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    config_file: Optional[Path] = None

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # ---- derived properties ---------------------------------------------- #
    @property
    def resolved_device(self) -> str:
        return detect_device(self.device)

    @property
    def is_url(self) -> bool:
        return self.source.lower().startswith(URL_SCHEMES)

    @property
    def is_webcam(self) -> bool:
        return not self.is_url and self.source.isdigit()

    @property
    def is_image(self) -> bool:
        return (
            not self.is_url
            and not self.is_webcam
            and Path(self.source).suffix.lower() in IMAGE_EXTENSIONS
        )

    @property
    def source_kind(self) -> str:
        if self.is_webcam:
            return "webcam"
        if self.is_url:
            return "stream"
        if self.is_image:
            return "image"
        return "video"

    # ---- runtime side effects -------------------------------------------- #
    def apply_threads(self) -> int:
        """Call ``torch.set_num_threads()`` and return the effective count."""
        count = resolve_threads(self.num_threads)
        try:
            import torch
        except ImportError:
            LOGGER.warning("torch is not importable; skipping thread configuration")
            return count
        torch.set_num_threads(count)
        LOGGER.info(
            "torch threads set to %d of %d logical CPUs", count, os.cpu_count() or count
        )
        return count

    # ---- validation ------------------------------------------------------- #
    def validate(self) -> "AppConfig":
        """Validate every sub-config. Raises ``ValueError`` with a clear message."""
        if not str(self.source).strip():
            raise ValueError("source must not be empty")

        self.num_threads = resolve_threads(self.num_threads)

        if self.resolved_device == "cpu":
            LOGGER.info(
                "Running CPU-only. Lower detector.imgsz (e.g. 480) or pick a "
                "smaller model if throughput is poor."
            )

        if not self.is_url and not self.is_webcam and not Path(self.source).exists():
            suffix = Path(self.source).suffix.lower()
            known = sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)
            raise ValueError(
                f"source file not found: {self.source!r} "
                f"(detected kind={self.source_kind}, suffix={suffix!r}, "
                f"known extensions={known})"
            )

        self.detector.validate()
        self.tracker.validate()
        self.visualization.validate()
        self.output.validate()

        LOGGER.info(
            "Config OK: source=%s(%s) device=%s model=%s imgsz=%d threads=%d",
            self.source,
            self.source_kind,
            self.resolved_device,
            self.detector.model,
            self.detector.imgsz,
            self.num_threads,
        )
        return self

    # ---- constructors ----------------------------------------------------- #
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "AppConfig":
        """Build a config from parsed CLI args, applying YAML overrides first.

        Args whose value is ``None`` are treated as "not supplied on the CLI"
        so that the YAML file is not clobbered by argparse defaults. That is
        why every parser default below is ``None`` rather than a real value.
        """
        supplied = {k: v for k, v in vars(args).items() if v is not None}
        cfg = cls()

        config_path = supplied.pop("config", None)
        if config_path:
            path = Path(config_path)
            if not path.exists():
                raise FileNotFoundError(f"config file not found: {path}")
            cfg = cfg.merged_with_yaml(path)
            cfg.config_file = path

        _apply_cli(cfg, supplied)
        return cfg

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        """Build a config purely from a YAML file, with no CLI involved."""
        return cls().merged_with_yaml(Path(path)).validate()

    def merged_with_yaml(self, path: Path) -> "AppConfig":
        """Return a copy of this config with ``path`` layered on top.

        Unknown keys are a hard error: a typo in a config file should fail
        loudly rather than be silently ignored.
        """
        if yaml is None:
            raise RuntimeError("PyYAML is required for --config; pip install PyYAML")
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(
                f"{path}: top level must be a mapping, got {type(data).__name__}"
            )

        cfg = dataclasses.replace(self)
        sections = {
            "detector": cfg.detector,
            "tracker": cfg.tracker,
            "visualization": cfg.visualization,
            "output": cfg.output,
        }
        for key, value in data.items():
            if key in sections:
                _merge_into(sections[key], value, f"{path}:{key}")
            elif key == "config_file":
                raise ValueError(f"{path}: config_file cannot be set from YAML")
            else:
                _set_field(cfg, key, value, str(path))
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable snapshot, useful for logging the effective config."""
        raw = dataclasses.asdict(self)
        raw["output"]["output_dir"] = str(self.output.output_dir)
        raw["config_file"] = str(self.config_file) if self.config_file else None
        raw["log_file"] = str(self.log_file) if self.log_file else None
        return raw


# --------------------------------------------------------------------------- #
# Assignment helpers
# --------------------------------------------------------------------------- #
def _coerce(obj: Any, name: str, value: Any) -> Any:
    """Coerce ``value`` to the declared type of field ``name`` on ``obj``.

    ``from __future__ import annotations`` makes every annotation a string, so
    the hints below are matched exactly rather than by substring. That matters:
    a substring test for ``int`` would also catch ``Optional[List[int]]`` and
    then try ``int([0, 2])``.
    """
    hint = next(f.type for f in fields(obj) if f.name == name)
    hint = hint if isinstance(hint, str) else getattr(hint, "__name__", str(hint))
    if hint in ("Path", "Optional[Path]"):
        return None if value is None else Path(value)
    if hint == "bool":
        return bool(value)
    if hint == "float":
        return float(value)
    if hint == "int":
        return int(value)
    if hint == "str":
        return str(value)
    return value


def _set_field(obj: Any, name: str, value: Any, where: str) -> None:
    """Assign one field, coercing to its declared type. Raises on unknown names."""
    valid = {f.name for f in fields(obj)}
    if name not in valid:
        raise ValueError(f"{where}: unknown option {name!r} (valid: {sorted(valid)})")
    setattr(obj, name, _coerce(obj, name, value))


def _merge_into(obj: Any, data: Any, where: str) -> None:
    """Merge a YAML mapping into a sub-config dataclass instance."""
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(data).__name__}")
    for key, value in data.items():
        _set_field(obj, key, value, where)


# Explicit CLI destination -> (section name, field name).
# Deliberately a literal table: reflection over argparse Namespace would
# silently accept a typo'd destination and write it to the wrong sub-config.
_CLI_MAP: Dict[str, Tuple[str, str]] = {
    # detector
    "model":             ("detector", "model"),
    "confidence":        ("detector", "conf"),
    "iou":               ("detector", "iou"),
    "imgsz":             ("detector", "imgsz"),
    "classes":           ("detector", "classes"),
    # tracker
    "track_high_thresh": ("tracker", "track_high_thresh"),
    "track_low_thresh":  ("tracker", "track_low_thresh"),
    "new_track_thresh":  ("tracker", "new_track_thresh"),
    "track_buffer":      ("tracker", "track_buffer"),
    "match_thresh":      ("tracker", "match_thresh"),
    # visualization
    "draw_trails":       ("visualization", "draw_trails"),
    "show_fps":          ("visualization", "show_fps"),
    # output
    "output_dir":        ("output", "output_dir"),
    "save_video":        ("output", "save_video"),
    "save_csv":          ("output", "save_csv"),
}

_TOP_LEVEL = {"source", "device", "num_threads", "log_level", "log_file"}


def _apply_cli(cfg: AppConfig, supplied: Dict[str, Any]) -> None:
    """Layer CLI values onto ``cfg`` using the explicit table above."""
    sections = {
        "detector": cfg.detector,
        "tracker": cfg.tracker,
        "visualization": cfg.visualization,
        "output": cfg.output,
    }
    for key, value in supplied.items():
        if key in _CLI_MAP:
            section, attr = _CLI_MAP[key]
            _set_field(sections[section], attr, value, "cli")
        elif key in _TOP_LEVEL:
            _set_field(cfg, key, value, "cli")
        else:
            raise ValueError(
                f"cli: argument {key!r} has no mapping; add it to _CLI_MAP or _TOP_LEVEL"
            )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    All defaults are ``None`` so :meth:`AppConfig.from_args` can distinguish
    "user did not pass this" from "user passed the default value". That is why
    ``--save-video``/``--no-save-video`` are used for booleans instead of
    ``store_true``, which cannot express "unset".
    """
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Real-time object detection + ByteTrack tracking (CPU-friendly).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("input")
    g.add_argument("--config", type=Path, default=None, help="YAML config file")
    g.add_argument(
        "--source",
        default=None,
        help="0 = webcam, an image/video file path, or an rtsp:// stream URL",
    )

    g = p.add_argument_group("detector")
    g.add_argument("--model", default=None, help="YOLO weights, e.g. yolov8n / yolo26n")
    g.add_argument("--confidence", type=float, default=None, help="Detection threshold")
    g.add_argument("--iou", type=float, default=None, help="NMS IoU threshold")
    g.add_argument("--imgsz", type=int, default=None, help="Inference size (multiple of 32)")
    g.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="Restrict to these COCO class ids, e.g. --classes 0 2",
    )

    g = p.add_argument_group("tracker")
    g.add_argument("--track-high-thresh", type=float, default=None)
    g.add_argument("--track-low-thresh", type=float, default=None)
    g.add_argument("--new-track-thresh", type=float, default=None)
    g.add_argument("--track-buffer", type=int, default=None)
    g.add_argument("--match-thresh", type=float, default=None)

    g = p.add_argument_group("runtime")
    g.add_argument("--device", default=None, help="auto | cpu | 0")
    g.add_argument(
        "--threads",
        "--num-threads",
        dest="num_threads",
        type=int,
        default=None,
        help=f"torch.set_num_threads() value (default {DEFAULT_THREADS})",
    )
    g.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    g.add_argument("--log-file", type=Path, default=None)

    g = p.add_argument_group("output")
    g.add_argument("--output-dir", type=Path, default=None)
    g.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=None)
    g.add_argument("--save-csv", action=argparse.BooleanOptionalAction, default=None)
    g.add_argument(
        "--draw-trails", action=argparse.BooleanOptionalAction, default=None
    )
    g.add_argument(
        "--no-fps",
        dest="show_fps",
        action="store_false",
        default=None,
        help="Hide the FPS/resolution HUD",
    )
    return p


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Configure root logging once, with timestamp + level prefixes."""
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
        except OSError as exc:
            LOGGER.warning("cannot open log file %s: %s", log_file, exc)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point so ``python config/config.py --help`` works standalone."""
    args = build_parser().parse_args(argv)
    cfg = AppConfig.from_args(args)
    setup_logging(cfg.log_level, cfg.log_file)
    try:
        cfg.validate()
    except (ValueError, FileNotFoundError) as exc:
        LOGGER.error("invalid configuration: %s", exc)
        return 2
    LOGGER.debug("effective config: %s", cfg.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())