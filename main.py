"""Command-line entry point: wire detector, tracker and visualizer together.

Examples::

    python main.py --source 0                      # webcam
    python main.py --source clip.mp4               # video file
    python main.py --source photo.jpg              # single image
    python main.py --source rtsp://cam/stream      # IP camera
    python main.py --config example_config.yaml    # YAML, CLI wins on conflict
    python main.py --source clip.mp4 --classes person car

The per-frame loop is deliberately linear and readable: read, detect, track,
draw, write, show. Configuration comes entirely from :mod:`config`, so no magic
numbers live here.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import logging
import pstats
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config import AppConfig, build_parser, setup_logging
from utils.detector import YOLODetector
from utils.stats import RunStats
from utils.tracker import BYTETracker
from utils.visualizer import TracksCsvWriter, Visualizer, open_writer, release_writer

LOGGER = logging.getLogger("main")

# Used when a container reports a nonsensical frame rate (common with webcams
# and some MP4s that carry no timing metadata).
FALLBACK_FPS = 25.0

# Fail this many consecutive empty reads before declaring the source finished.
MAX_CONSECUTIVE_EMPTY_READS = 10


@dataclass
class FrameStats:
    """Rolling per-stage timings, used for the end-of-run summary."""

    total_frames: int = 0
    detect_ms: List[float] = field(default_factory=list)
    track_ms: List[float] = field(default_factory=list)
    draw_ms: List[float] = field(default_factory=list)
    unique_ids: set = field(default_factory=set)

    def record(self, detect_ms: float, track_ms: float, draw_ms: float, tracks: Sequence) -> None:
        """Accumulate one frame's stage timings and register its track IDs."""
        self.total_frames += 1
        self.detect_ms.append(detect_ms)
        self.track_ms.append(track_ms)
        self.draw_ms.append(draw_ms)
        for track in tracks:
            self.unique_ids.add(track.track_id)

    @staticmethod
    def _mean(values: List[float]) -> float:
        """Arithmetic mean of ``values``; 0.0 for an empty sequence."""
        return sum(values) / len(values) if values else 0.0

    @property
    def avg_fps(self) -> float:
        """Throughput implied by recorded stage timings (frames per second)."""
        total = sum(self.detect_ms) + sum(self.track_ms) + sum(self.draw_ms)
        if total <= 0 or self.total_frames == 0:
            return 0.0
        return self.total_frames / (total / 1000.0)

    @property
    def mean_detect_ms(self) -> float:
        """Mean detection time per frame, in milliseconds."""
        return self._mean(self.detect_ms)

    @property
    def mean_track_ms(self) -> float:
        """Mean tracking time per frame, in milliseconds."""
        return self._mean(self.track_ms)

    @property
    def mean_draw_ms(self) -> float:
        """Mean drawing time per frame, in milliseconds."""
        return self._mean(self.draw_ms)

    def summary(self) -> str:
        """Human-readable breakdown of where the frame budget went."""
        detect = self.mean_detect_ms
        track = self.mean_track_ms
        draw = self.mean_draw_ms
        total = detect + track + draw

        def pct(value: float) -> str:
            """Stage time as a share of the total budget."""
            return f"{100.0 * value / total:5.1f}%" if total else "  n/a"

        return (
            f"frames processed : {self.total_frames}\n"
            f"unique track IDs : {len(self.unique_ids)}\n"
            f"average FPS      : {self.avg_fps:.2f}\n"
            f"detect  {detect:7.2f} ms  ({pct(detect)})\n"
            f"track   {track:7.2f} ms  ({pct(track)})\n"
            f"draw    {draw:7.2f} ms  ({pct(draw)})"
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _extract_class_tokens(argv: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Pull ``--classes`` values out of ``argv`` before argparse sees them.

    ``config.build_parser`` declares ``--classes`` as ``type=int``, which cannot
    express ``--classes person car``. Removing the option up front lets this
    module accept both class ids and COCO class names, resolving names to ids
    once the model has loaded. Returns ``(clean_argv, class_tokens)``.
    """
    cleaned: List[str] = []
    tokens: List[str] = []
    i = 0
    argv = list(argv)
    while i < len(argv):
        token = argv[i]
        if token == "--classes":
            i += 1
            while i < len(argv) and not argv[i].startswith("-"):
                tokens.append(argv[i])
                i += 1
            continue
        if token.startswith("--classes="):
            tokens.append(token.split("=", 1)[1])
            i += 1
            continue
        cleaned.append(token)
        i += 1
    return cleaned, tokens


def build_cli(argv: Sequence[str]) -> Tuple[argparse.ArgumentParser, List[str]]:
    """Extend the shared config parser with presentation-only flags.

    Args:
        argv: Raw command-line arguments.

    Returns:
        ``(parser, class_tokens)``, where ``class_tokens`` holds the raw
        ``--classes`` values, since names need a loaded model to resolve.
    """
    cleaned, class_tokens = _extract_class_tokens(argv)
    parser = build_parser()
    parser.add_argument(
        "--output",
        dest="output_file",
        default=None,
        help="Output video file (or directory). Overrides --output-dir.",
    )
    parser.add_argument(
        "--no-video",
        dest="save_video",
        action="store_false",
        default=None,
        help="Skip saving the annotated video.",
    )
    parser.add_argument(
        "--no-csv",
        dest="save_csv",
        action="store_false",
        default=None,
        help="Skip writing the tracking CSV.",
    )
    parser.add_argument(
        "--no-display",
        dest="display",
        action="store_false",
        default=None,
        help="Do not open a preview window (required on headless machines).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Write a cProfile report to output/profile.pstats after the run.",
    )
    return parser, class_tokens


# --------------------------------------------------------------------------- #
# Source handling
# --------------------------------------------------------------------------- #
def _sanitised_fps(raw: float) -> float:
    """Coerce a container-reported frame rate into something usable."""
    if raw is None or raw != raw or raw <= 0:  # NaN check is `raw != raw`
        LOGGER.warning("source reported an invalid FPS (%s); using %.1f", raw, FALLBACK_FPS)
        return FALLBACK_FPS
    return float(raw)


def open_source(cfg: AppConfig) -> Tuple[cv2.VideoCapture, float, Tuple[int, int]]:
    """Open the configured source.

    Handles all four kinds: a webcam index, an ``rtsp://``/``http://`` stream, a
    video file, or a single image. ``cv2.VideoCapture`` covers all of them, so
    the only real work is picking the right constructor argument and guarding
    the frame rate.

    Returns:
        ``(capture, fps, (width, height))``.

    Raises:
        FileNotFoundError: the source does not exist.
        RuntimeError: OpenCV could not open it.
    """
    source = cfg.source
    if cfg.is_webcam:
        capture = cv2.VideoCapture(int(source))
        LOGGER.info("opening webcam index %s", source)
    else:
        path = Path(source)
        if not cfg.is_url and not path.is_file():
            raise FileNotFoundError(f"source file not found: {source}")
        capture = cv2.VideoCapture(source)
        LOGGER.info("opening %s source: %s", cfg.source_kind, source)

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"could not open source {source!r} as a {cfg.source_kind}. "
            "Check the path, the codec, or whether another app is holding the "
            "webcam."
        )

    fps = _sanitised_fps(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"source {source!r} reported an invalid size {width}x{height}")
    LOGGER.info("source ready: %dx%d @ %.2f fps", width, height, fps)
    return capture, fps, (width, height)


def _resolve_output_paths(
    cfg: AppConfig, output_file: Optional[str], source_kind: str
) -> Tuple[Optional[Path], Path]:
    """Decide the video and CSV destinations.

    Returns ``(video_path_or_None, csv_path)``. For an image source the video
    path is ``None`` because a single frame cannot be muxed into a usable MP4;
    those runs write a PNG instead.
    """
    csv_path = cfg.output.output_dir / "tracking.csv"
    if not cfg.output.save_video:
        return None, csv_path

    if output_file:
        candidate = Path(output_file)
        if candidate.suffix:
            cfg.output.output_dir = candidate.parent
            return candidate, cfg.output.output_dir / "tracking.csv"
        cfg.output.output_dir = candidate
        video_path = candidate / "result.mp4"
    else:
        stem = "result"
        if source_kind in ("video", "image") and Path(cfg.source).stem:
            stem = f"{Path(cfg.source).stem}_tracked"
        video_path = cfg.output.output_dir / f"{stem}.mp4"
    csv_path = cfg.output.output_dir / "tracking.csv"
    return video_path, csv_path


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def _finish_run(
    stats: FrameStats,
    run_stats: RunStats,
    profiler: Optional[cProfile.Profile],
    cfg: AppConfig,
    outputs: dict,
    wall_seconds: float,
) -> None:
    """Write the optional cProfile report and the run-statistics JSON.

    Called on both clean completion and Ctrl-C so an interrupted run still
    leaves useful artefacts behind. Individual write failures are logged, not
    raised: statistics must never turn a successful run into a failure.
    """
    if profiler is not None:
        try:
            profiler.disable()
            pstats_path = cfg.output.output_dir / "profile.pstats"
            profiler.dump_stats(str(pstats_path))
            stream = io.StringIO()
            stats_report = pstats.Stats(profiler, stream=stream)
            stats_report.strip_dirs().sort_stats("cumulative").print_stats(15)
            LOGGER.info(
                "profile written to %s (top by cumulative time)\n%s",
                pstats_path,
                stream.getvalue(),
            )
        except Exception:  # pragma: no cover - defensive
            LOGGER.exception("could not write profile")

    if cfg.output.save_stats:
        try:
            run_stats.set_summary(
                wall_seconds=wall_seconds,
                avg_fps=stats.avg_fps,
                mean_detect_ms=stats.mean_detect_ms,
                mean_track_ms=stats.mean_track_ms,
                mean_draw_ms=stats.mean_draw_ms,
                outputs=outputs,
            )
            run_stats.write(cfg.output.output_dir / "run_stats.json")
        except Exception:  # pragma: no cover - defensive
            LOGGER.exception("could not write run statistics")


@dataclass
class _Pipeline:
    """Opened source, loaded model and output writers for one run.

    Built by :func:`_prepare` and owned by :func:`run`, which releases
    everything in its ``finally`` block.
    """

    capture: cv2.VideoCapture
    source_fps: float
    frame_size: Tuple[int, int]
    first_frame: np.ndarray
    detector: YOLODetector
    tracker: BYTETracker
    visualizer: Visualizer
    writer: Optional[cv2.VideoWriter]
    csv_out: Optional[TracksCsvWriter]
    video_path: Optional[Path]
    csv_path: Optional[Path]
    png_path: Optional[Path] = None


def _prepare(
    cfg: AppConfig,
    output_file: Optional[str],
    class_tokens: Optional[Sequence[str]],
) -> _Pipeline:
    """Open the source and model, warm up, and open the output writers.

    The capture is released again if any later step fails, so a partial
    setup never leaks the webcam or file handle.

    Raises:
        FileNotFoundError: the source file does not exist.
        RuntimeError: the source cannot be opened or has no readable frames.
        ValueError: ``--classes`` names cannot be resolved against the model.
    """
    capture, source_fps, frame_size = open_source(cfg)
    try:
        video_path, csv_path = _resolve_output_paths(cfg, output_file, cfg.source_kind)
        detector = YOLODetector(cfg.detector, device=cfg.resolved_device)

        # Class names are resolved here, against the loaded model, rather than
        # in main(): doing it earlier would need a second model instance.
        if class_tokens:
            try:
                cfg.detector.classes = detector.resolve_class_names(class_tokens)
            except ValueError as exc:
                raise ValueError(f"cannot resolve --classes: {exc}") from exc
            LOGGER.info("--classes %s resolved to ids %s", list(class_tokens), cfg.detector.classes)
        if cfg.detector.classes:
            LOGGER.info("restricting to class ids %s", cfg.detector.classes)
        tracker = BYTETracker(cfg.tracker)
        visualizer = Visualizer(cfg.visualization)

        # Read one frame to learn the true size, then warm up the model so the
        # first measured frame is not paying for lazy initialisation.
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"could not read any frames from {cfg.source!r}")
        actual_size = (frame.shape[1], frame.shape[0])
        if actual_size != frame_size:
            LOGGER.info(
                "container size %s differs from actual %s; using actual",
                frame_size,
                actual_size,
            )
            frame_size = actual_size
        detector.warmup(frame.shape)

        writer: Optional[cv2.VideoWriter] = None
        if video_path is not None and cfg.source_kind != "image":
            writer = open_writer(video_path, source_fps, frame_size)
        elif video_path is not None:
            LOGGER.info("image source: saving a PNG instead of a video")
        csv_out = TracksCsvWriter(csv_path) if cfg.output.save_csv else None

        return _Pipeline(
            capture=capture,
            source_fps=source_fps,
            frame_size=frame_size,
            first_frame=frame,
            detector=detector,
            tracker=tracker,
            visualizer=visualizer,
            writer=writer,
            csv_out=csv_out,
            video_path=video_path,
            csv_path=csv_path,
        )
    except BaseException:
        capture.release()
        raise


def _step(
    cfg: AppConfig,
    frame: np.ndarray,
    frame_id: int,
    pipe: _Pipeline,
    stats: FrameStats,
    run_stats: RunStats,
    display: bool,
) -> Tuple[bool, Optional[Path]]:
    """Run detect → track → draw → write → preview for a single frame.

    Returns:
        ``(quit_requested, png_path)``: whether the user pressed ``q``, and
        (for image sources) the PNG that was written.
    """
    t0 = time.perf_counter()
    detections = pipe.detector.detect(frame)
    t1 = time.perf_counter()
    tracks = pipe.tracker.update(detections, frame_id=frame_id)
    t2 = time.perf_counter()
    annotated = pipe.visualizer.draw(frame, tracks, fps=stats.avg_fps)
    t3 = time.perf_counter()

    stats.record((t1 - t0) * 1000.0, (t2 - t1) * 1000.0, (t3 - t2) * 1000.0, tracks)
    run_stats.record(detections, tracks)

    png_path: Optional[Path] = None
    if pipe.csv_out is not None:
        pipe.csv_out.write(frame_id, tracks)
    if pipe.writer is not None:
        pipe.writer.write(annotated)
    elif cfg.source_kind == "image":
        png_path = cfg.output.output_dir / "result.png"
        cv2.imwrite(str(png_path), annotated)
        LOGGER.info("image result written to %s", png_path)

    quit_requested = False
    if display:
        cv2.imshow("object_detection_tracking", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            LOGGER.info("quit requested via 'q'")
            quit_requested = True

    if (frame_id + 1) % 100 == 0:
        LOGGER.info(
            "frame %d | tracks=%d | avg fps=%.1f | detect=%.0fms",
            frame_id + 1,
            len(tracks),
            stats.avg_fps,
            pipe.detector.avg_detect_ms,
        )
    return quit_requested, png_path


def _frame_loop(
    cfg: AppConfig,
    pipe: _Pipeline,
    stats: FrameStats,
    run_stats: RunStats,
    display: bool,
) -> None:
    """Read, process and write frames until the source ends or the user quits.

    Stops early on ``q`` in the preview window, and after a single frame
    for image sources. Ends after :data:`MAX_CONSECUTIVE_EMPTY_READS`
    consecutive failed reads so truncated videos terminate cleanly.
    """
    frame = pipe.first_frame
    frame_id = 0
    empty_reads = 0
    first_frame = True

    while True:
        if first_frame:
            # Reuse the frame read during warm-up rather than decoding twice.
            first_frame = False
        else:
            ok, frame = pipe.capture.read()
            if not ok or frame is None:
                empty_reads += 1
                if empty_reads > MAX_CONSECUTIVE_EMPTY_READS:
                    LOGGER.info("source ended after %d frames", frame_id)
                    break
                continue
            empty_reads = 0

        quit_requested, png_path = _step(cfg, frame, frame_id, pipe, stats, run_stats, display)
        if png_path is not None:
            pipe.png_path = png_path
        if quit_requested:
            break

        frame_id += 1
        if cfg.source_kind == "image":
            LOGGER.info("image source: single frame processed")
            break


def run(
    cfg: AppConfig,
    output_file: Optional[str] = None,
    display: bool = True,
    class_tokens: Optional[Sequence[str]] = None,
    profile: bool = False,
) -> int:
    """Execute the pipeline and return a process exit code.

    Args:
        cfg: Validated configuration.
        output_file: Optional ``--output`` override.
        display: Open a preview window.
        class_tokens: Raw ``--classes`` values. Resolved against the loaded
            model's class names, so the model is loaded exactly once.
        profile: Collect cProfile data and write ``output/profile.pstats``.

    Returns:
        ``0`` on a clean run, ``1`` on error, ``130`` on interrupt.
    """
    started = time.perf_counter()
    stats = FrameStats()
    run_stats = RunStats(
        source=cfg.source,
        source_kind=cfg.source_kind,
        device=cfg.resolved_device,
        model=cfg.detector.model,
        imgsz=cfg.detector.imgsz,
    )
    profiler = cProfile.Profile() if profile else None
    pipe: Optional[_Pipeline] = None

    def _outputs() -> dict:
        """Artefact paths to record in the statistics report."""
        writer = pipe.writer if pipe is not None else None
        csv_out = pipe.csv_out if pipe is not None else None
        png_path = pipe.png_path if pipe is not None else None
        return {
            "video": pipe.video_path if writer is not None else None,
            "csv": pipe.csv_path if csv_out is not None else None,
            "png": png_path,
            "stats": cfg.output.output_dir / "run_stats.json" if cfg.output.save_stats else None,
            "profile": cfg.output.output_dir / "profile.pstats" if profile else None,
        }

    try:
        pipe = _prepare(cfg, output_file, class_tokens)
        LOGGER.info(
            "starting loop: %s source, device=%s, imgsz=%d",
            cfg.source_kind,
            cfg.resolved_device,
            cfg.detector.imgsz,
        )

        if profiler is not None:
            profiler.enable()
        _frame_loop(cfg, pipe, stats, run_stats, display)
        if profiler is not None:
            profiler.disable()

        _finish_run(
            stats,
            run_stats,
            profiler,
            cfg,
            _outputs(),
            time.perf_counter() - started,
        )
        LOGGER.info("run complete\n%s", stats.summary())
        return 0

    except KeyboardInterrupt:
        LOGGER.warning("interrupted by user; shutting down cleanly")
        _finish_run(
            stats,
            run_stats,
            profiler,
            cfg,
            _outputs(),
            time.perf_counter() - started,
        )
        LOGGER.info("partial results\n%s", stats.summary())
        return 130
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        LOGGER.error("fatal: %s", exc)
        return 1
    finally:
        if pipe is not None:
            release_writer(pipe.writer)
            if pipe.csv_out is not None:
                pipe.csv_out.close()
            pipe.capture.release()
        if display:
            try:
                cv2.destroyAllWindows()
            except cv2.error:  # pragma: no cover
                pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse arguments, validate, and run. Returns a process exit code."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser, class_tokens = build_cli(raw_argv)
    args = parser.parse_args()

    # Presentation-only flags belong to main, not to the AppConfig dataclasses.
    # They are captured here and removed before building the config, because
    # config's explicit CLI mapping raises on anything it does not recognise
    # (by design: a typo should fail loudly, not be silently dropped).
    output_file = getattr(args, "output_file", None)
    display_flag = getattr(args, "display", None)
    profile_flag = bool(getattr(args, "profile", False))
    for key in ("output_file", "display", "classes", "profile"):
        if hasattr(args, key):
            delattr(args, key)

    cfg = AppConfig.from_args(args)
    setup_logging(cfg.log_level, cfg.log_file)

    display = display_flag is not False
    if display_flag is False:
        LOGGER.info("preview window disabled")

    try:
        cfg.apply_threads()
        cfg.validate()
    except (ValueError, FileNotFoundError) as exc:
        LOGGER.error("invalid configuration: %s", exc)
        return 2

    return run(
        cfg,
        output_file=output_file,
        display=display,
        class_tokens=class_tokens or None,
        profile=profile_flag,
    )


if __name__ == "__main__":
    raise SystemExit(main())
