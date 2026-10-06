"""Frame annotation and output writing.

Three responsibilities, kept in one module because they share OpenCV state:

* :class:`Visualizer` draws boxes, labels, trails and an HUD.
* :func:`open_writer` produces a ``VideoWriter`` with a codec fallback chain.
* :class:`TracksCsvWriter` appends one row per track per frame.

Per-track colours are derived deterministically from the track id, so the same
object keeps the same colour for the entire video without storing any state.
"""

from __future__ import annotations

import colorsys
import csv
import logging
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Deque, Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from config.config import VisualizationConfig

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, erased at runtime
    from utils.tracker import Track

LOGGER = logging.getLogger(__name__)

# Tried in order when opening a writer. mp4v is the safest on Windows with the
# bundled FFmpeg; avc1 (H.264) is preferred when available but is often missing
# on a stock opencv-python build; MJPG in an .avi is the universal fallback.
CODEC_FALLBACKS: Tuple[str, ...] = ("mp4v", "avc1", "MJPG")

CSV_COLUMNS = (
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
)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def open_writer(
    output_path: Path,
    fps: float,
    frame_size: Tuple[int, int],
    codecs: Sequence[str] = CODEC_FALLBACKS,
) -> Optional[cv2.VideoWriter]:
    """Open a ``VideoWriter``, trying each codec until one initialises.

    Args:
        output_path: Destination file. The extension is respected; if a codec
            forces a container change (MJPG needs .avi) the suffix is adjusted.
        fps: Output frame rate. Clamped to ``(0, 240]`` because some containers
            reject 0 or absurd values.
        frame_size: ``(width, height)``.
        codecs: Codec candidates, in preference order.

    Returns:
        An open writer, or ``None`` if every candidate failed. Returning
        ``None`` rather than raising lets the pipeline continue in preview-only
        mode instead of dying when a codec is missing.

    Note:
        The fallback chain only triggers when ``isOpened()`` reports False.
        Some OpenCV/FFmpeg builds accept an unknown FourCC tag with nothing
        more than a console warning and still report success, producing a file
        that may not play everywhere. There is no portable way to pre-validate
        a tag, so ``mp4v`` is listed first precisely because it is the most
        widely decodable.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rate = float(fps) if fps and fps == fps and fps > 0 else 25.0
    rate = min(max(rate, 1.0), 240.0)
    width, height = int(frame_size[0]), int(frame_size[1])

    for codec in codecs:
        target = output_path
        if codec == "MJPG" and target.suffix.lower() != ".avi":
            target = target.with_suffix(".avi")
        try:
            writer = cv2.VideoWriter(
                str(target), cv2.VideoWriter_fourcc(*codec), rate, (width, height)
            )
        except cv2.error as exc:  # pragma: no cover
            LOGGER.warning("codec %s raised on open: %s", codec, exc)
            continue
        if writer.isOpened():
            if target != output_path:
                LOGGER.info("codec %s requires a .avi container -> %s", codec, target.name)
            LOGGER.info(
                "video writer open: %s (%s @ %.2f fps, %dx%d)", target, codec, rate, width, height
            )
            return writer
        writer.release()
        LOGGER.warning("codec %s unavailable for %s; trying next", codec, target.name)

    LOGGER.error("no usable codec from %s; video output disabled", list(codecs))
    return None


def release_writer(writer: Optional[cv2.VideoWriter]) -> None:
    """Release a writer, tolerating ``None`` and already-released handles."""
    if writer is None:
        return
    try:
        writer.release()
    except cv2.error as exc:  # pragma: no cover
        LOGGER.warning("error releasing writer: %s", exc)


class TracksCsvWriter:
    """Writes one CSV row per confirmed track per frame.

    Opens the file lazily on first write so that a run producing no tracks
    still leaves a header-only file rather than nothing at all.

    Example::

        with TracksCsvWriter(Path("output/tracking.csv")) as csv_out:
            for frame_id, tracks in enumerate(all_tracks):
                csv_out.write(frame_id, tracks)
    """

    def __init__(self, path: Path) -> None:
        """Store the destination; the file is created lazily by :meth:`open`."""
        self.path = Path(path)
        self._handle = None
        self._writer: Optional[csv.writer] = None
        self.row_count = 0

    def open(self) -> None:
        """Create the file and write the header row. Idempotent."""
        if self._handle is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle)
        self._writer.writerow(CSV_COLUMNS)
        LOGGER.info("csv writer open: %s", self.path)

    def write(self, frame_id: int, tracks: Iterable) -> int:
        """Append one row per track. Returns the number of rows written.

        Args:
            frame_id: Zero-based frame counter.
            tracks: Anything iterable yielding objects with ``track_id``,
                ``last_class_id``, ``last_class_name``, ``last_score``,
                ``tlbr`` and ``age``.
        """
        self.open()
        assert self._writer is not None
        count = 0
        for track in tracks:
            x1, y1, x2, y2 = track.tlbr
            self._writer.writerow(
                [
                    int(frame_id),
                    int(track.track_id),
                    int(track.last_class_id),
                    track.last_class_name,
                    round(float(track.last_score), 4),
                    round(float(x1), 2),
                    round(float(y1), 2),
                    round(float(x2), 2),
                    round(float(y2), 2),
                    int(track.age),
                ]
            )
            count += 1
        self.row_count += count
        return count

    def close(self) -> None:
        """Flush and close. Safe to call more than once."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._writer = None
            LOGGER.info("csv writer closed: %s (%d rows)", self.path, self.row_count)

    def __enter__(self) -> "TracksCsvWriter":
        """Enter the context manager; the file is opened on first write."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the CSV file when leaving the ``with`` block."""
        self.close()


class Visualizer:
    """Draws detections, tracks and a HUD onto frames.

    Args:
        config: Rendering settings from
            :class:`~config.config.VisualizationConfig`.
    """

    def __init__(self, config: Optional[VisualizationConfig] = None) -> None:
        """Validate rendering settings and clear trail/colour caches (see class docstring)."""
        self.config = config or VisualizationConfig()
        self.config.validate()
        self._trails: Dict[int, Deque[Tuple[int, int]]] = {}
        self._color_cache: Dict[int, Tuple[int, int, int]] = {}

    # -- colours ------------------------------------------------------------ #
    @staticmethod
    def track_color(track_id: int) -> Tuple[int, int, int]:
        """A stable BGR colour for a track id.

        The hue is derived from the id via the golden-ratio conjugate, so
        consecutive ids land far apart on the colour wheel and ids remain
        visually distinct for hundreds of objects rather than cycling through a
        short palette. Gold saturation and value keep every colour legible
        against both bright and dark footage.
        """
        hue = (int(track_id) * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
        return (int(b * 255), int(g * 255), int(r * 255))

    def color_for(self, track_id: int) -> Tuple[int, int, int]:
        """Cached :meth:`track_color`, so hues are computed once per track."""
        if track_id not in self._color_cache:
            self._color_cache[track_id] = self.track_color(track_id)
        return self._color_cache[track_id]

    # -- trails -------------------------------------------------------------- #
    def _push_trail(self, track_id: int, center: Tuple[int, int]) -> None:
        """Append ``center`` to the track's ring buffer, creating it if new."""
        trail = self._trails.setdefault(track_id, deque(maxlen=self.config.trail_length))
        trail.append(center)

    def prune_trails(self, live_ids: Iterable[int]) -> None:
        """Drop trail history for tracks that no longer exist."""
        live = set(live_ids)
        for track_id in [k for k in self._trails if k not in live]:
            del self._trails[track_id]

    def reset(self) -> None:
        """Forget all trail history. Call between videos."""
        self._trails.clear()
        self._color_cache.clear()

    def _draw_trail(self, frame: np.ndarray, track_id: int, color: Tuple[int, int, int]) -> None:
        """Draw a fading polyline through a track's recent centers."""
        points = list(self._trails.get(track_id, ()))
        if len(points) < 2:
            return
        overlay = frame.copy()
        for i in range(len(points) - 1):
            start, end = points[i], points[i + 1]
            # Older segments are dimmed by blending toward black, so the head
            # of the trail is brightest without needing per-segment compositing.
            fade = 0.25 + 0.75 * (i / (len(points) - 1))
            shade = tuple(int(c * fade) for c in color)
            cv2.line(overlay, start, end, shade, self.config.thickness, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # -- drawing ------------------------------------------------------------- #
    def _draw_label(self, frame: np.ndarray, track: "Track", color: Tuple[int, int, int]) -> None:
        """Draw a filled caption box above the bounding box."""
        parts = [f"ID:{track.track_id}", track.last_class_name]
        if self.config.show_confidence:
            parts.append(f"{track.last_score:.2f}")
        text = " ".join(parts)

        x1, y1, x2, _ = track.tlbr
        thickness = self.config.thickness
        (tw, th), baseline = cv2.getTextSize(text, FONT, self.config.font_scale, thickness)
        x1i, y1i = int(x1), int(y1)
        # Place the label above the box, or inside it when there is no room.
        label_top = max(y1i - th - baseline - 4, 0)
        label_bottom = label_top + th + baseline + 4
        cv2.rectangle(frame, (x1i, label_top), (x1i + tw + 8, label_bottom), color, -1)
        cv2.putText(
            frame,
            text,
            (x1i + 4, label_bottom - baseline - 1),
            FONT,
            self.config.font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    def _draw_hud(self, frame: np.ndarray, fps: float, track_count: int) -> None:
        """Draw FPS, resolution and live-track count in the top-left."""
        if not self.config.show_fps:
            return
        height, width = frame.shape[:2]
        lines = [
            f"FPS: {fps:5.1f}",
            f"RES: {width}x{height}",
            f"TRACKS: {track_count}",
        ]
        scale = self.config.font_scale
        sizes = [cv2.getTextSize(line, FONT, scale, 1)[0] for line in lines]
        line_h = max(h for (_, h) in sizes) + 6
        box_w = max(w for (w, _) in sizes) + 16

        cv2.rectangle(frame, (0, 0), (box_w, line_h * len(lines)), (0, 0, 0), -1)
        for i, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (8, line_h * i + line_h - 8),
                FONT,
                scale,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def draw(self, frame: np.ndarray, tracks: Sequence, fps: float = 0.0) -> np.ndarray:
        """Annotate one frame and return it.

        The input is not modified in place when trails are disabled, which
        keeps the caller's frame reusable; with trails enabled an internal
        overlay copy is composited back onto the returned frame.

        Args:
            frame: ``HxWx3`` uint8 BGR frame.
            tracks: Iterable of :class:`~utils.tracker.Track`.
            fps: Current throughput, shown in the HUD.

        Returns:
            The annotated frame.
        """
        if frame is None:
            raise ValueError("draw() received a None frame")

        canvas = frame.copy() if self.config.draw_trails else frame
        thickness = self.config.thickness

        for track in tracks:
            color = self.color_for(track.track_id)
            x1, y1, x2, y2 = (int(v) for v in track.tlbr)

            if self.config.draw_trails:
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                self._push_trail(track.track_id, center)
                self._draw_trail(canvas, track.track_id, color)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
            self._draw_label(canvas, track, color)

        if self.config.draw_trails:
            self.prune_trails(t.track_id for t in tracks)

        self._draw_hud(canvas, fps, len(tracks))
        return canvas
