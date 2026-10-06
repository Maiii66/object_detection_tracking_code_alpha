"""Run statistics: per-class usage counters exported as JSON.

:class:`RunStats` is deliberately independent of the timing summary in
``main.FrameStats``: timings describe *speed*, this describes *what was
seen*. Both are merged into ``output/run_stats.json`` at the end of a run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Set

LOGGER = logging.getLogger(__name__)


def _utc_now() -> str:
    """ISO-8601 UTC timestamp with second precision."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class RunStats:
    """Accumulates per-class usage counters for one pipeline run.

    Args:
        source: Raw ``--source`` value (path, index or URL).
        source_kind: ``webcam`` / ``stream`` / ``image`` / ``video``.
        device: Resolved device string, e.g. ``cpu``.
        model: Detector model name, e.g. ``yolov8n``.
        imgsz: Inference size used for the run.

    ``record`` is called once per frame with the raw detections and the
    confirmed tracks; ``set_summary`` folds in the timing numbers computed
    by ``main.FrameStats``; ``write`` emits the JSON report.
    """

    source: str
    source_kind: str
    device: str
    model: str
    imgsz: int

    started_at: str = field(default_factory=_utc_now)
    finished_at: str = ""
    wall_seconds: float = 0.0

    frames: int = 0
    detections_total: int = 0

    avg_fps: float = 0.0
    mean_detect_ms: float = 0.0
    mean_track_ms: float = 0.0
    mean_draw_ms: float = 0.0

    outputs: Dict[str, str] = field(default_factory=dict)

    # Internal per-class accumulators: raw detection counts and the set of
    # unique track ids seen for each class name.
    _class_detections: Dict[str, int] = field(default_factory=dict, repr=False)
    _class_track_ids: Dict[str, Set[int]] = field(default_factory=dict, repr=False)

    # ---- accumulation ---------------------------------------------------- #
    def record(
        self,
        detections: Iterable,
        tracks: Iterable,
    ) -> None:
        """Fold one frame's detections and tracks into the counters.

        Args:
            detections: Objects exposing ``class_name``.
            tracks: Objects exposing ``track_id`` and ``last_class_name``.
        """
        self.frames += 1
        for det in detections:
            self.detections_total += 1
            name = getattr(det, "class_name", "unknown")
            self._class_detections[name] = self._class_detections.get(name, 0) + 1
        for track in tracks:
            name = getattr(track, "last_class_name", "unknown")
            self._class_track_ids.setdefault(name, set()).add(int(track.track_id))

    def set_summary(
        self,
        *,
        wall_seconds: float,
        avg_fps: float,
        mean_detect_ms: float,
        mean_track_ms: float,
        mean_draw_ms: float,
        outputs: Dict[str, Any],
    ) -> None:
        """Attach timing results and output paths; marks the run finished."""
        self.finished_at = _utc_now()
        self.wall_seconds = round(wall_seconds, 3)
        self.avg_fps = round(avg_fps, 2)
        self.mean_detect_ms = round(mean_detect_ms, 3)
        self.mean_track_ms = round(mean_track_ms, 3)
        self.mean_draw_ms = round(mean_draw_ms, 3)
        self.outputs = {k: str(v) for k, v in outputs.items() if v is not None}

    # ---- export ---------------------------------------------------------- #
    @property
    def unique_track_ids(self) -> int:
        """Number of distinct track ids seen across all frames."""
        return len(set().union(*self._class_track_ids.values())) if self._class_track_ids else 0

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready snapshot of the run."""
        return {
            "source": self.source,
            "source_kind": self.source_kind,
            "device": self.device,
            "model": self.model,
            "imgsz": self.imgsz,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "wall_seconds": self.wall_seconds,
            "frames": self.frames,
            "detections_total": self.detections_total,
            "unique_track_ids": self.unique_track_ids,
            "avg_fps": self.avg_fps,
            "mean_detect_ms": self.mean_detect_ms,
            "mean_track_ms": self.mean_track_ms,
            "mean_draw_ms": self.mean_draw_ms,
            "per_class": {
                name: {
                    "detections": self._class_detections.get(name, 0),
                    "unique_track_ids": len(self._class_track_ids.get(name, set())),
                }
                for name in sorted(set(self._class_detections) | set(self._class_track_ids))
            },
            "outputs": self.outputs,
        }

    def write(self, path: Path) -> Path:
        """Write the JSON report to ``path`` (parents created). Returns ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        LOGGER.info("run statistics written to %s", path)
        return path
