"""Detection, tracking and rendering components.

Re-exports the pipeline's public API so callers can use::

    from utils import YOLODetector, BYTETracker, Visualizer

The submodules are imported in dependency order: the tracker depends on the
tracker's ``Detection`` type, so ``detector`` must load first.
"""

from .detector import Detection, YOLODetector
from .stats import RunStats
from .tracker import BYTETracker, KalmanFilterXYAH, Track, TrackState
from .visualizer import TracksCsvWriter, Visualizer, open_writer, release_writer

__all__ = [
    "BYTETracker",
    "Detection",
    "KalmanFilterXYAH",
    "RunStats",
    "Track",
    "TrackState",
    "TracksCsvWriter",
    "Visualizer",
    "YOLODetector",
    "open_writer",
    "release_writer",
]
