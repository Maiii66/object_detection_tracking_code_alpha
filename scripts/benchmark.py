"""Micro-benchmark YOLO inference at several image sizes.

Usage::

    python scripts/benchmark.py                       # defaults, 20 frames
    python scripts/benchmark.py --frames 50 --imgsz 320 480
    python scripts/benchmark.py --source test_video.mp4

Times ``YOLODetector.detect`` only (no tracking, no drawing, no encoding),
with warm-up frames discarded, so the numbers are directly comparable to the
"Measured baselines" table in the README. Run it on your own machine rather
than trusting figures published elsewhere.
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DetectorConfig  # noqa: E402
from utils import YOLODetector  # noqa: E402


def load_frame(source: str | None) -> np.ndarray:
    """Return a real frame from ``source``, or a random one if unavailable."""
    if source:
        import cv2

        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                return frame
        print(f"warning: cannot read {source!r}; using a random frame")
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)


def bench_size(detector: YOLODetector, frame: np.ndarray, imgsz: int, frames: int) -> List[float]:
    """Run ``frames`` detections at ``imgsz``; returns per-frame milliseconds."""
    detector.config.imgsz = imgsz
    detector._warm = False
    for _ in range(3):  # warm-up: first call pays for lazy model init
        detector.detect(frame)
    times: List[float] = []
    for _ in range(frames):
        t0 = time.perf_counter()
        detector.detect(frame)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the benchmark and print a markdown table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=20, help="timed frames per size")
    parser.add_argument(
        "--imgsz", type=int, nargs="+", default=[320, 480, 640], help="sizes to test"
    )
    parser.add_argument("--source", default=None, help="media file to take a frame from")
    parser.add_argument("--model", default="yolov8n")
    args = parser.parse_args(argv)

    frame = load_frame(args.source)
    print(f"machine : {platform.processor()} ({os.cpu_count()} logical cores)")
    print(f"python  : {platform.python_version()}  model: {args.model}")
    print(f"frame   : {frame.shape[1]}x{frame.shape[0]}, {args.frames} timed frames\n")

    detector = YOLODetector(DetectorConfig(model=args.model))
    print("| imgsz | mean ms | median ms | p95 ms | FPS |")
    print("|---|---|---|---|---|")
    for imgsz in args.imgsz:
        times = bench_size(detector, frame, imgsz, args.frames)
        mean = statistics.mean(times)
        median = statistics.median(times)
        p95 = sorted(times)[max(0, int(len(times) * 0.95) - 1)]
        print(f"| {imgsz} | {mean:.1f} | {median:.1f} | {p95:.1f} | {1000.0 / mean:.1f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
