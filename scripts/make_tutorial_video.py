"""Render a captioned slide sequence for the project tutorial.

Produces ``output/tutorial.mp4`` (OpenCV mp4v), a short video walking through
setup, runs, outputs and next steps. Text is ASCII-only (cv2.putText renders
Hershey stroke fonts, no Unicode).

Usage::

    python scripts/make_tutorial_video.py [--out PATH] [--fps 24] [--seconds 4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

WIDTH, HEIGHT = 1280, 720
BG = (18, 18, 24)
FG = (235, 235, 235)
ACCENT = (80, 200, 255)
DIM = (140, 140, 150)

SLIDES: list[tuple[str, list[tuple[str, bool]]]] = [
    (
        "YOLO + ByteTrack Tracker",
        [
            ("Object detection and multi-object tracking", False),
            ("in one CPU-friendly pipeline", False),
            ("", False),
            ("github.com/Maiii66/object_detection_tracking_code_alpha", True),
        ],
    ),
    (
        "Step 1 - Setup",
        [
            ("python -m venv venv", True),
            (".\\venv\\Scripts\\Activate.ps1", True),
            ("pip install -r requirements.txt", True),
            ("python verify_env.py   ->  [OK] all checks passed", True),
        ],
    ),
    (
        "Step 2 - First run",
        [
            ("python main.py --source test_video.mp4", True),
            ("", False),
            ("Preview window: press q to quit", False),
            ("Summary: frames, unique IDs, FPS, per-stage ms", False),
        ],
    ),
    (
        "Step 3 - Images and headless runs",
        [
            ("python main.py --source test_image.jpg --no-display", True),
            ("", False),
            ("Writes output/result.png with boxes drawn", False),
            ("--no-display works over SSH and in CI", False),
        ],
    ),
    (
        "Step 4 - Your outputs",
        [
            ("output/<name>_tracked.mp4   annotated video", False),
            ("output/tracking.csv         one row per track", False),
            ("output/run_stats.json       run + per-class stats", False),
            ("output/profile.pstats       cProfile (with --profile)", False),
        ],
    ),
    (
        "Step 5 - Tune the run",
        [
            ("--imgsz 320          fastest on CPU", False),
            ("--confidence 0.15    catch small objects", False),
            ("--draw-trails        show motion history", False),
            ("--output-dir DIR     write anywhere", False),
        ],
    ),
    (
        "Step 6 - Measure",
        [
            ("python main.py --source clip.mp4 --profile", True),
            ("python scripts/benchmark.py --source test_video.mp4", True),
            ("", False),
            ("Baselines: 37.8 FPS @320, 25.6 FPS @480 (yolov8n, CPU)", False),
        ],
    ),
    (
        "Step 7 - Test and lint",
        [
            ("pytest -q            80 tests, integration included", False),
            ("flake8 .             lint: clean", False),
            ("black --check .      format: clean", False),
        ],
    ),
    (
        "Next steps",
        [
            ("docs/TUTORIAL.md   guided walkthrough", False),
            ("docs/API.md        embed the pipeline in your code", False),
            ("README.md          tuning, architecture, troubleshooting", False),
        ],
    ),
]


def wrap(text: str, max_chars: int = 58) -> list[str]:
    """Hard-wrap ``text`` on whitespace for single-line slide bullets."""
    words, lines, current = text.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > max_chars:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines or [""]


def render_slide(
    title: str, bullets: list[tuple[str, bool]], index: int, total: int, progress: float
) -> np.ndarray:
    """Draw one frame: title, wrapped bullets, footer and progress bar."""
    frame = np.full((HEIGHT, WIDTH, 3), BG, dtype=np.uint8)

    cv2.rectangle(frame, (0, 0), (WIDTH, 8), ACCENT, -1)
    cv2.putText(frame, title, (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.7, FG, 3, cv2.LINE_AA)

    y = 200
    for text, is_code in bullets:
        for line in wrap(text):
            color = ACCENT if is_code else FG
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                frame, "-  " if not is_code else "   ", (60, y), font, 0.8, DIM, 2, cv2.LINE_AA
            )
            cv2.putText(frame, line, (110, y), font, 0.8, color, 2, cv2.LINE_AA)
            y += 52
        y += 14

    footer = "docs/TUTORIAL.md  -  full walkthrough"
    cv2.putText(
        frame, footer, (60, HEIGHT - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, DIM, 2, cv2.LINE_AA
    )
    cv2.putText(
        frame,
        f"{index + 1} / {total}",
        (WIDTH - 140, HEIGHT - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        DIM,
        2,
        cv2.LINE_AA,
    )

    bar_y = HEIGHT - 18
    cv2.rectangle(frame, (60, bar_y), (WIDTH - 60, bar_y + 6), (50, 50, 60), -1)
    x_end = int(60 + (WIDTH - 120) * progress)
    cv2.rectangle(frame, (60, bar_y), (x_end, bar_y + 6), ACCENT, -1)
    return frame


def make_video(out_path: Path, fps: int, seconds: float) -> int:
    """Render every slide for ``seconds`` each and write the MP4."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_per_slide = max(1, int(round(fps * seconds)))
    total_slides = len(SLIDES)

    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        print(f"error: could not open VideoWriter for {out_path}", file=sys.stderr)
        return 1

    total_frames = frames_per_slide * total_slides
    written = 0
    try:
        for i, (title, bullets) in enumerate(SLIDES):
            for f in range(frames_per_slide):
                progress = (i + f / frames_per_slide) / total_slides
                frame = render_slide(title, bullets, i, total_slides, progress)
                writer.write(frame)
                written += 1
    finally:
        writer.release()

    print(
        f"wrote {out_path} ({total_frames} frames, {total_slides} slides, "
        f"{fps} fps, {total_frames / fps:.1f}s)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "output" / "tutorial.mp4")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seconds", type=float, default=4.0, help="seconds each slide is shown")
    args = parser.parse_args(argv)
    if args.seconds <= 0 or args.fps <= 0:
        parser.error("--seconds and --fps must be positive")
    return make_video(args.out, args.fps, args.seconds)


if __name__ == "__main__":
    raise SystemExit(main())
