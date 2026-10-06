# object_detection_tracking_code_alpha

Real-time object detection and multi-object tracking with YOLO and a hand-rolled
ByteTrack implementation, tuned to run on CPU-only hardware.

Built as a reference implementation: every algorithmic step is explicit rather
than delegated to a library, so the detection → association → rendering chain
is readable end to end.

---

## Table of contents

1. [Overview](#1-overview)
2. [Hardware requirements](#2-hardware-requirements)
3. [Installation](#3-installation)
4. [Usage](#4-usage)
5. [CLI flags reference](#5-cli-flags-reference)
6. [Configuration](#6-configuration)
7. [Output files](#7-output-files)
8. [Performance tuning](#8-performance-tuning)
9. [Architecture](#9-architecture)
10. [Algorithm details](#10-algorithm-details)
11. [Testing](#11-testing)
12. [Troubleshooting](#12-troubleshooting)
13. [License](#13-license)
14. [Future enhancements](#14-future-enhancements)
15. [API reference](docs/API.md) — full class/function signatures
16. [Tutorial](docs/TUTORIAL.md) — guided walkthrough, first run to outputs

---

## 1. Overview

### What it does

Reads frames from a webcam, video file, single image, or IP camera, detects
objects with YOLO, assigns each object a stable identity across frames using
ByteTrack, and writes an annotated video plus a per-frame tracking CSV.

### Key features

- **Four input sources** — webcam index, video file, still image, `rtsp://` stream.
- **Hand-rolled ByteTrack** — 8-dimensional Kalman filter plus Hungarian
  assignment, no `lapx` compile step and no reliance on ultralytics' built-in
  tracker.
- **CPU-first defaults** — `yolov8n` at `imgsz=480` with `half=False` and an
  explicit thread count, tuned for an Intel Iris Xe iGPU.
- **Two-stage association** — keeps low-confidence detections to recover tracks
  through occlusion, which is the core ByteTrack insight.
- **Deterministic track colours** — hue derived from the track id via the
  golden-ratio conjugate, so IDs stay visually distinct for hundreds of objects.
- **Layered configuration** — dataclass defaults < YAML file < CLI flags, with
  validation that fails loudly and names the offending field.
- **Per-stage timing** — the run summary attributes the frame budget to
  detection, tracking and drawing so bottlenecks are identifiable.
- **Run statistics** — every run writes `output/run_stats.json`: frames, FPS,
  per-stage means and a per-class breakdown of detections and track ids.
- **Profiling on demand** — `--profile` adds a cProfile report
  (`output/profile.pstats`) for digging into where time actually goes.
- **Graceful degradation** — codec fallback chain, NaN/zero FPS sanitising,
  and a clear install hint instead of a bare `ModuleNotFoundError`.

### Use cases

- Counting and measuring object throughput on a single CPU budget.
- Producing frame-accurate tracking data for downstream analysis (the CSV).
- Studying multi-object tracking internals without a black-box tracker.

---

## 2. Hardware requirements

| Component | Minimum | Notes |
|---|---|---|
| OS | Windows 10/11, Linux, macOS | Developed on Windows 11 |
| Python | 3.9+ | Verified on 3.13.2 |
| CPU | Any x86-64 with 4+ threads | Developed on i5-1235U (10C/12T) |
| RAM | 4 GB | ~2 GB while a model is loaded |
| Disk | 2 GB | Torch CPU wheel is ~200 MB; models are ~6 MB each |
| GPU | **Not used** | CPU-only inference throughout |

Expected throughput on CPU: roughly **5–15 FPS** at `imgsz=480` with
`yolov8n`, depending heavily on core count and scene complexity. Detection
dominates the frame budget; see [Performance tuning](#8-performance-tuning).

---

## 3. Installation

### Prerequisites

Python 3.9 or newer. Verify with `python --version`.

### Create a virtual environment

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

On macOS/Linux the activation line is `source venv/bin/activate`.

### Install PyTorch (CPU-only) — critical

A plain `pip install torch` on Windows downloads the **CUDA** build, roughly
2.5 GB, which an Intel Iris Xe cannot use. Always specify the CPU index:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Both packages must come from the same command: `torchvision` is version-coupled
to `torch`, and installing them separately lets pip resolve a mismatched pair.

Verify you did not get the CUDA build:

```powershell
python -c "import torch; print(torch.__version__)"
```

The output **must** contain `+cpu` (for example `2.14.1+cpu`). If it does not,
uninstall and reinstall with the `--index-url` above.

### Install remaining dependencies

```powershell
pip install -r requirements.txt
```

### Verify the environment

```powershell
python verify_env.py
```

Checks every import, confirms the PyTorch build is CPU-only, reports the active
thread count, verifies OpenCV has FFmpeg support, and runs a synthetic
detect → track → draw pass. Exits `0` on success, `1` otherwise.

### Install the development tools (optional)

Only needed to run the test suite, linter and formatter:

```powershell
pip install -r requirements-dev.txt
```

---

## 4. Usage

```powershell
# Webcam
python main.py --source 0

# Video file
python main.py --source data/sample.mp4

# Single image (writes output/result.png)
python main.py --source data/photo.jpg

# IP camera
python main.py --source rtsp://user:pass@192.168.1.50:554/stream

# YAML configuration, with one CLI override
python main.py --config example_config.yaml --imgsz 320

# Only people and cars
python main.py --source 0 --classes person car

# Preview only, write nothing
python main.py --source clip.mp4 --no-video --no-csv

# Profile the run: writes output/profile.pstats and prints hot functions
python main.py --source clip.mp4 --no-display --profile

# Skip the JSON statistics report
python main.py --source clip.mp4 --no-stats

# Custom output directory (video, CSV, stats and profile all land there)
python main.py --source clip.mp4 --output-dir results/session1

# Batch: annotate every clip in a folder (PowerShell)
foreach ($f in Get-ChildItem data/*.mp4) {
    python main.py --source $f.FullName --no-display --output-dir "out/$($f.BaseName)"
}

# Read the first few tracking rows for analysis
Get-Content output/tracking.csv -TotalCount 5

# Inspect what the run saw
Get-Content output/run_stats.json
```

Press `q` in the preview window to quit. `Ctrl+C` exits cleanly, still
prints the run summary, and writes the statistics/profile artefacts too.

---

## 5. CLI flags reference

| Flag | Default | Description |
|---|---|---|
| `--source` | `0` | Webcam index, image/video path, or stream URL |
| `--config` | none | YAML config file (see `example_config.yaml`) |
| `--model` | `yolov8n` | YOLO weights; `yolo26n` is also available |
| `--confidence` | `0.25` | Detection confidence threshold |
| `--iou` | `0.45` | NMS IoU threshold |
| `--imgsz` | `480` | Inference size; must be a multiple of 32 |
| `--classes` | all | Restrict to class ids or names, e.g. `person car` or `0 2` |
| `--track-high-thresh` | `0.5` | Lower bound for stage-1 association |
| `--track-low-thresh` | `0.1` | Lower bound for stage-2 recovery |
| `--new-track-thresh` | `0.6` | Minimum score to spawn a new track |
| `--track-buffer` | `30` | Frames a lost track survives |
| `--match-thresh` | `0.8` | Minimum IoU for stage-2 association |
| `--device` | `auto` | `auto`, `cpu`, or `cuda:0` |
| `--threads` / `--num-threads` | `6` | `torch.set_num_threads()` |
| `--output` | `output/` | Output video file or directory |
| `--output-dir` | `output/` | Output directory |
| `--save-video` / `--no-save-video` | save | Toggle video output |
| `--no-video` | — | Alias for `--no-save-video` |
| `--save-csv` / `--no-save-csv` | save | Toggle CSV output |
| `--no-csv` | — | Alias for `--no-save-csv` |
| `--no-stats` | stats written | Skip `output/run_stats.json` |
| `--profile` | off | Write `output/profile.pstats` (cProfile) after the run |
| `--draw-trails` / `--no-draw-trails` | off | Motion trails per track |
| `--no-fps` | HUD shown | Hide the FPS/resolution HUD |
| `--no-display` | display shown | Headless mode; no preview window |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--log-file` | stderr only | Also write logs to this file |

---

## 6. Configuration

Precedence, lowest to highest:

```
dataclass defaults  <  YAML config  <  CLI flags
```

CLI flags beat YAML, and anything not supplied on either side keeps its
dataclass default. Booleans use `--flag` / `--no-flag` pairs so "unset" is
distinguishable from "explicitly set to the default".

### Using flags

```powershell
python main.py --source clip.mp4 --confidence 0.6 --model yolov8s
```

### Using YAML

```powershell
python main.py --config example_config.yaml
python main.py --config example_config.yaml --confidence 0.7   # CLI wins
```

`example_config.yaml` documents every option with its default.

### Validation

Invalid configuration fails immediately with a message naming the field, and
exit code `2`:

```
ERROR  main: invalid configuration: detector.imgsz must be a positive multiple of 32, got 999
```

Checked: `0 < confidence < 1`, `0 < iou < 1`, `imgsz > 0` and a multiple of 32,
`0 < track_low_thresh < track_high_thresh < 1`, `0 < new_track_thresh < 1`,
`0 < match_thresh <= 1`, `track_buffer >= 1`, `num_threads >= 1`, class ids
non-negative and within the 80 COCO classes, the source file exists, and the
output directory is writable. `half=True` is downgraded to `False` with a
warning rather than failing, since it is unusable but not fatal on CPU.

---

## 7. Output files

| File | Contents |
|---|---|
| `output/<name>_tracked.mp4` | Annotated video. Falls back to `.avi` if the codec requires it. |
| `output/result.png` | Annotated still, when the source is a single image. |
| `output/tracking.csv` | One row per track per frame. |
| `output/run_stats.json` | Run summary: frames, FPS, per-stage means, per-class counts (suppress with `--no-stats`). |
| `output/profile.pstats` | cProfile dump, only with `--profile`. Open with `python -m pstats output/profile.pstats`. |

CSV columns:

```
frame,track_id,class_id,class_name,confidence,x1,y1,x2,y2,age
```

| Column | Meaning |
|---|---|
| `frame` | Zero-based frame index |
| `track_id` | Stable identity for the object's lifetime |
| `class_id` | COCO class index |
| `class_name` | COCO class name |
| `confidence` | Detection score, 4 decimal places |
| `x1,y1,x2,y2` | Bounding box in pixels |
| `age` | Frames the track has existed |

Example:

```csv
frame,track_id,class_id,class_name,confidence,x1,y1,x2,y2,age
0,1,0,person,0.9134,52.31,98.77,114.02,201.44,0
0,2,2,car,0.7842,402.10,118.55,472.88,231.02,0
```

`run_stats.json` example (structure abridged, numbers from a sample 140-frame
4K run — they vary by machine and run):

```json
{
  "source": "test_video.mp4",
  "source_kind": "video",
  "device": "cpu",
  "model": "yolov8n",
  "imgsz": 480,
  "wall_seconds": 25.8,
  "frames": 140,
  "detections_total": 796,
  "unique_track_ids": 31,
  "avg_fps": 18.4,
  "mean_detect_ms": 53.0,
  "mean_track_ms": 1.0,
  "mean_draw_ms": 0.4,
  "per_class": {
    "car": { "detections": 723, "unique_track_ids": 31 },
    "person": { "detections": 32, "unique_track_ids": 0 }
  },
  "outputs": { "video": "output/test_video_tracked.mp4" }
}
```

---

## 8. Performance tuning

Every run prints a per-stage breakdown so you can see where time actually goes:

```
frames processed : 300
unique track IDs : 4
average FPS      : 9.87
detect  82.40 ms  ( 87.9%)
track    9.10 ms  (  9.7%)
draw     1.55 ms  (  1.7%)
```

### Faster on CPU

- **`--imgsz 320`** — the single biggest lever. Compute scales roughly with the
  square of image size, so 480 → 320 is about a 2.2x speedup.
- **`--model yolov8n`** — smallest YOLO; already the default.
- **Raise `--threads`.** The default is 6 on a 12-thread machine. Test 6, 8 and
  10: hyperthreading helps BLAS-bound convolution less than it helps scalar code.
- **`--no-video`** — skips encoding, which is not free at low resolutions.
- **Close other applications.** Torch competes for the same cores.

### Better accuracy

- **`--model yolov8s`**, or `yolov8m` if throughput allows.
- **Raise `--imgsz`** to 640.
- **Lower `--confidence`** to 0.3–0.4. Be aware this feeds more low-confidence
  detections into the tracker, where stage 2 can use them — but also risks
  spurious new tracks above `new_track_thresh`.
- **Raise `--track-buffer`** when objects are frequently occluded, at the cost of
  slower removal of objects that genuinely left.

### Measured baselines

Detection only (no tracking/drawing/encoding), `yolov8n`, CPU-only, on an
**i5-1235U (10C/12T)**, 20 timed frames after warm-up, measured
2026-10-06 with `python scripts/benchmark.py`:

| imgsz | mean ms | median ms | p95 ms | FPS |
|---|---|---|---|---|
| 320 | 26.5 | 26.2 | 30.5 | 37.8 |
| 480 | 39.1 | 38.7 | 43.9 | 25.6 |
| 640 | 59.5 | 55.5 | 63.7 | 16.8 |

End-to-end throughput is lower than these figures suggest: tracking, drawing
and video encoding sit on top of detection. Full 4K runs at `imgsz=480`
measured **18–22 FPS** overall (detect ~41–53 ms / track ~1 ms / draw <4 ms).

Reproduce on your own hardware:

```powershell
python scripts/benchmark.py --source test_video.mp4
```

Numbers depend on core count, thermals and scene complexity — benchmark
locally rather than trusting published figures.

---

## 9. Architecture

```
main.py                    CLI, setup (_prepare), frame loop, timing summary
  |
  +-- config/
  |     config.py           dataclasses, CLI, YAML merging, validation
  |     __init__.py         re-exports AppConfig, build_parser
  |
  +-- utils/
  |     detector.py         ultralytics wrapper -> Detection
  |     tracker.py          Kalman + Hungarian -> Track
  |     visualizer.py       drawing, VideoWriter, CSV writer
  |     stats.py            RunStats -> output/run_stats.json
  |     __init__.py         re-exports the public API
  |
  +-- scripts/
  |     benchmark.py        detector micro-benchmark (README baselines)
  |     make_tutorial_video.py  renders the captioned tutorial MP4
  |
  +-- tests/                pytest suite (unit + end-to-end)
  +-- docs/                 API.md, TUTORIAL.md
  +-- verify_env.py         environment and integration checks
```

The per-frame path is linear and lives in `main._step`, called by
`main._frame_loop`:

```
cap.read() -> detector.detect() -> tracker.update() -> visualizer.draw()
            -> writer.write() / csv.write() -> cv2.imshow()
```

**Layering rules.** `config` depends on nothing. `detector` depends on `config`.
`tracker` depends on `config` and `detector` (for the `Detection` type).
`visualizer` depends on `config`. `stats` depends on nothing. Only `main.py`
knows about all of them.

`ultralytics` is imported inside `YOLODetector._load`, never at module scope, so
every module imports cleanly with no ML stack installed — which is what makes
offline testing of the tracker and visualizer possible.

---

## 10. Algorithm details

### ByteTrack: two-stage association

A conventional tracker discards detections below its confidence threshold. That
throws away exactly the boxes you need when an object is occluded or
motion-blurred. ByteTrack keeps them and uses them in a second pass.

Per frame:

1. **Predict.** Advance every active track one step through the Kalman filter.
2. **Stage 1 — confident detections vs active tracks.** Score each pair by
   Mahalanobis distance, gated by `CHI2INV95 = 9.4877` (the 95% chi-square
   threshold for 4 degrees of freedom). Solve the assignment optimally.
3. **Stage 2 — weak detections vs lost tracks.** Tracks stage 1 could not match
   are re-associated using plain IoU against `track_low_thresh` detections.
   This is the occlusion-survival step.
4. **Stage 3 — new tracks.** Still-unmatched detections scoring above
   `new_track_thresh` spawn new tracks.
5. **Retire.** Tracks lost for more than `track_buffer` frames are removed.

Thresholds interact as follows: `track_low_thresh < track_high_thresh`, and a
detection in between feeds stage 2. Validation enforces the ordering.

### Kalman filter

An 8-dimensional constant-velocity filter over `(cx, cy, aspect_ratio, height)`
plus their velocities. Aspect ratio and height are tracked rather than width
because they stay closer to linear as an object approaches or recedes.

Process noise is parameterised by measurement standard deviations
(`1/20` for position, `1/160` for velocity), so the values remain meaningful
across image resolutions. A key detail: the gating-distance matrix must be
**normalised or left un-normalised consistently**. The reference ByteTrack
implementation divides by `sqrt(CHI2INV95)` and compares against `0.98`; this
implementation uses raw Mahalanobis distances against `9.4877`. Mixing the two
conventions silently rejects every pairing.

### Hungarian assignment

`scipy.optimize.linear_sum_assignment` finds the minimum-cost one-to-one
matching. Pairs exceeding the cost ceiling are masked to a large sentinel
*before* solving, not filtered afterwards — this keeps the solver globally
optimal over the allowed pairs rather than greedily accepting the cheapest pair
and stranding a better overall matching.

### Track lifecycle

```
NEW ──► TRACKED ──► LOST ──► REMOVED
          ▲           │
          └───────────┘   recovered by stage 2
```

A track lives in exactly one of two lists (`_active` or `_lost`), which removes
any ambiguity about double-counting a recovered object. `Track` sets `eq=False`,
because a generated `__eq__` would compare numpy arrays element-wise and raise
"truth value of an array is ambiguous".

---

## 11. Testing

### Environment check

```powershell
python verify_env.py
```

### Test suite

```powershell
pip install -r requirements-dev.txt

pytest                    # full suite: 80 unit tests + integration
pytest -m "not slow"      # fast tests only (skip the end-to-end run)
pytest tests/test_tracker.py -q
```

The integration tests (`-m slow`) run the real pipeline on tiny generated
videos/images and are skipped automatically when `models/yolov8n.pt` is
absent.

### Lint and format

```powershell
flake8 .
black --check .
black .                   # reformat
```

### Benchmark

```powershell
python scripts/benchmark.py --source test_video.mp4
```

### Offline tracker check

The tracker and visualizer need no model, so they can be validated with
synthetic detections before installing PyTorch:

```python
import numpy as np
from config import TrackerConfig
from utils import Detection, BYTETracker

tracker = BYTETracker(TrackerConfig())
for frame_id in range(30):
    x = 100 + 4 * frame_id
    det = Detection(tlbr=np.array([x, 100, x + 40, 180], np.float32),
                    score=0.9, class_id=0, class_name="person")
    for track in tracker.update([det], frame_id=frame_id):
        print(track.track_id, track.last_class_name)
```

A single object moving linearly should keep ID `1` for all 30 frames. Note
that `verify_env.py` also performs a version of this check automatically.

---

## 12. Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Make sure the virtual environment is active first.

### `torch.__version__` has no `+cpu`

The CUDA build was installed. Remove and reinstall:

```powershell
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Very low FPS (under 5)

- Lower `--imgsz` to 320 — the biggest single lever.
- Confirm `--model yolov8n` (not `yolov8s` or larger).
- Raise `--threads`; test 6, 8, 10.
- Use `--no-video` to remove encoding cost.
- Read the run summary to see which stage dominates.

### Video file is not created

- Check the `output/` directory exists and is writable — validation creates it.
- Check the log for `no usable codec`; that means video output was disabled and
  the run continued in preview-only mode.
- `output/tracking.csv` is still written even when video fails, so you can
  confirm the pipeline ran.

### Video has no annotations

- Lower `--confidence`. The default is `0.25`; very small or distant objects may
  sit below it.
- Confirm the model downloaded into `models/`, not somewhere else.
- Raise `--track-high-thresh` and `--track-low-thresh` if detections exist but
  nothing is being tracked.

### CSV is empty

The header row is always written, so an empty body means no tracks were
confirmed in any frame. Check `--confidence` and `--new-track-thresh`.

### Preview window does not appear

Use `--no-display`. The window needs a desktop session and will fail over SSH
or in some CI environments.

### `Assertion failed: !filename_pattern.empty`

An OpenCV error raised while opening an image-sequence writer. Usually a
corrupt or zero-byte video file. Check the source path.

### Port 445 / runtime permission errors

Windows firewall or antivirus scanning `models/` on first download. Retry.

### Model weights not found / download fails

`yolov8n.pt` is downloaded on first run into `models/`. If the machine has no
network access, copy the file there manually from another machine, or set
`--model` to a path you already have. Offline runs fail fast with a clear
error.

### RTSP camera will not connect

- Check the URL: `rtsp://user:pass@192.168.1.5:554/stream`.
- Verify credentials and that the camera and PC are on the same network.
- Some cameras need a specific profile/path — open the stream in VLC first.
- Corporate firewalls often block TCP 554. Test with
  `ffplay rtsp://...` to separate camera problems from code problems.

### `PermissionError` writing outputs

The `output/` (or `--output-dir`) path is read-only or locked by another
program. Either free the directory, or point elsewhere:

```powershell
python main.py --source clip.mp4 --output-dir $env:TEMP\tracking_out
```

### `No module named 'pytest'` (or `flake8` / `black`)

Development tools are optional and separate:

```powershell
pip install -r requirements-dev.txt
```

### `--device cuda` falls back to `cpu`

The installed PyTorch build has no CUDA support (`torch.cuda.is_available()`
is `False` — `verify_env.py` reports this). Reinstall a CUDA build of torch
if the machine has an NVIDIA GPU, or stay on CPU.

### `run_stats.json` or `profile.pstats` missing

- `run_stats.json` — suppressed by `--no-stats`; check the log for
  `run statistics written to ...` to confirm it was written.
- `profile.pstats` — only written when `--profile` is passed, and only once
  the run finishes (or is interrupted with `Ctrl+C`).

---

## 13. License

Released under the **GNU Affero General Public License v3.0**. See `LICENSE`.

This matches the license of [Ultralytics](https://github.com/ultralytics/ultralytics),
which this project depends on. Note that depending on an AGPL-licensed library
as a dependency does not by itself require your own code to be AGPL — that
choice is yours, and AGPL is a reasonable default if you plan to run this as a
network service.

Copyright (C) 2026 maiyarasu.

---

## 14. Future enhancements

- **Appearance embeddings** (DeepSORT-style ReID network) to recover tracks
  across long occlusions where ByteTrack's spatial-only association fails.
- **GPU inference.** The code already accepts `--device cuda:0` and `half=True`
  works there; it is simply untested on this hardware.
- **ONNX / OpenVINO export** for faster CPU inference than PyTorch.
- **Object counting** with in/out line crossing and region geofencing.
- **Multi-camera** support with per-camera tracker instances.
- **Trajectory export** (MOT-challenge format) for benchmark comparison.

---

## Attribution

- **YOLO** — [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0)
- **ByteTrack** — Zhang et al., *ByteTrack: Multi-Object Tracking by Associating
  Every Detection Box*, ECCV 2022
- **Kalman filtering and Hungarian assignment** — Kalman (1960) and Kuhn (1955),
  as applied in SORT (Bewley et al., 2016) and DeepSORT ( Wojke et al., 2017)
- **OpenCV** — [opencv.org](https://opencv.org) (Apache-2.0)