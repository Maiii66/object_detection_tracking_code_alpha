# API reference

Programmatic entry points for embedding the pipeline in your own code.
Import from the package roots:

```python
from config import AppConfig, TrackerConfig, build_parser, setup_logging
from utils import Detection, YOLODetector, BYTETracker, Visualizer, RunStats
import main
```

All signatures below match the current source.

---

## Configuration — `config/config.py`

### `AppConfig`

Top-level configuration. Built by the CLI (`build_parser`), or assembled
manually and validated with `.validate()`.

| Field | Type | Default |
|---|---|---|
| `source` | `str` | `"0"` (webcam) |
| `device` | `str` | `"auto"` |
| `num_threads` | `int` | CPU core count |
| `log_level` | `str` | `"INFO"` |
| `log_file` | `Optional[Path]` | `None` |
| `config_file` | `Optional[Path]` | `None` |
| `detector` | `DetectorConfig` | see below |
| `tracker` | `TrackerConfig` | see below |
| `visualization` | `VisualizationConfig` | see below |
| `output` | `OutputConfig` | see below |

- `AppConfig.to_dict() -> Dict[str, Any]` — nested dict for logging/JSON.

### `DetectorConfig`

`model="yolov8n"`, `conf=0.25`, `iou=0.45`, `imgsz=480`, `half=False`,
`classes=None`, `max_det=100`.

### `TrackerConfig`

`track_high_thresh=0.5`, `track_low_thresh=0.1`, `new_track_thresh=0.6`,
`track_buffer=30`, `match_thresh=0.8`, `frame_rate=30`.

### `VisualizationConfig`

`show_fps=True`, `show_confidence=True`, `draw_trails=False`,
`trail_length=30`, `thickness=2`, `font_scale=0.5`.

### `OutputConfig`

`save_video=True`, `save_csv=True`, `save_stats=True`,
`output_dir=Path("output")`, `codec="mp4v"`.

### Module functions

| Signature | Purpose |
|---|---|
| `build_parser(argv) -> (ArgumentParser, List[str])` | CLI definition; returns parser + remaining args. |
| `setup_logging(level="INFO", log_file=None) -> None` | Console (and optional file) logging. |
| `detect_device(requested="auto") -> str` | Resolve `auto`/`cuda`/`cpu` to a concrete torch device string. |
| `resolve_threads(requested) -> int` | Cap torch/OpenCV thread counts to the machine. |

---

## Detection — `utils/detector.py`

### `Detection`

Dataclass — one detector output box.

| Field | Type |
|---|---|
| `tlbr` | `np.ndarray` — `[x1, y1, x2, y2]` float32 |
| `score` | `float` |
| `class_id` | `int` |
| `class_name` | `str` |

### `YOLODetector(config: Optional[DetectorConfig] = None, device: str = "auto")`

Ultralytics YOLO wrapper producing `List[Detection]`.

| Method | Signature | Notes |
|---|---|---|
| `detect` | `(frame: np.ndarray) -> List[Detection]` | One forward pass; returns empty list on failure. |
| `warmup` | `(shape: Tuple[int, int, int]) -> None` | Dummy pass to settle kernels. |
| `resolve_class_names` | `(names: Sequence[str]) -> List[int]` | Map class-name tokens to ids. |
| `resolve_device` | `(requested="auto") -> str` (staticmethod) | Device string resolution. |

---

## Tracking — `utils/tracker.py`

### `TrackState`

`Enum` — `NEW(0)`, `TRACKED(1)`, `LOST(2)`, `REMOVED(3)`.

### `Track`

One tracked object.

Key fields: `track_id`, `tlbr`, `score`, `class_id`, `class_name`, `state`,
`age`, `hits`, `time_since_update`, `start_frame`, `frame_id`,
`last_detection`, `last_score`, `last_class_id`, `last_class_name`.

| Method | Signature |
|---|---|
| `update` | `(detection: Detection, frame_id: int) -> None` |
| `predict` | `() -> np.ndarray` — Kalman-projected box. |

### `BYTETracker(config: Optional[TrackerConfig] = None)`

Two-stage association (high-score then low-score), Kalman motion model,
Hungarian assignment.

| Method | Signature |
|---|---|
| `update` | `(detections: List[Detection], frame_id: int = 0) -> List[Track]` — returns *currently active* tracks (`TRACKED`). |
| `reset` | `() -> None` — clear all state between sources. |

### `KalmanFilterXYAH(std_weight_position=0.05, std_weight_velocity=0.00625)`

State: `[cx, cy, aspect, height]` + velocities.

| Method | Signature |
|---|---|
| `initiate` | `(measurement: Sequence[float]) -> (mean, covariance)` |
| `predict` | `(mean=None, covariance=None) -> (mean, covariance)` |
| `update` | `(measurement: Sequence[float]) -> (mean, covariance)` |

---

## Visualization & I/O — `utils/visualizer.py`

### `Visualizer(config: Optional[VisualizationConfig] = None)`

| Method | Signature | Notes |
|---|---|---|
| `draw` | `(frame, tracks, fps=0.0) -> np.ndarray` | Boxes, labels, optional trails/FPS. **Mutates and returns `frame` when trails are disabled** — copy first if you need the original. |
| `track_color` | `(track_id) -> Tuple[int, int, int]` (staticmethod) | Deterministic per-id BGR colour. |
| `prune_trails` | `(live_ids: Iterable[int]) -> None` | Drop trails for dead tracks. |
| `reset` | `() -> None` | Clear trail state (new source). |

### `TracksCsvWriter(path: Path)`

| Method | Signature |
|---|---|
| `open` | `() -> None` — writes the header. |
| `write` | `(frame_id: int, tracks: Iterable) -> int` — rows written. |
| `close` | `() -> None` |

Columns: `frame,track_id,class_id,class_name,confidence,x1,y1,x2,y2,age`.

### Module functions

| Signature | Purpose |
|---|---|
| `open_writer(output_path, fps, frame_size, codecs=("mp4v","avc1","MJPG")) -> Optional[cv2.VideoWriter]` | Try codecs in order until one opens. |
| `release_writer(writer) -> None` | Flush and release; no-op on `None`. |

---

## Run statistics — `utils/stats.py`

### `RunStats(source, source_kind, device, model, imgsz)`

Collects per-run counters; written as JSON at the end of a run.

| Method | Signature |
|---|---|
| `record` | `(detections, tracks) -> None` — call once per frame. |
| `set_summary` | `(*, wall_seconds, avg_fps, mean_detect_ms, mean_track_ms, mean_draw_ms, outputs) -> None` |
| `to_dict` | `() -> Dict[str, Any]` |
| `write` | `(path: Path) -> Path` |

Public fields include `frames`, `detections_total`, `avg_fps`,
`mean_detect_ms`, `mean_track_ms`, `mean_draw_ms`, `outputs`; per-class
counts are private and surface only via `to_dict()`/`write()`.

---

## Pipeline — `main.py`

| Signature | Purpose |
|---|---|
| `build_cli(argv) -> (ArgumentParser, List[str])` | The full command-line interface. |
| `main(argv=None) -> int` | Entry point; returns a process exit code. |
| `run(cfg, output_file=None, display=True, class_tokens=None, profile=False) -> int` | Programmatic pipeline. `profile=True` writes `output/profile.pstats` and prints the top functions. |
| `open_source(cfg) -> (cv2.VideoCapture, float, Tuple[int, int])` | Open file/URL/webcam/RTSP/image; returns capture, fps, frame size. |
| `setup_logging(level="INFO", log_file=None)` | Re-exported from `config`. |
| `FrameStats` | Rolling per-stage timings backing the end-of-run summary. |

### Embedding example

```python
from config import AppConfig
from utils import RunStats
import main

cfg = AppConfig(source="clip.mp4", device="cpu")
cfg.output.save_video = False
cfg.output.save_stats = False
exit_code = main.run(cfg, display=False)
```

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Completed (including zero detection frames). |
| `1` | Source failed to open, or an unrecoverable runtime error. |
| `130` | Interrupted with `Ctrl+C` (summary and outputs still written). |
