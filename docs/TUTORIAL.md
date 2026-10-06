# Tutorial

A guided walkthrough of the YOLO + ByteTrack pipeline, from a fresh checkout to
analysed outputs. Every command is real and was executed on Windows PowerShell
against this repository; outputs are abridged where marked.

Prerequisites: Python 3.13, ~1.5 GB disk for PyTorch, and the repository root
as your working directory.

---

## Step 1 — Set up the environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Verify everything is wired together — this runs import checks plus a synthetic
detect → track → draw pass:

```powershell
python verify_env.py
```

Expected tail:

```
[OK] all checks passed
```

If you plan to run the test suite or linter, also install the dev tools:

```powershell
pip install -r requirements-dev.txt
```

## Step 2 — Run on the sample video

```powershell
python main.py --source test_video.mp4
```

A preview window opens; press `q` to quit early or let it run to the end.
The summary looks like:

```
frames processed : 140
unique track IDs : 31
average FPS      : 18.37
detect    53.02 ms  ( 97.4%)
track      0.97 ms  (  1.8%)
draw       0.45 ms  (  0.8%)
run statistics written to output\run_stats.json
run complete
```

Detection dominates the frame budget — that is normal for CPU inference and
tells you where tuning pays off (see README §8).

## Step 3 — Run on an image (headless)

```powershell
python main.py --source test_image.jpg --no-display
```

Produces `output/result.png` with boxes drawn. `--no-display` matters on
machines without a desktop session (SSH, CI).

## Step 4 — Know your outputs

```powershell
Get-ChildItem output\
```

| File | What it is |
|---|---|
| `test_video_tracked.mp4` | Annotated video |
| `result.png` | Annotated still |
| `tracking.csv` | One row per track per frame |
| `run_stats.json` | Run summary with per-class counts |
| `profile.pstats` | cProfile dump (only with `--profile`) |

Peek at the CSV:

```powershell
Get-Content output\tracking.csv -TotalCount 3
```

```csv
frame,track_id,class_id,class_name,confidence,x1,y1,x2,y2,age
0,1,0,person,0.9134,52.31,98.77,114.02,201.44,0
0,2,2,car,0.7842,402.10,118.55,472.88,231.02,0
```

## Step 5 — Tune the run

Common adjustments (full flag table in README §5):

```powershell
# Faster: smaller input size, no video encoding
python main.py --source test_video.mp4 --imgsz 320 --no-video

# See smaller/distant objects: lower the confidence gate
python main.py --source test_video.mp4 --confidence 0.15

# Highlight motion: draw ID trails
python main.py --source test_video.mp4 --draw-trails

# Webcam (device 0) with everything written elsewhere
python main.py --source 0 --output-dir results\session1
```

Re-run with `--imgsz 320` and compare the summary's `detect` line to the
original — the drop is the single biggest lever on CPU.

## Step 6 — Profile a run

```powershell
python main.py --source test_video.mp4 --no-display --profile
```

On completion the top 15 functions by cumulative time are printed and
`output/profile.pstats` is written. Inspect it interactively:

```powershell
python -m pstats output/profile.pstats
(pstats) sort cumtime
(pstats) stats 10
```

Use this when deciding whether a stage (detect / track / draw / encode)
deserves optimisation — the printed run summary already attributes the frame
budget per stage.

## Step 7 — Test and lint

```powershell
pytest -m "not slow" -q     # fast unit tests
pytest -q                   # full suite incl. end-to-end (downloads/uses models/yolov8n.pt)
flake8 .                    # lint: must be clean
black --check .             # format: must be clean
```

The end-to-end tests generate tiny videos/images in a temp directory, run the
real pipeline, and assert outputs exist — no sample data needed beyond the
model weights.

## Step 8 — Embed the pipeline in your own code

```python
from config import AppConfig
import main

cfg = AppConfig(source="clip.mp4", device="cpu")
cfg.detector.imgsz = 320
cfg.output.save_video = False
cfg.output.save_stats = False
cfg.validate()

exit_code = main.run(cfg, display=False)
```

For the full surface (detector, tracker, visualizer, stats), see
[API.md](API.md).

---

## Where to go next

- README §8 — performance tuning knobs and measured baselines
- README §12 — troubleshooting (RTSP, codecs, permissions)
- `scripts/benchmark.py` — measure detection latency on your own hardware
