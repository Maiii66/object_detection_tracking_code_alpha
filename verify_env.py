"""Environment verification for object_detection_tracking_code_alpha.

Checks that every dependency is importable and correctly built, with special
attention to the two failure modes that matter on a CPU-only Windows machine:

  * a CUDA build of PyTorch installed when CPU was intended (huge, slow, and
    still unusable on an Intel Iris Xe), and
  * a too-small thread count, which silently costs a large amount of
    throughput.

Run it after ``pip install -r requirements.txt``::

    python verify_env.py

Exit code is 0 when every check passes, 1 otherwise, so it can gate CI or a
setup script.
"""

from __future__ import annotations

import importlib
import os
import platform
import sys
import time
from typing import List, Optional, Tuple

RESULTS: List[Tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str = "") -> None:
    """Store one check result."""
    RESULTS.append((name, passed, detail))


def mark(passed: bool) -> str:
    """Status label. Plain ASCII so no console encoding can break the report."""
    return "PASS" if passed else "FAIL"


def safe_print(text: str) -> None:
    """Print, degrading gracefully if the console cannot encode the text."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))


def check_import(module: str, label: Optional[str] = None) -> Optional[object]:
    """Import a module, timing it. Records pass/fail and returns the module."""
    label = label or module
    started = time.perf_counter()
    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - any failure is a failed check
        elapsed = (time.perf_counter() - started) * 1000
        record(label, False, f"{type(exc).__name__}: {exc}")
        return None
    elapsed = (time.perf_counter() - started) * 1000
    version = getattr(mod, "__version__", None) or mod.__name__
    record(label, True, f"{version} ({elapsed:.0f} ms)")
    return mod


def main() -> int:
    """Run every check and print a report. Returns the process exit code."""
    header = "=" * 72
    safe_print(header)
    safe_print("ENVIRONMENT VERIFICATION")
    safe_print(header)
    safe_print(f"Python     : {platform.python_version()} ({sys.executable})")
    safe_print(f"Platform   : {platform.system()} {platform.release()} "
               f"({platform.machine()})")
    safe_print(f"CPU cores  : {os.cpu_count()} logical")
    safe_print("")

    # ---- 1. Core numeric / IO stack ------------------------------------- #
    safe_print("1. Core dependencies")
    numpy = check_import("numpy")
    cv2 = check_import("cv2", "opencv-python")
    scipy = check_import("scipy")
    yaml = check_import("yaml", "PyYAML")
    safe_print("")

    # ---- 2. Deep learning stack ------------------------------------------ #
    safe_print("2. Deep learning stack")
    torch = check_import("torch")
    check_import("torchvision")
    check_import("ultralytics")
    safe_print("")

    # ---- 3. PyTorch build correctness ------------------------------------ #
    if torch is not None:
        version = torch.__version__
        is_cpu = "+cpu" in version
        record("torch is a CPU build", is_cpu,
               version if is_cpu
               else f"{version} is NOT +cpu - reinstall using "
                    f"--index-url https://download.pytorch.org/whl/cpu")
        record("threads >= 4", torch.get_num_threads() >= 4,
               f"torch.get_num_threads() = {torch.get_num_threads()}")
        try:
            cuda = torch.cuda.is_available()
            if cuda:
                safe_print("   note: CUDA is available; CPU-only defaults still apply")
            record("cuda availability probed", True, f"cuda.is_available() = {cuda}")
        except Exception as exc:  # noqa: BLE001
            record("cuda availability probed", False, str(exc))
        safe_print("")

    # ---- 4. Project modules ---------------------------------------------- #
    safe_print("3. Project modules")
    config_mod = check_import("config", "config package")
    if config_mod is not None:
        record("AppConfig importable", hasattr(config_mod, "AppConfig"))
    check_import("utils.detector")
    check_import("utils.tracker")
    check_import("utils.visualizer")
    safe_print("")

    # ---- 5. Cross-component integration ---------------------------------- #
    safe_print("4. Integration")
    if scipy is not None:
        try:
            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment(
                __import__("numpy").array([[0.1, 5.0], [5.0, 0.1]])
            )
            record("Hungarian assignment", len(rows) == 2,
                   f"matched {len(rows)} pairs")
        except Exception as exc:  # noqa: BLE001
            record("Hungarian assignment", False, str(exc))

    if numpy is not None and cv2 is not None:
        info = cv2.getBuildInformation()
        has_ffmpeg = "FFMPEG:                      YES" in info
        record("OpenCV has FFmpeg", has_ffmpeg,
               "video writing enabled" if has_ffmpeg
               else "video output will be unavailable")
        if config_mod is not None and cv2 is not None:
            from config.config import VisualizationConfig
            from utils.detector import Detection
            from utils.tracker import BYTETracker
            from utils.visualizer import Visualizer
            frame = numpy.zeros((240, 320, 3), dtype=numpy.uint8)
            det = Detection(tlbr=numpy.array([10, 10, 80, 90]),
                            score=0.9, class_id=0, class_name="person")
            tracks = BYTETracker().update([det], frame_id=0)
            viz = Visualizer(VisualizationConfig())
            out = viz.draw(frame, tracks, fps=12.5)
            record("synthetic detect->track->draw", out.shape == frame.shape,
                   f"tracks={len(tracks)} out={out.shape}")

    safe_print("")

    # ---- Report ------------------------------------------------------------ #
    safe_print(header)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    for name, ok, detail in RESULTS:
        glyph = mark(ok)
        suffix = f"  {detail}" if detail else ""
        safe_print(f"  {glyph} {name:<32}{suffix}")
    safe_print(header)

    if passed == total:
        safe_print(f"RESULT: PASS - {passed}/{total} checks")
        safe_print("Ready to run:  python main.py --source 0")
        return 0

    safe_print(f"RESULT: FAIL - {passed}/{total} checks passed")
    failed = [n for n, ok, _ in RESULTS if not ok]
    safe_print("Failing checks: " + ", ".join(failed))
    safe_print("")
    safe_print("Common fixes:")
    safe_print("  python -m pip install --upgrade pip setuptools wheel")
    safe_print("  pip install torch torchvision "
               "--index-url https://download.pytorch.org/whl/cpu")
    safe_print("  pip install -r requirements.txt")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())