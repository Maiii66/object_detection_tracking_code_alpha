"""YOLO inference wrapper.

Isolates all ``ultralytics`` API contact in this one module so the rest of the
pipeline never imports it directly. Benefits:

  * A friendly error message instead of ``ModuleNotFoundError`` when the ML
    stack is missing.
  * Model weights land in ``models/`` rather than the working directory.
  * ``torch.inference_mode()`` and CPU-safe precision are enforced in one place.
  * The returned :class:`Detection` objects use plain numpy, keeping the
    tracker free of any torch tensor leakage.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config.config import PROJECT_ROOT, DetectorConfig, detect_device

LOGGER = logging.getLogger(__name__)

MODEL_DIR = PROJECT_ROOT / "models"

_INSTALL_HINT = """\
ultralytics is not installed, so detection cannot run.

Install the CPU-only stack (this project does NOT use CUDA):

    python -m venv venv
    venv\\Scripts\\Activate.ps1
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt

Then run `python verify_env.py` to confirm the environment.
"""


@dataclass
class Detection:
    """A single object detection in pixel coordinates.

    Attributes:
        tlbr: ``(x1, y1, x2, y2)`` as a float32 numpy array of shape ``(4,)``.
            A numpy array (rather than a tuple) keeps the Hungarian assignment
            and Kalman updates allocation-free; ``tuple(det.tlbr)`` still works
            if a tuple is ever needed.
        score: Confidence in ``[0, 1]``.
        class_id: COCO class index.
        class_name: Human-readable class name.
    """

    tlbr: np.ndarray
    score: float
    class_id: int
    class_name: str

    def __post_init__(self) -> None:
        """Coerce the box to a float32 ``(4,)`` array and normalise scalars."""
        arr = np.asarray(self.tlbr, dtype=np.float32).reshape(4)
        self.tlbr = arr
        self.score = float(self.score)
        self.class_id = int(self.class_id)
        self.class_name = str(self.class_name)

    # -- convenience geometry ---------------------------------------------- #
    @property
    def tlwh(self) -> np.ndarray:
        """Center-x, center-y, width, height."""
        x1, y1, x2, y2 = self.tlbr
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], dtype=np.float32)

    @property
    def xyah(self) -> np.ndarray:
        """Center-x, center-y, aspect ratio (w/h), height."""
        cx, cy, w, h = self.tlwh
        return np.array([cx, cy, w / max(h, 1e-6), h], dtype=np.float32)

    @property
    def area(self) -> float:
        """Box area in pixels, clamped to zero for inverted coordinates."""
        x1, y1, x2, y2 = self.tlbr
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        """Compact single-line representation without numpy array noise."""
        x1, y1, x2, y2 = self.tlbr
        return (
            f"Detection({self.class_name} #{self.class_id} "
            f"{self.score:.2f} [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}])"
        )


class YOLODetector:
    """Wraps a YOLO model and returns :class:`Detection` objects.

    Args:
        config: Inference settings. Defaults to :class:`DetectorConfig`,
            which already holds CPU-friendly values (``imgsz=480``,
            ``half=False``).
        device: ``"auto"``, ``"cpu"``, ``"cuda:0"``, ``0``, ... Device is a
            constructor argument rather than a config field because it is a
            property of the machine, not of the run.

    The ultralytics import is deferred until :meth:`_load` so that importing
    this module never fails on a machine without the ML stack installed.
    """

    def __init__(
        self,
        config: Optional[DetectorConfig] = None,
        device: str = "auto",
    ) -> None:
        """Initialise config, device and rolling timing counters (see class docstring)."""
        self.config = config or DetectorConfig()
        self.device = self.resolve_device(device)
        self._model: Any = None
        self._names: Dict[int, str] = {}
        self._warm = False
        # Rolling per-stage timing, in milliseconds.
        self.avg_detect_ms: float = 0.0
        self.total_frames: int = 0

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    @staticmethod
    def resolve_device(requested: str = "auto") -> str:
        """Resolve ``auto`` to ``cuda:0`` when available, else ``cpu``.

        Delegated to :func:`config.config.detect_device` so device policy lives
        in exactly one place.
        """
        return detect_device(requested)

    @staticmethod
    def resolve_model_path(model: str) -> Path:
        """Resolve a model specifier to a path under ``models/``.

        ``"yolov8n"`` becomes ``models/yolov8n.pt``. Ultralytics downloads to
        whatever path it is given, so pointing it at ``models/`` keeps weights
        out of the working directory (and out of git, via ``.gitignore``).

        An existing file path is returned untouched, so custom weights keep
        working. A bare name without ``.pt`` gets the extension appended.
        """
        candidate = Path(model)
        if candidate.exists() or candidate.is_absolute():
            return candidate
        if candidate.suffix:
            return MODEL_DIR / candidate.name
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        return MODEL_DIR / f"{candidate.name}.pt"

    def _load(self) -> None:
        """Import ultralytics and instantiate the model (idempotent)."""
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(_INSTALL_HINT) from exc

        path = self.resolve_model_path(self.config.model)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        LOGGER.info("loading model %s on device=%s", path, self.device)
        try:
            model = YOLO(str(path))
        except Exception as exc:  # noqa: BLE001 - surface the real cause
            raise RuntimeError(
                f"failed to load YOLO weights from {path}: {exc}\n"
                f"Check the --model value, or delete a corrupt download."
            ) from exc

        model.to(self.device)
        # half precision is unsupported/erratic for many CPU kernels.
        self._model = model
        self._names = {int(k): str(v) for k, v in getattr(model, "names", {}).items()}
        LOGGER.info(
            "model ready: %d classes, names=%s...",
            len(self._names),
            list(self._names.values())[:6],
        )

    @property
    def names(self) -> Dict[int, str]:
        """Class-id to class-name map, populated after the first load."""
        if not self._names:
            self._load()
        return self._names

    def resolve_class_names(self, names: Sequence[str]) -> List[int]:
        """Translate class names (``person``, ``car``) into COCO class ids.

        Raises:
            ValueError: if any name is unknown, listing the valid ones.
        """
        mapping = self.names
        lower = {v.lower(): k for k, v in mapping.items()}
        ids: List[int] = []
        unknown: List[str] = []
        for raw in names:
            key = str(raw).strip().lower()
            if key.isdigit():
                ids.append(int(key))
            elif key in lower:
                ids.append(lower[key])
            else:
                unknown.append(str(raw))
        if unknown:
            sample = sorted(mapping.values())[:15]
            raise ValueError(f"unknown class name(s) {unknown}; expected COCO names like {sample}")
        return sorted(set(ids))

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on one BGR frame.

        Args:
            frame: ``HxWx3`` uint8 BGR array, as returned by
                ``cv2.VideoCapture.read``.

        Returns:
            Detections above the configured thresholds, highest score first.
            Empty list when nothing is found.

        Raises:
            RuntimeError: if ultralytics is missing or weights fail to load.
        """
        self._load()
        cfg = self.config
        started = time.perf_counter()

        import torch

        with torch.inference_mode():
            results = self._model.predict(
                frame,
                conf=cfg.conf,
                iou=cfg.iou,
                imgsz=cfg.imgsz,
                device=self.device,
                half=cfg.half,  # always False on CPU
                classes=cfg.classes or None,
                max_det=cfg.max_det,
                verbose=False,  # suppress per-frame ultralytics spam
            )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.total_frames += 1
        # Exponential moving average smooths the jitter in single-frame times.
        alpha = 0.1
        self.avg_detect_ms = (
            elapsed_ms
            if self.total_frames == 1
            else (1 - alpha) * self.avg_detect_ms + alpha * elapsed_ms
        )

        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        confs = boxes.conf.detach().cpu().numpy().astype(np.float32)
        clss = boxes.cls.detach().cpu().numpy().astype(np.int32)

        detections = [
            Detection(
                tlbr=xyxy[i],
                score=confs[i],
                class_id=clss[i],
                class_name=self._names.get(int(clss[i]), str(int(clss[i]))),
            )
            for i in range(len(xyxy))
        ]
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections

    def warmup(self, shape: Tuple[int, int, int]) -> None:
        """Run one throwaway inference so the first real frame is not slow.

        The very first ``predict`` call pays for lazy weight paging, memory
        allocation and (on CPU) oneDNN primitive selection. Doing it on a blank
        frame keeps that cost out of your measured frame times.
        """
        if self._warm:
            return
        self._load()
        dummy = np.zeros(shape, dtype=np.uint8)
        LOGGER.info("warming up at imgsz=%d on %dx%d", self.config.imgsz, shape[1], shape[0])
        started = time.perf_counter()
        self.detect(dummy)
        self._warm = True
        LOGGER.info("warmup complete in %.0f ms", (time.perf_counter() - started) * 1000.0)
