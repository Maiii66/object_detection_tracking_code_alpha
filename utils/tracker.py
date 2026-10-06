"""Hand-rolled ByteTrack: multi-object tracking with a Kalman filter.

ByteTrack's insight is that a detector which throws away low-confidence boxes
loses information exactly when an object is occluded or motion-blurred. So
instead of discarding them, ByteTrack keeps them and uses them in a *second*
association pass to recover tracks that the high-confidence pass lost.

Per frame:
    1. Predict every active track forward with the Kalman filter.
    2. Stage 1 - match high-confidence detections to active tracks using
       Mahalanobis distance gated by each track's own covariance.
    3. Stage 2 - match the leftover low-confidence detections to the tracks
       stage 1 could not match, using plain IoU. This is the occlusion-survival
       step and is what distinguishes ByteTrack from a plain IoU tracker.
    4. Stage 3 - spawn new tracks from still-unmatched high-confidence
       detections at or above ``new_track_thresh``.
    5. Retire tracks that have been lost for more than ``track_buffer`` frames.

Only ``numpy`` and ``scipy`` are required. ``linear_sum_assignment`` supplies
the optimal (Hungarian) assignment, so no ``lapx``/``lap`` compile step is
needed and ultralytics' built-in tracker is never touched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import linear_sum_assignment

from config.config import CHI2INV95, TrackerConfig
from utils.detector import Detection

LOGGER = logging.getLogger(__name__)

# Cost assigned to a forbidden pairing. Must exceed every real distance so the
# Hungarian solver prefers an allowed (even mediocre) pairing over this.
BIG_COST = 1e5

# IoU lives in [0, 1], so a distance of 1.0 means "no overlap at all".
NO_OVERLAP_IOU = 1.0


class TrackState(IntEnum):
    """Lifecycle of a track.

    ``NEW -> TRACKED -> LOST -> REMOVED``, with ``LOST -> TRACKED`` when the
    second association pass recovers a track.
    """

    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


@dataclass(eq=False)
class Track:
    """A single tracked object.

    ``last_*`` fields cache the most recent matched detection so the
    visualizer and CSV writer never have to reach into history.

    ``eq=False`` is deliberate: a generated ``__eq__`` would compare the
    numpy ``tlbr`` field element-wise, and ``bool()`` on that array raises
    "truth value of an array is ambiguous". Identity comparison is also what
    we actually want when de-duplicating track lists.
    """

    track_id: int
    kalman_filter: "KalmanFilterXYAH"
    tlbr: np.ndarray
    score: float = 0.0
    class_id: int = -1
    class_name: str = "unknown"

    state: TrackState = TrackState.NEW
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    start_frame: int = 0
    frame_id: int = 0

    last_detection: Optional[Detection] = None
    last_score: float = 0.0
    last_class_id: int = -1
    last_class_name: str = "unknown"

    def __post_init__(self) -> None:
        """Coerce the box to a float32 ``(4,)`` array."""
        self.tlbr = np.asarray(self.tlbr, dtype=np.float32).reshape(4)

    @property
    def is_activated(self) -> bool:
        """True for a confirmed track that was matched on this very frame.

        A brand-new track is already active: single-frame flicker is suppressed
        by ``new_track_thresh`` rather than by requiring two hits, which would
        instead delay every ID by one frame.
        """
        return self.state == TrackState.TRACKED and self.time_since_update == 0

    @property
    def tlwh(self) -> np.ndarray:
        """Center-x, center-y, width, height."""
        x1, y1, x2, y2 = self.tlbr
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], dtype=np.float32)

    def mark_lost(self) -> None:
        """Move the track to ``LOST`` and count a frame without a match."""
        self.state = TrackState.LOST
        self.time_since_update += 1

    def mark_removed(self) -> None:
        """Move the track to ``REMOVED``; it is dropped on the next update."""
        self.state = TrackState.REMOVED

    def update(self, detection: Detection, frame_id: int) -> None:
        """Absorb a matched detection and correct the Kalman state."""
        self.frame_id = frame_id
        self.age += 1
        self.time_since_update = 0
        self.hits += 1
        self.state = TrackState.TRACKED
        self.last_detection = detection
        self.last_score = detection.score
        self.last_class_id = detection.class_id
        self.last_class_name = detection.class_name
        self.score = detection.score
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        mean, _ = self.kalman_filter.update(detection.xyah)
        self.tlbr = _xyah_to_tlbr(mean[:4])

    def predict(self) -> np.ndarray:
        """Advance the state one frame and return the predicted ``tlbr``."""
        mean, _ = self.kalman_filter.predict()
        self.age += 1
        self.time_since_update += 1
        self.tlbr = _xyah_to_tlbr(mean[:4])
        return self.tlbr

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        """Compact single-line representation without numpy array noise."""
        x1, y1, x2, y2 = self.tlbr
        return (
            f"Track(id={self.track_id} {self.last_class_name} "
            f"hits={self.hits} age={self.age} state={self.state.name} "
            f"[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}])"
        )


def _xyah_to_tlbr(xyah: Sequence[float]) -> np.ndarray:
    """Convert center/size ``(cx, cy, a, h)`` to ``(x1, y1, x2, y2)``."""
    cx, cy, a, h = xyah[:4]
    w = max(a * h, 1e-6)
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)


class KalmanFilterXYAH:
    """Constant-velocity Kalman filter over ``(cx, cy, aspect, height)``.

    The 8-dimensional state is ``[cx, cy, a, h, vx, vy, va, vh]``: the first
    four are the measurement, the last four the per-frame velocity. Aspect
    ratio and height are tracked instead of width because they stay closer to
    linear as an object approaches or recedes from the camera.

    Process noise is parameterised by measurement standard deviations rather
    than a raw covariance, so the values remain meaningful across image scales.
    Position is allowed to move roughly 20x more freely than velocity, which
    stops the filter chasing pixel noise while still following real motion.
    """

    NDIM = 4

    def __init__(
        self,
        std_weight_position: float = 1.0 / 20,
        std_weight_velocity: float = 1.0 / 160,
    ) -> None:
        """Set noise weights and allocate the motion/update matrices (see class docstring)."""
        self._std_pos = std_weight_position
        self._std_vel = std_weight_velocity

        ndim, dt = self.NDIM, 1.0
        # F: identity, with the velocity columns feeding the position rows.
        self._motion_mat = np.eye(2 * ndim, dtype=np.float32)
        for i in range(ndim):
            self._motion_mat[ndim + i, i] = dt
        # H: selects the first four state components.
        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float32)
        self._mean: Optional[np.ndarray] = None
        self._covariance: Optional[np.ndarray] = None

    # -- construction ------------------------------------------------------ #
    def initiate(self, measurement: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Seed the filter from a first measurement.

        Position variance scales with the measurement magnitude, so a large
        object starts less confident than a small one. Velocity variance starts
        comparatively large because we have no motion information yet.
        """
        m = np.asarray(measurement, dtype=np.float32)
        mean = np.r_[m, np.zeros(self.NDIM, dtype=np.float32)]
        std = [
            2 * self._std_pos * m[0],
            2 * self._std_pos * m[1],
            1e-2,
            2 * self._std_pos * m[2],
            10 * self._std_vel * m[0],
            10 * self._std_vel * m[1],
            1e-5,
            10 * self._std_vel * m[2],
        ]
        covariance = np.diag(np.square(np.asarray(std, dtype=np.float32)))
        self._mean, self._covariance = mean, covariance
        return mean, covariance

    # -- prediction -------------------------------------------------------- #
    def predict(
        self, mean: Optional[np.ndarray] = None, covariance: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Advance one step, returning the new ``(mean, covariance)``.

        Called with no arguments on a freshly initiated filter, then with the
        previous step's output thereafter.
        """
        if mean is None or covariance is None:
            if self._mean is None or self._covariance is None:
                raise RuntimeError("predict() called before initiate()")
            mean, covariance = self._mean, self._covariance

        std = [
            self._std_pos * mean[0],
            self._std_pos * mean[1],
            1e-2,
            self._std_pos * mean[2],
            self._std_vel * mean[0],
            self._std_vel * mean[1],
            1e-5,
            self._std_vel * mean[2],
        ]
        motion_cov = np.diag(np.square(np.asarray(std, dtype=np.float32)))

        mean = (self._motion_mat @ mean.reshape(-1, 1)).ravel()
        covariance = (self._motion_mat @ covariance @ self._motion_mat.T + motion_cov).astype(
            np.float32
        )
        self._mean, self._covariance = mean, covariance
        return mean, covariance

    # -- correction -------------------------------------------------------- #
    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project the state down into the 4-D measurement space."""
        std = [
            self._std_pos * mean[0],
            self._std_pos * mean[1],
            1e-1,
            self._std_pos * mean[2],
        ]
        innovation_cov = np.diag(np.square(np.asarray(std, dtype=np.float32)))
        mean = (self._update_mat @ mean.reshape(-1, 1)).ravel()
        covariance = self._update_mat @ covariance @ self._update_mat.T
        return mean, covariance + innovation_cov

    def update(self, measurement: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Fold in a measurement and correct the state.

        The Kalman gain is obtained by Cholesky factorisation, which is both
        faster and more numerically stable than an explicit inverse. If the
        projected covariance is not positive-definite (which can happen for
        degenerate zero-area boxes) we fall back to the pseudo-inverse.
        """
        if self._mean is None or self._covariance is None:
            raise RuntimeError("update() called before initiate()")

        projected_mean, projected_cov = self.project(self._mean, self._covariance)
        try:
            gain = cho_solve(
                cho_factor(projected_cov, lower=True, check_finite=False),
                (self._covariance @ self._update_mat.T).T,
                check_finite=False,
            ).T
        except np.linalg.LinAlgError:
            LOGGER.debug("projected covariance not PD; using pseudo-inverse")
            gain = (self._covariance @ self._update_mat.T) @ np.linalg.pinv(projected_cov)

        innovation = np.asarray(measurement, dtype=np.float32) - projected_mean
        new_mean = self._mean + (gain @ innovation.reshape(-1, 1)).ravel()
        new_covariance = (self._covariance - gain @ projected_cov @ gain.T).astype(np.float32)

        self._mean, self._covariance = new_mean, new_covariance
        return new_mean, new_covariance

    # -- gating ------------------------------------------------------------ #
    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
        metric: str = "maha",
    ) -> np.ndarray:
        """Distance from the predicted mean to each measurement.

        Args:
            only_position: Compare ``(cx, cy)`` only.
            metric: ``"maha"`` for Mahalanobis distance (scale-aware, the
                default) or ``"gaussian"`` for squared Euclidean.

        Returns:
            ``(N,)`` array of distances, gated later against
            :data:`~config.config.CHI2INV95`.
        """
        mean, covariance = self.project(mean, covariance)
        measurements = np.asarray(measurements, dtype=np.float32)
        if measurements.size == 0:
            return np.zeros((0,), dtype=np.float32)

        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        if metric == "maha":
            try:
                chol = np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError:
                chol = np.linalg.cholesky(
                    covariance + np.eye(covariance.shape[0], dtype=np.float32) * 1e-6
                )
            # d is (N, 4); d.T is (4, N); solve returns (4, N); the trailing
            # transpose is what makes the axis=1 sum run over the four state
            # dimensions and yield one distance per measurement. Omitting it
            # silently returns (4,) for a single detection.
            z = solve_triangular(chol, d.T, lower=True, check_finite=False).T
            return np.sum(z * z, axis=1)
        raise ValueError(f"invalid distance metric {metric!r}; use 'maha' or 'gaussian'")


# ---------------------------------------------------------------------- #
# Association
# ---------------------------------------------------------------------- #
def linear_assignment(
    cost_matrix: np.ndarray, max_cost: float
) -> Tuple[Tuple[np.ndarray, np.ndarray], List[int], List[int]]:
    """Optimal one-to-one assignment minimising total cost.

    Pairs costing more than ``max_cost`` are excluded by masking them to
    :data:`BIG_COST` before the solve, rather than being filtered afterwards.
    That keeps the solver globally optimal over the *allowed* pairs instead of
    greedily accepting the cheapest pair first and stranding a better overall
    matching.

    Args:
        cost_matrix: ``(num_rows, num_cols)`` cost matrix.
        max_cost: Pairs above this are rejected.

    Returns:
        ``(matches, unmatched_rows, unmatched_cols)`` where ``matches`` is a
        tuple of index arrays.
    """
    n_rows, n_cols = cost_matrix.shape
    if cost_matrix.size == 0:
        empty = np.empty((0,), dtype=np.int64)
        return (empty, empty), list(range(n_rows)), list(range(n_cols))

    usable = np.isfinite(cost_matrix) & (cost_matrix <= max_cost)
    gated = np.where(usable, cost_matrix, BIG_COST)

    row_ind, col_ind = linear_sum_assignment(gated)
    keep = gated[row_ind, col_ind] < BIG_COST

    matches = (row_ind[keep], col_ind[keep])
    unmatched_rows = np.setdiff1d(np.arange(n_rows), row_ind[keep])
    unmatched_cols = np.setdiff1d(np.arange(n_cols), col_ind[keep])
    return matches, list(unmatched_rows), list(unmatched_cols)


def matching_cascade(
    distance_matrix: np.ndarray,
    max_distance: float = CHI2INV95,
    threshold: float = CHI2INV95,
) -> Tuple[Tuple[np.ndarray, np.ndarray], List[int], List[int]]:
    """Associate rows to columns under a chi-square Mahalanobis gate.

    A pairing is accepted only when its distance is below *both* gates:
    ``threshold`` (9.4877, the 95% chi-square cutoff for 4 degrees of freedom)
    and ``max_distance``. The effective ceiling is therefore
    ``min(max_distance, threshold)``.

    Do not pass a ``max_distance`` of 1.0 here: the distances in this matrix are
    raw Mahalanobis values, not the ``distance / sqrt(CHI2INV95)`` normalisation
    used by the reference ByteTrack implementation. Un-normalised, a ceiling of
    1.0 rejects essentially every pairing and every track is re-created each
    frame. Leave ``max_distance`` at its default for stage 1.

    Note:
        ``threshold`` comes from :data:`config.config.CHI2INV95`. There is no
        ``scipy.stats.chi2inv95``; that name existed only as a local constant in
        the original SORT code. The value equals
        ``scipy.stats.chi2.ppf(0.95, 4) == 9.4877``.
    """
    return linear_assignment(distance_matrix, max_cost=min(max_distance, threshold))


def iou_distance(atlbrs: Sequence[np.ndarray], btlbrs: Sequence[np.ndarray]) -> np.ndarray:
    """Pairwise ``1 - IoU`` between two sets of boxes. Shape ``(M, N)``."""
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)

    a = np.asarray(atlbrs, dtype=np.float32)[:, None, :]  # (M, 1, 4)
    b = np.asarray(btlbrs, dtype=np.float32)[None, :, :]  # (1, N, 4)

    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2])
    y2 = np.minimum(a[..., 3], b[..., 3])

    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return (NO_OVERLAP_IOU - iou).astype(np.float32)


# ---------------------------------------------------------------------- #
# Tracker
# ---------------------------------------------------------------------- #
class BYTETracker:
    """ByteTrack multi-object tracker.

    A track always lives in exactly one of two lists, which removes any
    ambiguity about double-counting a recovered object:

    * ``_active`` - state ``NEW`` or ``TRACKED``
    * ``_lost``   - state ``LOST``, awaiting recovery by stage 2

    Args:
        config: Thresholds from :class:`~config.config.TrackerConfig`.

    Example::

        tracker = BYTETracker(TrackerConfig())
        for frame_id in range(n_frames):
            tracks = tracker.update(detections, frame_id=frame_id)
            for t in tracks:
                if t.is_activated:
                    print(t.track_id, t.last_class_name, t.last_score)
    """

    def __init__(self, config: Optional[TrackerConfig] = None) -> None:
        """Validate config, allocate track pools and scale the lost buffer (see class docstring)."""
        self.config = config or TrackerConfig()
        self.config.validate()
        self._active: List[Track] = []
        self._lost: List[Track] = []
        self._next_id: int = 1
        self._removed_this_frame: List[Track] = []
        # ``track_buffer`` is specified at 30 fps; scale it to the real rate so
        # the buffer means the same number of seconds on a 60 fps stream.
        self._max_time_lost: int = max(
            1, int(self.config.track_buffer * self.config.frame_rate / 30.0)
        )

    # -- introspection ------------------------------------------------------ #
    @property
    def track_count(self) -> int:
        """Number of tracks currently in the active pool."""
        return len(self._active)

    @property
    def lost_count(self) -> int:
        """Number of unmatched tracks still awaiting recovery."""
        return len(self._lost)

    @property
    def next_id(self) -> int:
        """Id that the next newly initiated track will receive."""
        return self._next_id

    def reset(self) -> None:
        """Drop all state. Call between videos, never mid-stream."""
        self._active.clear()
        self._lost.clear()
        self._next_id = 1
        self._removed_this_frame.clear()

    # -- internals ---------------------------------------------------------- #
    def _initiate_track(self, detection: Detection, frame_id: int) -> Track:
        """Create a brand-new track seeded from ``detection``."""
        kf = KalmanFilterXYAH()
        kf.initiate(detection.xyah)
        track = Track(
            track_id=self._next_id,
            kalman_filter=kf,
            tlbr=detection.tlbr,
            score=detection.score,
            class_id=detection.class_id,
            class_name=detection.class_name,
            state=TrackState.TRACKED,
            start_frame=frame_id,
            frame_id=frame_id,
            last_detection=detection,
            last_score=detection.score,
            last_class_id=detection.class_id,
            last_class_name=detection.class_name,
        )
        self._next_id += 1
        return track

    def _mahalanobis_matrix(self, tracks: List[Track], detections: List[Detection]) -> np.ndarray:
        """``(len(tracks), len(detections))`` Mahalanobis distance matrix."""
        if not tracks or not detections:
            return np.zeros((len(tracks), len(detections)), dtype=np.float32)
        measurements = np.asarray([d.xyah for d in detections], dtype=np.float32)
        rows = [
            t.kalman_filter.gating_distance(
                t.kalman_filter._mean, t.kalman_filter._covariance, measurements
            )
            for t in tracks
        ]
        return np.vstack(rows).astype(np.float32)

    # -- main entry point ---------------------------------------------------- #
    def update(self, detections: List[Detection], frame_id: int = 0) -> List[Track]:
        """Advance the tracker by one frame.

        Args:
            detections: This frame's detections, in any order.
            frame_id: Monotonic frame counter, driving the lost-track timeout.

        Returns:
            Confirmed, currently-visible tracks ready for drawing.
        """
        cfg = self.config
        self._removed_this_frame = []

        # ---- 0. Predict ------------------------------------------------- #
        for track in self._active:
            track.predict()

        pool_high = [d for d in detections if d.score >= cfg.track_high_thresh]
        pool_low = [
            d for d in detections if cfg.track_low_thresh <= d.score < cfg.track_high_thresh
        ]

        matched_high: set[int] = set()
        activated: List[Track] = []
        lost: List[Track] = []

        # ---- 1. Stage 1: high-confidence detections vs active tracks ----- #
        if self._active and pool_high:
            # Both gates are the chi-square cutoff: distances here are raw
            # Mahalanobis values, so 1.0 would reject everything.
            cost = self._mahalanobis_matrix(self._active, pool_high)
            matches, u_track, u_det = matching_cascade(
                cost, max_distance=CHI2INV95, threshold=CHI2INV95
            )
            for row, col in zip(matches[0], matches[1]):
                track, det = self._active[row], pool_high[col]
                track.update(det, frame_id)
                activated.append(track)
                matched_high.add(int(col))
            for idx in u_track:
                unmatched = self._active[idx]
                unmatched.mark_lost()
                lost.append(unmatched)
        elif self._active:
            # No confident detections at all: everything becomes a lost track.
            for track in self._active:
                track.mark_lost()
                lost.append(track)

        unmatched_high = [d for i, d in enumerate(pool_high) if i not in matched_high]

        # Stage-1 matched tracks leave the active pool; unmatched ones move over.
        # Identity comparison via id(): Track sets eq=False, and even so two
        # distinct tracks can share identical field values.
        lost_ids = {id(t) for t in lost}
        self._active = [t for t in self._active if id(t) not in lost_ids]
        self._lost.extend(lost)

        # ---- 2. Stage 2: low-confidence detections vs lost tracks -------- #
        # The occlusion-survival step: these boxes were too weak to start a
        # track but are often good enough to re-attach one that stage 1 dropped.
        if self._lost and pool_low:
            cost = iou_distance([t.tlbr for t in self._lost], [d.tlbr for d in pool_low])
            matches, u_lost, u_det = linear_assignment(cost, max_cost=cfg.match_thresh)
            recovered: List[Track] = []
            for row, col in zip(matches[0], matches[1]):
                track, det = self._lost[row], pool_low[col]
                track.update(det, frame_id)
                recovered.append(track)
                activated.append(track)
            recovered_ids = {id(t) for t in recovered}
            self._lost = [t for t in self._lost if id(t) not in recovered_ids]
            self._active.extend(recovered)

        # ---- 3. Retire tracks lost for too long --------------------------- #
        survivors: List[Track] = []
        for track in self._lost:
            if frame_id - track.frame_id > self._max_time_lost:
                track.mark_removed()
                self._removed_this_frame.append(track)
            else:
                survivors.append(track)
        self._lost = survivors

        # ---- 4. Stage 3: spawn new tracks --------------------------------- #
        for det in unmatched_high:
            if det.score < cfg.new_track_thresh:
                continue
            track = self._initiate_track(det, frame_id)
            activated.append(track)
            self._active.append(track)

        return [t for t in activated if t.is_activated]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        """Compact pool summary: active/lost counts and the next id."""
        return (
            f"BYTETracker(active={len(self._active)} lost={len(self._lost)} "
            f"next_id={self._next_id})"
        )
