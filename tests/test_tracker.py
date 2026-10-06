"""Tests for ByteTrack association: identity stability and track lifecycle."""

from __future__ import annotations

from typing import List

from config import TrackerConfig
from utils import BYTETracker, Detection, Track


def _run_linear(
    tracker: BYTETracker, n_frames: int, step: float = 4.0, start: float = 100.0
) -> List[List[Track]]:
    """Feed a box moving ``step`` px/frame and collect per-frame outputs."""
    outputs = []
    for frame_id in range(n_frames):
        x = start + step * frame_id
        det = Detection(
            tlbr=__import__("numpy").array([x, 100.0, x + 40.0, 180.0], "float32"),
            score=0.9,
            class_id=0,
            class_name="person",
        )
        outputs.append(tracker.update([det], frame_id=frame_id))
    return outputs


class TestIdentityStability:
    """A single moving object must keep one id for its whole lifetime."""

    def test_linear_motion_single_id(self) -> None:
        """The README example: 30 frames, id must stay 1."""
        tracker = BYTETracker(TrackerConfig())
        outputs = _run_linear(tracker, 30)
        ids = [t.track_id for frame in outputs for t in frame]
        assert ids, "tracker produced no tracks at all"
        assert set(ids) == {1}

    def test_track_is_activated_when_matched(self) -> None:
        """Matched tracks are reported as activated on every frame."""
        tracker = BYTETracker(TrackerConfig())
        outputs = _run_linear(tracker, 10)
        assert all(frame and frame[0].is_activated for frame in outputs)

    def test_two_separate_objects_get_distinct_ids(self) -> None:
        """Two boxes moving apart are tracked under two ids."""
        import numpy as np

        tracker = BYTETracker(TrackerConfig())
        seen = set()
        for frame_id in range(20):
            shift = 3.0 * frame_id
            dets = [
                Detection(
                    tlbr=np.array([50 + shift, 50, 90 + shift, 130], np.float32),
                    score=0.9,
                    class_id=0,
                    class_name="person",
                ),
                Detection(
                    tlbr=np.array([400 - shift, 50, 440 - shift, 130], np.float32),
                    score=0.9,
                    class_id=0,
                    class_name="person",
                ),
            ]
            for track in tracker.update(dets, frame_id=frame_id):
                seen.add(track.track_id)
        assert len(seen) == 2


class TestLifecycle:
    """Lost, recovered and removed states behave as documented."""

    def test_recovery_after_short_occlusion(self) -> None:
        """An object vanishing for 3 frames keeps its original id."""
        import numpy as np

        tracker = BYTETracker(TrackerConfig())
        first_id = None
        for frame_id in range(10):
            x = 100.0 + 4.0 * frame_id
            det = Detection(
                tlbr=np.array([x, 100, x + 40, 180], np.float32),
                score=0.9,
                class_id=0,
                class_name="person",
            )
            for track in tracker.update([det], frame_id=frame_id):
                first_id = track.track_id
        assert first_id == 1

        # Frames 10-12: occluded. Frame 13: reappears on the predicted path.
        # ByteTrack re-attaches lost tracks only via stage 2, which works on
        # the low-score pool (track_low_thresh <= score < track_high_thresh).
        for frame_id in range(10, 13):
            assert tracker.update([], frame_id=frame_id) == []
        x = 100.0 + 4.0 * 13
        det = Detection(
            tlbr=np.array([x, 100, x + 40, 180], np.float32),
            score=0.45,
            class_id=0,
            class_name="person",
        )
        recovered = tracker.update([det], frame_id=13)
        assert [t.track_id for t in recovered] == [first_id]

    def test_empty_frames_are_safe(self) -> None:
        """No detections must never raise and never invent tracks."""
        tracker = BYTETracker(TrackerConfig())
        for frame_id in range(5):
            assert tracker.update([], frame_id=frame_id) == []
        assert tracker.track_count == 0

    def test_low_score_detection_does_not_spawn(self) -> None:
        """Scores below ``new_track_thresh`` cannot start a new track."""
        import numpy as np

        tracker = BYTETracker(TrackerConfig())
        det = Detection(
            tlbr=np.array([10, 10, 50, 90], np.float32),
            score=0.3,
            class_id=0,
            class_name="person",
        )
        assert tracker.update([det], frame_id=0) == []
        assert tracker.next_id == 1

    def test_lost_track_removed_after_buffer(self) -> None:
        """A track missing for longer than ``track_buffer`` is dropped."""
        import numpy as np

        cfg = TrackerConfig(track_buffer=5)
        tracker = BYTETracker(cfg)
        det = Detection(
            tlbr=np.array([10, 10, 50, 90], np.float32),
            score=0.9,
            class_id=0,
            class_name="person",
        )
        tracker.update([det], frame_id=0)
        for frame_id in range(1, 40):
            tracker.update([], frame_id=frame_id)
        assert tracker.track_count == 0
        assert tracker.lost_count == 0

    def test_reset_clears_state(self) -> None:
        """``reset()`` returns the tracker to a pristine state."""
        tracker = BYTETracker(TrackerConfig())
        _run_linear(tracker, 5)
        assert tracker.next_id > 1
        tracker.reset()
        assert tracker.next_id == 1
        assert tracker.track_count == 0
        assert tracker.lost_count == 0
