"""Tests for the PerfTracker performance-tracking tool (octopus_ai/perf.py)."""
from octopus_ai.perf import PerfTracker


def test_disabled_tracker_is_a_noop():
    p = PerfTracker(enabled=False)
    with p.track("step"):
        pass
    p.end_frame()
    assert p.frames == 0
    assert p.summary() == "(performance tracking disabled)"


def test_enabled_tracker_records_steps_and_frames():
    p = PerfTracker(enabled=True, track_memory=False, label="unit")
    for _ in range(3):
        with p.track("a"):
            sum(range(1000))
        with p.track("b"):
            sum(range(10))
        p.end_frame()

    assert p.frames == 3
    assert p._calls["a"] == 3 and p._calls["b"] == 3
    assert p._total["a"] >= 0.0

    text = p.summary()
    # Report mentions each tracked step, the frame count, and memory.
    assert "a" in text and "b" in text
    assert "frames: 3" in text
    assert "peak RSS" in text


def test_unseen_step_has_no_entry():
    p = PerfTracker(enabled=True, track_memory=False)
    with p.track("only"):
        pass
    assert "only" in p._order
    assert "missing" not in p._order
