"""Tests for the headless runner (record & replay, Phase 3).

Covers the shared sim loop, the frame-0 seam, cancel/failure finalization,
determinism, the serialize_state wire shape, and the CLI wiring. The recorder
is stubbed with a FakeRecorder (patched into the module) so the loop can be
observed without touching DuckDB; a couple of tests drive the real SimRecorder
to assert on-disk determinism and iLQR capture.
"""

import os
import sys
import tempfile
import unittest
from dataclasses import replace
from typing import ClassVar

import duckdb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helpers import make_config

import simulator.headless_runner as hr
from simulator.headless_runner import HeadlessRunner
from simulator.simutil import MovementMode


def _cfg(frames=3, **overrides):
    params = dict(
        x_len=8,
        y_len=8,
        limb_rows=4,
        limb_cols=2,
        octo_num_arms=3,
        agent_number_of_agents=2,
        record_run=True,
    )
    params.update(overrides)
    base = make_config(**params)
    return replace(base, run=replace(base.run, num_iterations=frames))


class FakeRecorder:
    """Records call counts + the before/after color seam, no DB."""

    instances: ClassVar = []

    def __init__(self, cfg, run_id=None, db_path=None, run_label=""):
        self.calls = {"surface": 0, "begin": 0, "state": 0, "colors": 0, "end": 0}
        self.frames = []  # frame indices seen by begin_frame
        self.before_snaps = []  # octo sucker colors at snapshot_state
        self.color_matrices = []  # color_matrix at snapshot_colors
        self.closed_status = None
        self.run_id = run_id
        self.db_path = db_path
        FakeRecorder.instances.append(self)

    def record_surface(self, surf):
        self.calls["surface"] += 1

    def begin_frame(self, frame):
        self.calls["begin"] += 1
        self.frames.append(frame)

    def snapshot_state(self, octo, ag, surf, captured):
        self.calls["state"] += 1
        self.before_snaps.append(
            [[(s.c.r, s.c.g, s.c.b) for s in limb.suckers] for limb in octo.limbs]
        )

    def snapshot_colors(self, color_matrix):
        self.calls["colors"] += 1
        self.color_matrices.append(
            [[(c.r, c.g, c.b) for c in row] for row in color_matrix]
        )

    def end_frame(self, wall_ms=None):
        self.calls["end"] += 1

    def close(self, status="complete"):
        self.closed_status = status


class TestLoopWithFakeRecorder(unittest.TestCase):
    def setUp(self):
        FakeRecorder.instances = []
        self._orig = hr.SimRecorder
        hr.SimRecorder = FakeRecorder

    def tearDown(self):
        hr.SimRecorder = self._orig

    def test_call_counts_and_seam(self):
        summary = HeadlessRunner(_cfg(frames=3), run_id="x").run()
        rec = FakeRecorder.instances[0]
        # Frame 0 + 3 stepped frames => 4 of each per-frame call.
        self.assertEqual(rec.frames, [0, 1, 2, 3])
        for k in ("begin", "state", "colors", "end"):
            self.assertEqual(rec.calls[k], 4)
        self.assertEqual(rec.calls["surface"], 1)
        self.assertEqual(rec.closed_status, "complete")
        self.assertEqual(summary.frames_recorded, 4)
        self.assertEqual(summary.status, "complete")

    def test_before_differs_from_after(self):
        """The camouflage step changes colors: before != after on frame 0."""
        HeadlessRunner(_cfg(frames=1), run_id="x").run()
        rec = FakeRecorder.instances[0]
        before = rec.before_snaps[0]
        after = rec.color_matrices[0]
        self.assertNotEqual(before, after)

    def test_cancel_finalizes_recorder(self):
        calls = {"n": 0}

        def stop():
            calls["n"] += 1
            return calls["n"] > 1  # stop before frame 2

        summary = HeadlessRunner(_cfg(frames=10), run_id="x").run(should_stop=stop)
        rec = FakeRecorder.instances[0]
        self.assertEqual(summary.status, "cancelled")
        self.assertEqual(rec.closed_status, "cancelled")
        # Frame 0 + frame 1 recorded, then cancelled before frame 2.
        self.assertEqual(summary.frames_recorded, 2)

    def test_exception_finalizes_recorder_as_failed(self):
        import simulator.octopus_generator as og

        orig = og.Octopus.find_color
        count = {"n": 0}

        def boom(self, *a, **k):
            count["n"] += 1
            if count["n"] > 2:
                raise RuntimeError("boom")
            return orig(self, *a, **k)

        og.Octopus.find_color = boom
        try:
            with self.assertRaises(RuntimeError):
                HeadlessRunner(_cfg(frames=5), run_id="x").run()
        finally:
            og.Octopus.find_color = orig
        rec = FakeRecorder.instances[0]
        self.assertEqual(rec.closed_status, "failed")


class TestValidation(unittest.TestCase):
    def test_non_positive_frames_raises(self):
        for n in (0, -1):
            cfg = make_config()
            cfg = replace(cfg, run=replace(cfg.run, num_iterations=n))
            with self.assertRaises(ValueError):
                HeadlessRunner(cfg)


class TestSerializeStateShape(unittest.TestCase):
    def test_golden_keys(self):
        FakeRecorder.instances = []
        orig = hr.SimRecorder
        hr.SimRecorder = FakeRecorder
        try:
            summary = HeadlessRunner(_cfg(frames=2), run_id="x").run()
        finally:
            hr.SimRecorder = orig
        state = summary.final_state
        self.assertEqual(set(state.keys()), {"octopus", "agents", "metadata"})
        self.assertEqual(set(state["octopus"].keys()),
                         {"head", "limbs", "suckers", "limb_states", "body_state"})
        self.assertEqual(
            set(state["octopus"]["suckers"][0].keys()),
            {"x", "y", "color", "color_before", "target_color", "state"},
        )
        self.assertEqual(
            set(state["agents"][0].keys()),
            {"id", "x", "y", "type", "vx", "vy", "angle", "behavior"},
        )
        self.assertEqual(
            set(state["metadata"].keys()),
            {
                "iteration",
                "visibility_score",
                "visibility_score_before",
                "prey_captured",
                "prey_captured_this_frame",
            },
        )


class TestDeterminismOnDisk(unittest.TestCase):
    def test_two_runs_identical_positions_and_colors(self):
        dbs = []
        for k in range(2):
            db = os.path.join(tempfile.mkdtemp(), f"det{k}.duckdb")
            dbs.append(db)
            cfg = _cfg(frames=3, rand_seed=7)
            HeadlessRunner(cfg, run_id=f"d{k}", db_path=db).run()
        con0 = duckdb.connect(dbs[0], read_only=True)
        con1 = duckdb.connect(dbs[1], read_only=True)
        for table, cols in (
            ("frames", "head_x, head_y, visibility_after"),
            ("suckers", "x, y, r_after, g_after, b_after"),
        ):
            a = con0.execute(f"SELECT {cols} FROM {table} ORDER BY ALL").fetchall()
            b = con1.execute(f"SELECT {cols} FROM {table} ORDER BY ALL").fetchall()
            self.assertEqual(a, b, f"{table} differs across identical runs")
        con0.close()
        con1.close()


class TestILQRSmoke(unittest.TestCase):
    """Tiny ILQR config so the compile stays cheap."""

    def test_recorder_receives_ilqr_history(self):
        db = os.path.join(tempfile.mkdtemp(), "ilqr.duckdb")
        cfg = _cfg(
            frames=2,
            x_len=10,
            y_len=10,
            limb_rows=4,
            octo_num_arms=2,
            agent_number_of_agents=3,
            record_ilqr_history=True,
            octo_movement_mode=MovementMode.ILQR,
            limb_movement_mode=MovementMode.ILQR,
            octo_ilqr_horizon=4,
            octo_ilqr_max_iters=3,
        )
        summary = HeadlessRunner(cfg, run_id="ilqr", db_path=db).run()
        self.assertEqual(summary.status, "complete")
        con = duckdb.connect(db, read_only=True)
        # 2 arms x 2 stepped frames (frame 0 has no solve).
        self.assertEqual(
            con.execute("SELECT count(*) FROM limb_solves").fetchone()[0], 4
        )
        self.assertGreater(
            con.execute("SELECT count(*) FROM ilqr_iters").fetchone()[0], 0
        )
        con.close()


class TestCLI(unittest.TestCase):
    def test_main_wires_frames_to_runner(self):
        """main() parses --frames and drives a runner; stub the runner so no
        heavy 8-arm iLQR sim runs (the real CLI run is the Phase 3
        checkpoint)."""
        captured = {}

        class StubRunner:
            def __init__(self, cfg, label=""):
                captured["frames"] = cfg.run.num_iterations
                self.run_id = "stub"
                self.db_path = None

            def run(self, progress_cb=None, should_stop=None):
                return hr.RunSummary(
                    status="complete", frames_recorded=3, elapsed_s=0.0, final_state={}
                )

        orig = hr.HeadlessRunner
        hr.HeadlessRunner = StubRunner
        try:
            rc = hr.main(["--frames", "2", "--no-ilqr-history"])
        finally:
            hr.HeadlessRunner = orig
        self.assertEqual(rc, 0)
        self.assertEqual(captured["frames"], 2)


if __name__ == "__main__":
    unittest.main()
