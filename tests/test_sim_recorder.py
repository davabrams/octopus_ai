"""Tests for the DuckDB SimRecorder (record & replay, Phase 2).

Follows the ForceLogger test pattern: a tmp db path, a tiny octopus via
make_config, a fixed construction order, and a hand-driven frame loop. A helper
runs the exact before/after color seam the headless runner will use.
"""

import os
import sys
import tempfile
import unittest
from itertools import pairwise

import duckdb
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helpers import make_config

from octopus_ai.config import config_to_flat
from simulator.agent_generator import AgentGenerator
from simulator.octopus_generator import Octopus
from simulator.sim_recorder import SimRecorder, new_run_id
from simulator.simutil import MLMode, MovementMode
from simulator.surface_generator import RandomSurface


def _tmp_db(name):
    return os.path.join(tempfile.mkdtemp(prefix="sim_recorder_"), name)


def _drive(rec, octo, ag, surf, frames, mode=MLMode.NO_MODEL, model=None, move=True):
    """Run `frames` frames through the recorder's before/after color seam.

    Frame 0 is the initial post-setup state (no move); 1..N step the sim.
    """
    for f in range(frames):
        rec.begin_frame(f)
        if move and f > 0:
            octo.move(ag)
            ag.increment_all(octo)
        captured = ag.remove_captured_prey(octo) if ag is not None else 0
        rec.snapshot_state(octo, ag, surf, captured)
        color_matrix = octo.find_color(surf, mode, model)
        for limb, c_array in zip(octo.limbs, color_matrix, strict=True):
            limb.force_color(c_array)
        rec.snapshot_colors(color_matrix)
        rec.end_frame(wall_ms=1.0)


def _build(seed=0, **overrides):
    """Deterministic tiny world; seed-first so runs are reproducible."""
    cfg = make_config(
        x_len=8,
        y_len=8,
        limb_rows=4,
        limb_cols=2,
        octo_num_arms=3,
        agent_number_of_agents=2,
        **overrides,
    )
    np.random.seed(seed)
    surf = RandomSurface(cfg)
    octo = Octopus(cfg)
    ag = AgentGenerator(cfg)
    ag.generate(2)
    return cfg, surf, octo, ag


class TestConstruction(unittest.TestCase):
    def test_reopen_is_idempotent_and_preserves_run_row(self):
        """(1a) Reopening the same db_path runs DDL idempotently and keeps
        the existing run row."""
        cfg, surf, octo, ag = _build()
        db = _tmp_db("reopen.duckdb")
        rec = SimRecorder(cfg, run_id="r1", db_path=db)
        rec.record_surface(surf)
        _drive(rec, octo, ag, surf, 2)
        rec.close()

        # Reopen with the same run_id + path: must not duplicate or clobber.
        rec2 = SimRecorder(cfg, run_id="r1", db_path=db)
        rec2.close()
        con = duckdb.connect(db, read_only=True)
        self.assertEqual(con.execute("SELECT count(*) FROM runs").fetchone()[0], 1)
        self.assertEqual(con.execute("SELECT count(*) FROM frames").fetchone()[0], 2)
        con.close()

    def test_two_runs_two_files(self):
        """(1b) Distinct run_ids produce independent files, each with only
        its own run_id."""
        cfg, surf, octo, ag = _build()
        db_a = _tmp_db("a.duckdb")
        db_b = _tmp_db("b.duckdb")
        ra = SimRecorder(cfg, run_id="aaa", db_path=db_a)
        ra.record_surface(surf)
        _drive(ra, octo, ag, surf, 1)
        ra.close()

        cfg2, surf2, octo2, ag2 = _build(seed=1)
        rb = SimRecorder(cfg2, run_id="bbb", db_path=db_b)
        rb.record_surface(surf2)
        _drive(rb, octo2, ag2, surf2, 1)
        rb.close()

        for path, rid in ((db_a, "aaa"), (db_b, "bbb")):
            con = duckdb.connect(path, read_only=True)
            ids = [
                r[0]
                for r in con.execute("SELECT DISTINCT run_id FROM frames").fetchall()
            ]
            self.assertEqual(ids, [rid])
            con.close()

    def test_new_run_id_shape(self):
        rid = new_run_id()
        self.assertRegex(rid, r"^\d{8}_\d{6}_[0-9a-f]{6}$")


class TestRowCounts(unittest.TestCase):
    def test_row_count_invariants(self):
        """(2) Row counts over a 3-frame loop."""
        cfg, surf, octo, ag = _build()
        db = _tmp_db("counts.duckdb")
        rec = SimRecorder(cfg, run_id="c", db_path=db)
        rec.record_surface(surf)
        _drive(rec, octo, ag, surf, 3)
        rec.close()

        arms = len(octo.limbs)
        nodes = octo.limbs[0].rows
        suckers = octo.limbs[0].rows * octo.limbs[0].cols
        con = duckdb.connect(db, read_only=True)

        def n(t):
            return con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]

        self.assertEqual(n("frames"), 3)
        self.assertEqual(n("limb_nodes"), 3 * arms * nodes)
        self.assertEqual(n("suckers"), 3 * arms * suckers)
        self.assertEqual(n("surface"), cfg.world.x_len * cfg.world.y_len)
        self.assertEqual(
            con.execute("SELECT frames_recorded FROM runs").fetchone()[0], 3
        )
        self.assertEqual(
            con.execute("SELECT status FROM runs").fetchone()[0], "complete"
        )
        con.close()


class TestColorMath(unittest.TestCase):
    def test_before_after_target_and_visibility(self):
        """(3) Color math and visibility_after ~= octo.visibility(surf)."""
        cfg, surf, octo, ag = _build()
        db = _tmp_db("color.duckdb")
        rec = SimRecorder(cfg, run_id="col", db_path=db)
        rec.record_surface(surf)
        _drive(rec, octo, ag, surf, 3)
        rec.close()

        # After the last frame the octopus colors match the recorded AFTER
        # colors, so octo.visibility (recomputed live) equals the recorded
        # visibility_after for that frame.
        live_vis = float(octo.visibility(surf))
        con = duckdb.connect(db, read_only=True)
        rec_vis = con.execute(
            "SELECT visibility_after FROM frames WHERE frame = 2"
        ).fetchone()[0]
        self.assertAlmostEqual(live_vis, rec_vis, places=4)

        # err_before/after == sum of squared per-channel deltas vs target.
        row = con.execute(
            "SELECT r_before,g_before,b_before,r_after,g_after,b_after,"
            "r_target,g_target,b_target,err_before,err_after FROM suckers "
            "WHERE frame=2 LIMIT 1"
        ).fetchone()
        before = np.array(row[0:3])
        after = np.array(row[3:6])
        target = np.array(row[6:9])
        self.assertAlmostEqual(row[9], float(np.sum((before - target) ** 2)), places=4)
        self.assertAlmostEqual(row[10], float(np.sum((after - target) ** 2)), places=4)
        con.close()

    def test_target_matches_get_surf_color(self):
        """(4) Recorded target == get_surf_color_at_this_sucker for samples."""
        cfg, surf, octo, ag = _build()
        db = _tmp_db("target.duckdb")
        rec = SimRecorder(cfg, run_id="tg", db_path=db)
        rec.record_surface(surf)
        _drive(rec, octo, ag, surf, 1)
        rec.close()

        con = duckdb.connect(db, read_only=True)
        rows = con.execute(
            "SELECT limb_ix,sucker_ix,r_target,g_target,b_target FROM suckers "
            "WHERE frame=0"
        ).fetchall()
        con.close()
        for limb_ix, sucker_ix, rt, gt, bt in rows[::5]:  # sample every 5th
            s = octo.limbs[limb_ix].suckers[sucker_ix]
            truth = s.get_surf_color_at_this_sucker(surf).to_rgb()
            self.assertAlmostEqual(rt, float(truth[0]), places=4)
            self.assertAlmostEqual(gt, float(truth[1]), places=4)
            self.assertAlmostEqual(bt, float(truth[2]), places=4)


class TestAgentIdentity(unittest.TestCase):
    def test_agent_id_stable_and_respawn_gets_new_id(self):
        """(5) Agent id stable across frames; a respawn gets a NEW id."""
        cfg, surf, octo, ag = _build()
        db = _tmp_db("agents.duckdb")
        rec = SimRecorder(cfg, run_id="ag", db_path=db)
        rec.record_surface(surf)

        # Frame 0 + 1 with the original agents.
        for f in range(2):
            rec.begin_frame(f)
            rec.snapshot_state(octo, ag, surf, 0)
            cm = octo.find_color(surf, MLMode.NO_MODEL, None)
            for limb, c in zip(octo.limbs, cm, strict=True):
                limb.force_color(c)
            rec.snapshot_colors(cm)
            rec.end_frame()

        first_ids = {a._rec_id for a in ag.agents}

        # Simulate a respawn: drop one agent, add a brand-new one.
        from simulator.simutil import Agent, AgentType

        ag.agents = [*ag.agents[:1], Agent(x=1.0, y=1.0, agent_type=AgentType.PREY)]
        rec.begin_frame(2)
        rec.snapshot_state(octo, ag, surf, 0)
        cm = octo.find_color(surf, MLMode.NO_MODEL, None)
        for limb, c in zip(octo.limbs, cm, strict=True):
            limb.force_color(c)
        rec.snapshot_colors(cm)
        rec.end_frame()
        rec.close()

        con = duckdb.connect(db, read_only=True)
        # The surviving agent keeps its id across all three frames.
        survivor = ag.agents[0]._rec_id
        cnt = con.execute(
            "SELECT count(*) FROM agents WHERE agent_id = ?", [survivor]
        ).fetchone()[0]
        self.assertEqual(cnt, 3)
        # The respawn produced an id not seen in the first batch.
        new_id = ag.agents[1]._rec_id
        self.assertNotIn(new_id, first_ids)
        con.close()


class TestILQRCapture(unittest.TestCase):
    def _run_ilqr(self, db, record_history):
        cfg = make_config(
            x_len=10,
            y_len=10,
            limb_rows=5,
            limb_cols=2,
            octo_num_arms=2,
            agent_number_of_agents=3,
            record_ilqr_history=record_history,
            octo_movement_mode=MovementMode.ILQR,
            limb_movement_mode=MovementMode.ILQR,
            octo_ilqr_horizon=4,
            octo_ilqr_max_iters=3,
        )
        np.random.seed(0)
        surf = RandomSurface(cfg)
        octo = Octopus(cfg)
        ag = AgentGenerator(cfg)
        ag.generate(3)
        rec = SimRecorder(cfg, run_id="ilqr", db_path=db, flush_every_frames=1)
        rec.record_surface(surf)
        _drive(rec, octo, ag, surf, 3)
        rec.close()
        return cfg

    def test_ilqr_end_to_end(self):
        """(6) iter 0 = 'init', accepted costs non-increasing, x_traj length,
        final_cost == last accepted cost."""
        db = _tmp_db("ilqr_on.duckdb")
        cfg = self._run_ilqr(db, record_history=True)
        horizon = cfg.octopus.limb.ilqr.horizon
        n_free = cfg.octopus.limb.rows - 1
        con = duckdb.connect(db, read_only=True)

        self.assertTrue(con.execute("SELECT has_ilqr_history FROM runs").fetchone()[0])
        # limb_solves: 2 arms x 2 stepped frames (frame 0 has no solve).
        self.assertEqual(
            con.execute("SELECT count(*) FROM limb_solves").fetchone()[0], 4
        )
        # iter 0 rows are all 'init' and accepted.
        init_phases = con.execute(
            "SELECT DISTINCT phase FROM ilqr_iters WHERE iter=0"
        ).fetchall()
        self.assertEqual(init_phases, [("init",)])
        # x_traj length for a carried (non-null) trajectory.
        traj_len = con.execute(
            "SELECT len(x_traj) FROM ilqr_iters WHERE x_traj IS NOT NULL LIMIT 1"
        ).fetchone()[0]
        self.assertEqual(traj_len, (horizon + 1) * 2 * n_free)

        # Accepted costs non-increasing within each solve; final_cost matches
        # the last accepted iteration's cost.
        keys = con.execute("SELECT DISTINCT frame, limb_ix FROM limb_solves").fetchall()
        for frame, limb_ix in keys:
            accepted = con.execute(
                "SELECT cost FROM ilqr_iters WHERE frame=? AND limb_ix=? "
                "AND phase='accepted' ORDER BY iter",
                [frame, limb_ix],
            ).fetchall()
            costs = [c[0] for c in accepted]
            self.assertTrue(
                all(b <= a for a, b in pairwise(costs)),
                f"accepted costs increased: {costs}",
            )
            final = con.execute(
                "SELECT final_cost FROM limb_solves WHERE frame=? AND limb_ix=?",
                [frame, limb_ix],
            ).fetchone()[0]
            self.assertAlmostEqual(costs[-1], final, places=4)
        con.close()

    def test_ilqr_flag_off_no_rows(self):
        """(6) Flag off => ilqr_iters/limb_solves empty."""
        db = _tmp_db("ilqr_off.duckdb")
        self._run_ilqr(db, record_history=False)
        con = duckdb.connect(db, read_only=True)
        self.assertEqual(
            con.execute("SELECT count(*) FROM ilqr_iters").fetchone()[0], 0
        )
        self.assertEqual(
            con.execute("SELECT count(*) FROM limb_solves").fetchone()[0], 0
        )
        self.assertFalse(con.execute("SELECT has_ilqr_history FROM runs").fetchone()[0])
        con.close()


class TestFlushAndCrash(unittest.TestCase):
    def test_partial_rows_visible_without_close(self):
        """(7) flush_every_frames=1: rows visible via a second connection,
        status stays 'running' when close() is never called."""
        cfg, surf, octo, ag = _build()
        db = _tmp_db("crash.duckdb")
        rec = SimRecorder(cfg, run_id="crash", db_path=db, flush_every_frames=1)
        rec.record_surface(surf)
        _drive(rec, octo, ag, surf, 2)
        # No close(): recorder still holds the write connection. A second
        # (read-only) connection cannot open the same live file, so verify the
        # flush landed by reading through the recorder's own connection.
        n = rec.conn.execute("SELECT count(*) FROM frames").fetchone()[0]
        self.assertEqual(n, 2)
        status = rec.conn.execute("SELECT status FROM runs").fetchone()[0]
        self.assertEqual(status, "running")
        rec.conn.close()

    def test_context_manager_aborts_on_exception(self):
        """(7) __exit__ on exception => status 'aborted'."""
        cfg, surf, octo, ag = _build()
        db = _tmp_db("abort.duckdb")
        with (
            self.assertRaises(RuntimeError),
            SimRecorder(cfg, run_id="ab", db_path=db) as rec,
        ):
            rec.record_surface(surf)
            _drive(rec, octo, ag, surf, 1)
            raise RuntimeError("boom")
        con = duckdb.connect(db, read_only=True)
        self.assertEqual(
            con.execute("SELECT status FROM runs").fetchone()[0], "aborted"
        )
        con.close()


class TestConfigAndSurfaceRoundTrip(unittest.TestCase):
    def test_config_and_surface_round_trip(self):
        """(8) config_json keys == config_to_flat keys; surface reassembles."""
        cfg, surf, octo, ag = _build()
        db = _tmp_db("roundtrip.duckdb")
        rec = SimRecorder(cfg, run_id="rt", db_path=db)
        rec.record_surface(surf)
        _drive(rec, octo, ag, surf, 1)
        rec.close()

        con = duckdb.connect(db, read_only=True)
        import json

        cj = con.execute("SELECT config_json FROM runs").fetchone()[0]
        self.assertEqual(set(json.loads(cj).keys()), set(config_to_flat(cfg).keys()))

        # Surface reassembles to surf.grid.
        rows = con.execute("SELECT y, x, r, g, b FROM surface").fetchall()
        con.close()
        rebuilt = np.zeros_like(np.asarray(surf.grid, dtype=np.float32))
        for y, x, r, g, b in rows:
            rebuilt[y, x] = (r, g, b)
        self.assertTrue(np.allclose(rebuilt, surf.grid, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
