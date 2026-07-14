"""
Unit tests for the SQLite force logger.
"""
import os
import sqlite3
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulator.octopus_generator import Octopus
from simulator.agent_generator import AgentGenerator
from simulator.simutil import MovementMode, Agent, AgentType
from simulator.force_logger import ForceLogger
from OctoConfig import GameParameters


class TestForceLogger(unittest.TestCase):
    def setUp(self):
        self.p = GameParameters.copy()
        self.p.update({
            'x_len': 20, 'y_len': 20, 'octo_num_arms': 4, 'limb_rows': 6,
            'limb_cols': 2, 'rand_seed': 3,
            'octo_movement_mode': MovementMode.ATTRACT_REPEL,
            'limb_movement_mode': MovementMode.ATTRACT_REPEL,
            'agent_movement_mode': MovementMode.ATTRACT_REPEL,
        })
        self.db = os.path.join(tempfile.gettempdir(),
                               "_force_logger_test.db")
        if os.path.exists(self.db):
            os.remove(self.db)

    def tearDown(self):
        if os.path.exists(self.db):
            os.remove(self.db)

    def _run(self, frames=5):
        octo = Octopus(self.p)
        ag = AgentGenerator(self.p)
        ag.agents = [Agent(x=octo.x + 4, y=octo.y,
                           agent_type=AgentType.PREY)]
        logger = ForceLogger(db_path=self.db, run_label="test")
        for f in range(frames):
            octo.move(ag)
            logger.log_frame(f, octo)
        logger.close()
        return octo

    def test_row_counts(self):
        frames = 5
        octo = self._run(frames)
        arms = len(octo.limbs)
        suckers_per_arm = octo.limbs[0].rows * octo.limbs[0].cols
        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        self.assertEqual(
            c.execute("SELECT COUNT(*) FROM body_forces").fetchone()[0],
            frames)
        self.assertEqual(
            c.execute("SELECT COUNT(*) FROM limb_forces").fetchone()[0],
            frames * arms)
        self.assertEqual(
            c.execute("SELECT COUNT(*) FROM sucker_positions").fetchone()[0],
            frames * arms * suckers_per_arm)
        conn.close()

    def test_derived_view_attributes_arm_force(self):
        self._run(5)
        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        # every sucker on an arm should carry that arm's tension exactly
        rows = c.execute("""
            SELECT sf.force_x, sf.force_y, lf.tension_x, lf.tension_y
            FROM sucker_forces sf
            JOIN limb_forces lf
              ON lf.run_id=sf.run_id AND lf.frame=sf.frame
             AND lf.arm_id=sf.arm_id
            WHERE sf.frame=4
        """).fetchall()
        self.assertGreater(len(rows), 0)
        for fx, fy, tx, ty in rows:
            self.assertAlmostEqual(fx, tx, places=9)
            self.assertAlmostEqual(fy, ty, places=9)
        conn.close()

    def test_view_row_count_matches_positions(self):
        self._run(5)
        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        n_pos = c.execute(
            "SELECT COUNT(*) FROM sucker_positions").fetchone()[0]
        n_view = c.execute(
            "SELECT COUNT(*) FROM sucker_forces").fetchone()[0]
        self.assertEqual(n_pos, n_view)
        conn.close()

    def test_multiple_runs_isolated(self):
        self._run(3)
        self._run(3)
        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        run_ids = [r[0] for r in
                   c.execute("SELECT run_id FROM runs").fetchall()]
        self.assertEqual(len(run_ids), 2)
        # each run has its own body rows
        for rid in run_ids:
            n = c.execute("SELECT COUNT(*) FROM body_forces WHERE run_id=?",
                          (rid,)).fetchone()[0]
            self.assertEqual(n, 3)
        conn.close()

    def test_body_force_columns_present(self):
        self._run(4)
        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        row = c.execute(
            "SELECT force_x, force_y, force_mag, drift_x, drift_y "
            "FROM body_forces WHERE frame=3").fetchone()
        self.assertIsNotNone(row)
        fx, fy, mag, dx, dy = row
        self.assertAlmostEqual(mag, float(np.hypot(fx, fy)), places=6)
        conn.close()


if __name__ == '__main__':
    unittest.main()
