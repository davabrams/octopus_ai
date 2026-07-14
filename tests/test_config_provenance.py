"""
Tests for config provenance: force data joined to the config that made it.
"""
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from OctoConfig import DEFAULT, to_game_parameters
from simulator.agent_generator import AgentGenerator
from simulator.force_logger import ForceLogger
from simulator.octopus_generator import Octopus
from simulator.simutil import Agent, AgentType, MovementMode


class TestConfigProvenance(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.gettempdir(), "_prov_test.db")
        if os.path.exists(self.db):
            os.remove(self.db)

    def tearDown(self):
        if os.path.exists(self.db):
            os.remove(self.db)

    def _cfg(self, stiffness=0.5):
        return replace(
            DEFAULT,
            world=replace(DEFAULT.world, x_len=20, y_len=20),
            octopus=replace(
                DEFAULT.octopus, num_arms=4,
                movement_mode=MovementMode.LUMPED_SPRING,
                limb=replace(
                    DEFAULT.octopus.limb, rows=6,
                    movement_mode=MovementMode.LUMPED_SPRING,
                    lumped=replace(DEFAULT.octopus.limb.lumped,
                                   arm_stiffness=stiffness),
                ),
            ),
        )

    def _run(self, cfg, label, frames=3):
        octo = Octopus(cfg)
        ag = AgentGenerator(cfg)
        ag.agents = [Agent(x=octo.x + 3, y=octo.y,
                           agent_type=AgentType.PREY)]
        lg = ForceLogger(db_path=self.db, run_label=label, config=cfg)
        for f in range(frames):
            octo.move(ag)
            lg.log_frame(f, octo)
        lg.close()

    def test_config_snapshot_is_stored(self):
        self._run(self._cfg(), "run")
        conn = sqlite3.connect(self.db)
        raw = conn.execute("SELECT config_json FROM runs").fetchone()[0]
        conn.close()
        self.assertIsNotNone(raw)
        snap = json.loads(raw)
        self.assertEqual(len(snap), len(to_game_parameters(DEFAULT)))

    def test_snapshot_records_the_actual_values_used(self):
        self._run(self._cfg(stiffness=4.0), "stiff")
        conn = sqlite3.connect(self.db)
        val = conn.execute(
            "SELECT json_extract(config_json, '$.octo_arm_stiffness') "
            "FROM runs").fetchone()[0]
        conn.close()
        self.assertEqual(val, 4.0)

    def test_enums_serialize_by_name(self):
        self._run(self._cfg(), "run")
        conn = sqlite3.connect(self.db)
        val = conn.execute(
            "SELECT json_extract(config_json, '$.octo_movement_mode') "
            "FROM runs").fetchone()[0]
        conn.close()
        self.assertEqual(val, "LUMPED_SPRING")

    def test_force_data_joins_to_its_config(self):
        """The point of the exercise: correlate a parameter with its effect
        in SQL. Two runs differing only in arm_stiffness."""
        self._run(self._cfg(stiffness=0.5), "soft")
        self._run(self._cfg(stiffness=4.0), "stiff")

        conn = sqlite3.connect(self.db)
        rows = conn.execute("""
            SELECT json_extract(r.config_json, '$.octo_arm_stiffness'),
                   AVG(b.force_mag)
            FROM runs r
            JOIN body_forces b ON b.run_id = r.run_id
            GROUP BY r.run_id
            ORDER BY r.run_id
        """).fetchall()
        conn.close()

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0], 0.5)
        self.assertEqual(rows[1][0], 4.0)
        # stiffer arms transmit more force to the body
        self.assertGreater(rows[1][1], rows[0][1])

    def test_config_is_optional(self):
        """Logging without a config must still work (column just NULL)."""
        cfg = self._cfg()
        octo = Octopus(cfg)
        lg = ForceLogger(db_path=self.db, run_label="no config")
        lg.log_frame(0, octo)
        lg.close()
        conn = sqlite3.connect(self.db)
        raw = conn.execute("SELECT config_json FROM runs").fetchone()[0]
        conn.close()
        self.assertIsNone(raw)

    def test_old_db_without_the_column_is_migrated(self):
        """CREATE TABLE IF NOT EXISTS won't alter an existing table, so a
        pre-snapshot forces.db would silently lack config_json."""
        conn = sqlite3.connect(self.db)
        conn.execute(
            "CREATE TABLE runs (run_id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " label TEXT, started_at REAL)")  # the old schema
        conn.commit()
        conn.close()

        lg = ForceLogger(db_path=self.db, run_label="after",
                         config=self._cfg())
        lg.close()

        conn = sqlite3.connect(self.db)
        cols = [r[1] for r in
                conn.execute("PRAGMA table_info(runs)").fetchall()]
        raw = conn.execute("SELECT config_json FROM runs").fetchone()[0]
        conn.close()
        self.assertIn("config_json", cols)
        self.assertIsNotNone(raw)


if __name__ == '__main__':
    unittest.main()
