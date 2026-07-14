"""Per-frame force logging to a local SQLite database.

Records the forces driving LUMPED_SPRING locomotion so a run can be
replayed or analyzed offline. Three physical tables plus one derived view:

  body_forces      one row / frame: summed arm tension + applied drift
  limb_forces      one row / frame / arm: f_attract, f_spring, net, tension
  sucker_positions one row / frame / sucker: (x, y) + parent arm id/index

  sucker_forces    VIEW: joins sucker_positions to limb_forces on
                   (run_id, frame, arm_id), attributing each arm's tension
                   to its suckers. No storage - computed at query time.

Every table carries run_id + frame so multiple runs coexist and each is
independently queryable.

Threading note: all writes go through one connection on the caller's
thread (the sim's main thread in Octopus.move), never inside the threaded
color pass - SQLite connections are not safe to share across threads.

Usage:
    logger = ForceLogger(run_label="attract_repel demo")
    ...
    logger.log_frame(frame_index, octo)
    ...
    logger.close()

The DB path defaults to logs/forces.db (created if absent).
"""
import os
import sqlite3
import time

import numpy as np


DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs", "forces.db",
)


class ForceLogger:
    def __init__(self, db_path: str = DEFAULT_DB_PATH,
                 run_label: str = ""):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        # check_same_thread stays True (default): we only ever touch this
        # connection from the thread that created it.
        self.conn = sqlite3.connect(db_path)
        self._create_schema()
        self.run_id = self._start_run(run_label)

    # ---- schema -------------------------------------------------------
    def _create_schema(self):
        c = self.conn.cursor()
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                label      TEXT,
                started_at REAL
            );

            CREATE TABLE IF NOT EXISTS body_forces (
                run_id     INTEGER,
                frame      INTEGER,
                x          REAL,
                y          REAL,
                force_x    REAL,
                force_y    REAL,
                force_mag  REAL,
                drift_x    REAL,
                drift_y    REAL,
                PRIMARY KEY (run_id, frame)
            );

            CREATE TABLE IF NOT EXISTS limb_forces (
                run_id      INTEGER,
                frame       INTEGER,
                arm_id      INTEGER,
                base_x      REAL,
                base_y      REAL,
                tip_x       REAL,
                tip_y       REAL,
                arm_length  REAL,
                attract_x   REAL,
                attract_y   REAL,
                spring_x    REAL,
                spring_y    REAL,
                net_x       REAL,
                net_y       REAL,
                tension_x   REAL,
                tension_y   REAL,
                tension_mag REAL,
                PRIMARY KEY (run_id, frame, arm_id)
            );

            CREATE TABLE IF NOT EXISTS sucker_positions (
                run_id    INTEGER,
                frame     INTEGER,
                arm_id    INTEGER,
                sucker_ix INTEGER,
                x         REAL,
                y         REAL,
                PRIMARY KEY (run_id, frame, arm_id, sucker_ix)
            );

            -- Derived per-sucker force: each sucker inherits its parent
            -- arm's tension (attributed force). Computed at query time so
            -- the arm force is stored once, not once per sucker.
            CREATE VIEW IF NOT EXISTS sucker_forces AS
                SELECT s.run_id, s.frame, s.arm_id, s.sucker_ix,
                       s.x, s.y,
                       l.tension_x   AS force_x,
                       l.tension_y   AS force_y,
                       l.tension_mag AS force_mag
                FROM sucker_positions s
                JOIN limb_forces l
                  ON  l.run_id = s.run_id
                  AND l.frame  = s.frame
                  AND l.arm_id = s.arm_id;
            """
        )
        self.conn.commit()

    def _start_run(self, label: str) -> int:
        c = self.conn.cursor()
        c.execute("INSERT INTO runs (label, started_at) VALUES (?, ?)",
                  (label, time.time()))
        self.conn.commit()
        return c.lastrowid

    # ---- per-frame write ---------------------------------------------
    def log_frame(self, frame: int, octo) -> None:
        """Write one frame's body/limb/sucker rows. Call on the main thread."""
        c = self.conn.cursor()

        bf = np.asarray(octo.last_body_force, dtype=float)
        bd = np.asarray(octo.last_body_drift, dtype=float)
        c.execute(
            "INSERT OR REPLACE INTO body_forces VALUES (?,?,?,?,?,?,?,?,?)",
            (self.run_id, frame, float(octo.x), float(octo.y),
             float(bf[0]), float(bf[1]), float(np.hypot(bf[0], bf[1])),
             float(bd[0]), float(bd[1])),
        )

        limb_rows = []
        sucker_rows = []
        for arm_id, limb in enumerate(octo.limbs):
            base = limb.center_line[0]
            tip = limb.center_line[-1]
            fa = np.asarray(limb.last_f_attract, dtype=float)
            fs = np.asarray(limb.last_f_spring, dtype=float)
            net = np.asarray(limb.last_net, dtype=float)
            ten = np.asarray(limb.last_tension, dtype=float)
            limb_rows.append((
                self.run_id, frame, arm_id,
                float(base.x), float(base.y), float(tip.x), float(tip.y),
                float(limb.last_arm_length),
                float(fa[0]), float(fa[1]),
                float(fs[0]), float(fs[1]),
                float(net[0]), float(net[1]),
                float(ten[0]), float(ten[1]),
                float(np.hypot(ten[0], ten[1])),
            ))
            for sucker_ix, s in enumerate(limb.suckers):
                sucker_rows.append((
                    self.run_id, frame, arm_id, sucker_ix,
                    float(s.x), float(s.y),
                ))

        c.executemany(
            "INSERT OR REPLACE INTO limb_forces VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", limb_rows)
        c.executemany(
            "INSERT OR REPLACE INTO sucker_positions VALUES (?,?,?,?,?,?)",
            sucker_rows)
        self.conn.commit()

    def close(self) -> None:
        if self.conn is not None:
            self.conn.commit()
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
