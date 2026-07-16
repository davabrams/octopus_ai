"""Per-frame simulation recording to a columnar DuckDB file (record & replay).

One `.duckdb` file per run (D1): `logs/runs/<run_id>.duckdb`. A run is
self-contained and independently queryable - by the playback server, by an
external `duckdb`/pandas session, or by a second run while this one is still
writing (DuckDB is single-writer-per-file, so per-run files keep completed
runs readable during a live simulate).

The recorder is the host-side per-frame sink of the headless runner. It copies
the ForceLogger pattern: an injectable db path (test seam), a JSON config
snapshot on the run row, and the one-thread rule (every write happens on the
sim thread - never inside the threaded color pass).

Per-frame call order, mirroring the sim loop's before/after color seam:

    rec.begin_frame(frame)
    ... (increment_all -> octo.move -> remove_captured_prey) ...
    rec.snapshot_state(octo, ag, surf, captured_this_frame)  # BEFORE find_color
    color_matrix = octo.find_color(surf, mode, model)
    for l, c in zip(octo.limbs, color_matrix): l.force_color(c)
    rec.snapshot_colors(color_matrix)                        # AFTER force_color
    rec.end_frame(wall_ms)

`snapshot_state` captures positions + BEFORE colors + target (surface) colors
and drains each limb's `last_ilqr_meta`/`last_ilqr_history`; `snapshot_colors`
completes the staged sucker rows with AFTER colors so each row is INSERTed once,
complete (no UPDATEs - DuckDB runs those as delete+insert). `end_frame` computes
the color-error / visibility aggregates and buffers the frame; buffers flush
every `flush_every_frames` frames so a crash loses at most that many frames.
"""

import json
import os
from itertools import count

import duckdb
import numpy as np

from octopus_ai.config import (
    ROOT_DIR,
    as_config,
    config_to_flat,
    json_default,
)
from simulator.simutil import MovementMode

SCHEMA_VERSION = 1

DEFAULT_RUNS_DIR = os.path.join(ROOT_DIR, "logs", "runs")

# Each statement executed once, idempotently, at construction. Kept as a list
# because DuckDB's Python client executes a single statement per call.
_DDL = [
    "CREATE TABLE IF NOT EXISTS schema_info (version INTEGER NOT NULL)",
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id          VARCHAR PRIMARY KEY,
        label           VARCHAR,
        started_at      TIMESTAMP DEFAULT current_timestamp,
        finished_at     TIMESTAMP,
        status          VARCHAR NOT NULL DEFAULT 'running',
        frames_recorded INTEGER,
        config_json     VARCHAR NOT NULL,
        x_len INTEGER NOT NULL, y_len INTEGER NOT NULL,
        num_arms INTEGER NOT NULL, limb_rows INTEGER NOT NULL,
        limb_cols INTEGER NOT NULL,
        ilqr_horizon INTEGER NOT NULL, ilqr_max_iters INTEGER NOT NULL,
        has_ilqr_history BOOLEAN NOT NULL DEFAULT FALSE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS surface (
        run_id VARCHAR NOT NULL, y SMALLINT NOT NULL, x SMALLINT NOT NULL,
        r FLOAT NOT NULL, g FLOAT NOT NULL, b FLOAT NOT NULL,
        PRIMARY KEY (run_id, y, x)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS frames (
        run_id VARCHAR NOT NULL, frame INTEGER NOT NULL,
        head_x FLOAT NOT NULL, head_y FLOAT NOT NULL,
        head_theta FLOAT NOT NULL DEFAULT 0,
        body_force_x FLOAT, body_force_y FLOAT,
        body_drift_x FLOAT, body_drift_y FLOAT,
        prey_captured_frame INTEGER NOT NULL,
        prey_captured_total INTEGER NOT NULL,
        visibility_before FLOAT NOT NULL,
        visibility_after  FLOAT NOT NULL,
        wall_ms FLOAT,
        PRIMARY KEY (run_id, frame)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS limb_nodes (
        run_id VARCHAR NOT NULL, frame INTEGER NOT NULL,
        limb_ix TINYINT NOT NULL, node_ix SMALLINT NOT NULL,
        x FLOAT NOT NULL, y FLOAT NOT NULL, t FLOAT NOT NULL,
        PRIMARY KEY (run_id, frame, limb_ix, node_ix)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS suckers (
        run_id VARCHAR NOT NULL, frame INTEGER NOT NULL,
        limb_ix TINYINT NOT NULL, sucker_ix SMALLINT NOT NULL,
        x FLOAT NOT NULL, y FLOAT NOT NULL,
        r_before FLOAT NOT NULL, g_before FLOAT NOT NULL, b_before FLOAT NOT NULL,
        r_after  FLOAT NOT NULL, g_after  FLOAT NOT NULL, b_after  FLOAT NOT NULL,
        r_target FLOAT NOT NULL, g_target FLOAT NOT NULL, b_target FLOAT NOT NULL,
        err_before FLOAT NOT NULL, err_after FLOAT NOT NULL,
        PRIMARY KEY (run_id, frame, limb_ix, sucker_ix)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS agents (
        run_id VARCHAR NOT NULL, frame INTEGER NOT NULL,
        agent_id INTEGER NOT NULL,
        agent_type TINYINT NOT NULL,
        x FLOAT NOT NULL, y FLOAT NOT NULL, t FLOAT NOT NULL,
        vx FLOAT NOT NULL, vy FLOAT NOT NULL, w FLOAT NOT NULL,
        PRIMARY KEY (run_id, frame, agent_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS limb_frames (
        run_id VARCHAR NOT NULL, frame INTEGER NOT NULL, limb_ix TINYINT NOT NULL,
        tension_x FLOAT, tension_y FLOAT, net_x FLOAT, net_y FLOAT,
        arm_length FLOAT,
        PRIMARY KEY (run_id, frame, limb_ix)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS limb_solves (
        run_id VARCHAR NOT NULL, frame INTEGER NOT NULL, limb_ix TINYINT NOT NULL,
        base_x FLOAT NOT NULL, base_y FLOAT NOT NULL,
        target_x FLOAT NOT NULL, target_y FLOAT NOT NULL,
        target_kind VARCHAR NOT NULL,
        threat_x FLOAT, threat_y FLOAT, threat_active BOOLEAN NOT NULL,
        x0 FLOAT[] NOT NULL,
        u_init FLOAT[],
        iterations INTEGER NOT NULL, converged BOOLEAN NOT NULL,
        final_cost DOUBLE NOT NULL,
        PRIMARY KEY (run_id, frame, limb_ix)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ilqr_iters (
        run_id VARCHAR NOT NULL, frame INTEGER NOT NULL, limb_ix TINYINT NOT NULL,
        iter SMALLINT NOT NULL,
        phase VARCHAR NOT NULL,
        accepted BOOLEAN NOT NULL,
        cost DOUBLE NOT NULL, alpha FLOAT, mu DOUBLE NOT NULL, rel_improve DOUBLE,
        x_traj FLOAT[],
        PRIMARY KEY (run_id, frame, limb_ix, iter)
    )
    """,
]

# INSERT templates keyed by table; column order matches the DDL above.
_INSERTS = {
    "surface": "INSERT INTO surface VALUES (?,?,?,?,?,?)",
    "frames": "INSERT INTO frames VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
    "limb_nodes": "INSERT INTO limb_nodes VALUES (?,?,?,?,?,?,?)",
    "suckers": "INSERT INTO suckers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
    "agents": "INSERT INTO agents VALUES (?,?,?,?,?,?,?,?,?,?)",
    "limb_frames": "INSERT INTO limb_frames VALUES (?,?,?,?,?,?,?,?)",
    "limb_solves": "INSERT INTO limb_solves VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
    "ilqr_iters": "INSERT INTO ilqr_iters VALUES (?,?,?,?,?,?,?,?,?,?,?)",
}

_BUFFERED_TABLES = (
    "frames",
    "limb_nodes",
    "suckers",
    "agents",
    "limb_frames",
    "limb_solves",
    "ilqr_iters",
)


def new_run_id() -> str:
    """Sortable, human-readable, collision-safe run id (D2).

    `%Y%m%d_%H%M%S` + a short uuid suffix. Kept import-light; the timestamp
    is read once per call.
    """
    import time
    import uuid

    return time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


class SimRecorder:
    """Buffered per-frame writer for one run's `.duckdb` file.

    All writes happen on the constructing thread. Use as a context manager to
    get the aborted-on-exception guarantee; otherwise call `close()` explicitly
    with the final status.
    """

    def __init__(
        self,
        config,
        run_id: str | None = None,
        db_path: str | None = None,
        run_label: str = "",
        flush_every_frames: int = 10,
    ):
        self.cfg = as_config(config)
        self.run_id = run_id or new_run_id()
        if db_path is None:
            os.makedirs(DEFAULT_RUNS_DIR, exist_ok=True)
            db_path = os.path.join(DEFAULT_RUNS_DIR, f"{self.run_id}.duckdb")
        else:
            parent = os.path.dirname(db_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
        self.db_path = db_path
        self.run_label = run_label
        self.flush_every_frames = max(1, int(flush_every_frames))

        self.conn = duckdb.connect(db_path)
        self._create_schema()
        self._insert_run_row()

        self._agent_id_counter = count()
        self._frames_recorded = 0
        self._buffers = {t: [] for t in _BUFFERED_TABLES}
        self._frames_since_flush = 0
        self._frame = None
        self._stage = None

    # ---- schema / run row -------------------------------------------------
    def _create_schema(self):
        for stmt in _DDL:
            self.conn.execute(stmt)
        # schema_info is a singleton; seed it only on a fresh file.
        n = self.conn.execute("SELECT count(*) FROM schema_info").fetchone()[0]
        if n == 0:
            self.conn.execute("INSERT INTO schema_info VALUES (?)", [SCHEMA_VERSION])

    def _insert_run_row(self):
        cfg = self.cfg
        config_json = json.dumps(
            config_to_flat(cfg), default=json_default, sort_keys=True
        )
        has_ilqr = bool(
            cfg.output.record_ilqr_history
            and cfg.octopus.limb.movement_mode == MovementMode.ILQR
        )
        # ON CONFLICT DO NOTHING: reopening the same file preserves the
        # existing run row rather than clobbering it (idempotent construction).
        self.conn.execute(
            "INSERT INTO runs (run_id, label, config_json, x_len, y_len, "
            "num_arms, limb_rows, limb_cols, ilqr_horizon, ilqr_max_iters, "
            "has_ilqr_history) VALUES (?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT (run_id) DO NOTHING",
            [
                self.run_id,
                self.run_label,
                config_json,
                cfg.world.x_len,
                cfg.world.y_len,
                cfg.octopus.num_arms,
                cfg.octopus.limb.rows,
                cfg.octopus.limb.cols,
                cfg.octopus.limb.ilqr.horizon,
                cfg.octopus.limb.ilqr.max_iters,
                has_ilqr,
            ],
        )

    # ---- surface (once) ---------------------------------------------------
    def record_surface(self, surf) -> None:
        """Persist the full RGB surface grid once (values, not the seed)."""
        grid = np.asarray(surf.grid, dtype=np.float32)  # (y_len, x_len, 3)
        y_len, x_len = grid.shape[0], grid.shape[1]
        rows = []
        for y in range(y_len):
            for x in range(x_len):
                r, g, b = grid[y, x]
                rows.append((self.run_id, y, x, float(r), float(g), float(b)))
        self.conn.execute("BEGIN")
        self.conn.executemany(_INSERTS["surface"], rows)
        self.conn.execute("COMMIT")

    # ---- per-frame --------------------------------------------------------
    def begin_frame(self, frame: int) -> None:
        self._frame = frame
        self._stage = {}

    def snapshot_state(self, octo, ag, surf, captured_this_frame: int) -> None:
        """Capture positions, BEFORE/target colors, agents, iLQR staging.

        Call AFTER remove_captured_prey, BEFORE find_color.
        """
        frame = self._frame
        rid = self.run_id
        st = self._stage

        # Head + body forces.
        st["head"] = (float(octo.x), float(octo.y))
        st["head_theta"] = float(getattr(octo, "theta", 0.0))
        bf = np.asarray(octo.last_body_force, dtype=float)
        bd = np.asarray(octo.last_body_drift, dtype=float)
        st["body_force"] = (float(bf[0]), float(bf[1]))
        st["body_drift"] = (float(bd[0]), float(bd[1]))
        st["prey_frame"] = int(captured_this_frame)
        st["prey_total"] = int(getattr(ag, "prey_captured", 0)) if ag else 0

        # Limb centerline nodes (0 = base .. rows-1 = tip) + per-limb forces.
        limb_node_rows = []
        limb_frame_rows = []
        for limb_ix, limb in enumerate(octo.limbs):
            for node_ix, cp in enumerate(limb.center_line):
                limb_node_rows.append(
                    (
                        rid,
                        frame,
                        limb_ix,
                        node_ix,
                        float(cp.x),
                        float(cp.y),
                        float(cp.t),
                    )
                )
            ten = np.asarray(limb.last_tension, dtype=float)
            net = np.asarray(limb.last_net, dtype=float)
            limb_frame_rows.append(
                (
                    rid,
                    frame,
                    limb_ix,
                    float(ten[0]),
                    float(ten[1]),
                    float(net[0]),
                    float(net[1]),
                    float(limb.last_arm_length),
                )
            )
        st["limb_nodes"] = limb_node_rows
        st["limb_frames"] = limb_frame_rows

        # Sucker positions + BEFORE colors, in (limb, sucker) order so
        # snapshot_colors can zip AFTER colors in the same order.
        suckers = [
            (limb_ix, sucker_ix, s)
            for limb_ix, limb in enumerate(octo.limbs)
            for sucker_ix, s in enumerate(limb.suckers)
        ]
        st["sucker_meta"] = [(li, si) for (li, si, _) in suckers]
        st["sucker_xy"] = np.array(
            [[s.x, s.y] for (_, _, s) in suckers], dtype=np.float64
        )
        st["before"] = np.array(
            [[s.c.r, s.c.g, s.c.b] for (_, _, s) in suckers], dtype=np.float32
        )
        # Target = surface color under each sucker, mirroring the batched
        # find_color path exactly: float64 round (so x.5+eps does not collapse
        # onto the .5 boundary) then clip to grid bounds.
        grid = np.asarray(surf.grid, dtype=np.float32)
        y_len, x_len = grid.shape[0], grid.shape[1]
        xy = st["sucker_xy"]
        ix = np.clip(np.round(xy[:, 0]).astype(int), 0, x_len - 1)
        iy = np.clip(np.round(xy[:, 1]).astype(int), 0, y_len - 1)
        st["target"] = grid[iy, ix].astype(np.float32)  # (N, 3)
        st["after"] = None

        # Agents: stamp a stable recorder id on first sight (the list index is
        # not stable across capture/respawn).
        agent_rows = []
        for agent in getattr(ag, "agents", None) or []:
            if not hasattr(agent, "_rec_id"):
                agent._rec_id = next(self._agent_id_counter)
            atype = agent.agent_type
            atype_val = int(atype.value) if atype is not None else -1
            agent_rows.append(
                (
                    rid,
                    frame,
                    int(agent._rec_id),
                    atype_val,
                    float(agent.x),
                    float(agent.y),
                    float(agent.t),
                    float(agent.vx),
                    float(agent.vy),
                    float(agent.w),
                )
            )
        st["agent_rows"] = agent_rows

        # Drain per-limb iLQR solve metadata/history (rebound every frame).
        solve_rows = []
        iter_rows = []
        for limb_ix, limb in enumerate(octo.limbs):
            meta = getattr(limb, "last_ilqr_meta", None)
            if meta is None:
                continue
            history = getattr(limb, "last_ilqr_history", None)
            threat = meta["threat"]
            solve_rows.append(
                (
                    rid,
                    frame,
                    limb_ix,
                    float(meta["base_xy"][0]),
                    float(meta["base_xy"][1]),
                    float(meta["target"][0]),
                    float(meta["target"][1]),
                    meta["target_kind"],
                    None if threat is None else float(threat[0]),
                    None if threat is None else float(threat[1]),
                    bool(meta["threat_active"]),
                    _as_float_list(meta["x0"]),
                    _as_float_list(meta["u_init"]),
                    int(meta["iterations"]),
                    bool(meta["converged"]),
                    float(meta["final_cost"]),
                )
            )
            for rec in history or []:
                iter_rows.append(
                    (
                        rid,
                        frame,
                        limb_ix,
                        int(rec.iteration),
                        rec.phase,
                        rec.phase in ("init", "accepted"),
                        float(rec.cost),
                        None if rec.alpha is None else float(rec.alpha),
                        float(rec.mu),
                        None if rec.rel_improve is None else float(rec.rel_improve),
                        _as_float_list(rec.x_traj),
                    )
                )
        st["limb_solves"] = solve_rows
        st["ilqr_iters"] = iter_rows

    def snapshot_colors(self, color_matrix) -> None:
        """Complete staged sucker rows with AFTER colors.

        color_matrix is the nested per-limb list of Color objects returned by
        Octopus.find_color, iterated in the same (limb, sucker) order as
        snapshot_state. Call AFTER the force_color loop.
        """
        after = []
        for c_array in color_matrix:
            for c in c_array:
                after.append([c.r, c.g, c.b])
        self._stage["after"] = np.array(after, dtype=np.float32)

    def end_frame(self, wall_ms: float | None = None) -> None:
        """Compute color-error/visibility aggregates and buffer the frame."""
        st = self._stage
        frame = self._frame
        rid = self.run_id

        before = st["before"]
        target = st["target"]
        after = st["after"]
        if after is None:
            raise ValueError("end_frame called before snapshot_colors")

        err_before = np.sum((before - target) ** 2, axis=1)  # (N,) in [0, 3]
        err_after = np.sum((after - target) ** 2, axis=1)
        vis_before = float(err_before.mean()) if len(err_before) else 0.0
        vis_after = float(err_after.mean()) if len(err_after) else 0.0

        head = st["head"]
        bf, bd = st["body_force"], st["body_drift"]
        self._buffers["frames"].append(
            (
                rid,
                frame,
                head[0],
                head[1],
                st["head_theta"],
                bf[0],
                bf[1],
                bd[0],
                bd[1],
                st["prey_frame"],
                st["prey_total"],
                vis_before,
                vis_after,
                None if wall_ms is None else float(wall_ms),
            )
        )

        # Complete sucker rows (one INSERT each, fully populated).
        sucker_rows = []
        for i, (limb_ix, sucker_ix) in enumerate(st["sucker_meta"]):
            xy = st["sucker_xy"][i]
            b = before[i]
            a = after[i]
            t = target[i]
            sucker_rows.append(
                (
                    rid,
                    frame,
                    limb_ix,
                    sucker_ix,
                    float(xy[0]),
                    float(xy[1]),
                    float(b[0]),
                    float(b[1]),
                    float(b[2]),
                    float(a[0]),
                    float(a[1]),
                    float(a[2]),
                    float(t[0]),
                    float(t[1]),
                    float(t[2]),
                    float(err_before[i]),
                    float(err_after[i]),
                )
            )
        self._buffers["suckers"].extend(sucker_rows)

        self._buffers["limb_nodes"].extend(st["limb_nodes"])
        self._buffers["limb_frames"].extend(st["limb_frames"])
        self._buffers["limb_solves"].extend(st["limb_solves"])
        self._buffers["ilqr_iters"].extend(st["ilqr_iters"])

        self._buffers["agents"].extend(st.get("agent_rows", []))

        self._frames_recorded += 1
        self._frames_since_flush += 1
        self._frame = None
        self._stage = None
        if self._frames_since_flush >= self.flush_every_frames:
            self.flush()

    # ---- flush / close ----------------------------------------------------
    def flush(self) -> None:
        self.conn.execute("BEGIN")
        for table in _BUFFERED_TABLES:
            rows = self._buffers[table]
            if rows:
                self.conn.executemany(_INSERTS[table], rows)
                rows.clear()
        self.conn.execute(
            "UPDATE runs SET frames_recorded = ? WHERE run_id = ?",
            [self._frames_recorded, self.run_id],
        )
        self.conn.execute("COMMIT")
        self._frames_since_flush = 0

    def close(self, status: str = "complete") -> None:
        if self.conn is None:
            return
        self.flush()
        self.conn.execute(
            "UPDATE runs SET status = ?, finished_at = current_timestamp "
            "WHERE run_id = ?",
            [status, self.run_id],
        )
        self.conn.close()
        self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.conn is None:
            return False
        self.close("aborted" if exc_type is not None else "complete")
        return False


def _as_float_list(arr):
    """numpy array (any shape) -> flat list[float] for a FLOAT[] column; None
    passes through (a NULL array element, e.g. frame-1 u_init or a rejected
    iteration's x_traj)."""
    if arr is None:
        return None
    return np.asarray(arr, dtype=np.float32).reshape(-1).tolist()
