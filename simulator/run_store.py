"""Read-only query layer over recorded runs (record & replay, Phase 4).

One `.duckdb` file per run under `logs/runs/`. Every method opens a fresh
`duckdb.connect(path, read_only=True)` (~1-5 ms) and closes it, so completed
runs stay readable even while another run is being written (per-run files, D1).
Pure sync - the server always calls these via `asyncio.to_thread`.

The store owns two pieces of shared reshaping so no client reimplements them:

- **ilqr trajectory reshape.** `ilqr_iters.x_traj` is a flat
  `(horizon+1)*2*n_free` float array in UNCLAMPED solver space; `get_frame`
  reshapes it into the wire's nested per-step `[x, y]` free-node lists.
- **carry-forward.** A rejected iteration stored its `x_traj` as NULL; the
  store carries the last accepted trajectory forward and flags the iteration
  `accepted: false`, so every client renders a pose for every iteration.

All emitted floats are rounded to 4 decimals (D10).
"""

import glob
import json
import os

import duckdb

from simulator.sim_recorder import DEFAULT_RUNS_DIR


class RunNotFoundError(Exception):
    """No `.duckdb` file for the requested run_id."""


class FrameOutOfRangeError(Exception):
    def __init__(self, frame, max_frame):
        self.frame = frame
        self.max_frame = max_frame
        super().__init__(f"frame {frame} out of range [0, {max_frame}]")


def _r4(v):
    """Round a float to 4 decimals; pass through None."""
    return None if v is None else round(float(v), 4)


def _frames_has_theta(con) -> bool:
    """Whether the frames table carries head_theta (runs recorded before the
    body-rotation change lack it; they read back as theta = 0)."""
    rows = con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'frames' AND column_name = 'head_theta'"
    ).fetchall()
    return bool(rows)


def _suckers_has_state(con) -> bool:
    """Whether the suckers table carries motor_state (runs recorded before the
    motor-state colour-coding lack it; they read back as state = 0 = idle)."""
    rows = con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'suckers' AND column_name = 'motor_state'"
    ).fetchall()
    return bool(rows)


def _col_exists(con, table: str, column: str) -> bool:
    """Whether `table` has `column` (for graceful reads of pre-v3 runs that
    lack the limb/body behavior-policy columns)."""
    rows = con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = ? AND column_name = ?",
        [table, column],
    ).fetchall()
    return bool(rows)


def _table_exists(con, table: str) -> bool:
    """Whether a table exists (runs recorded before it was added won't have
    it - e.g. explore_map on pre-exploration runs)."""
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_name = ?",
        [table],
    ).fetchall()
    return bool(rows)


def _runs_has_explore(con) -> bool:
    """Value of runs.has_explore, or False if the column predates exploration."""
    cols = con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'runs' AND column_name = 'has_explore'"
    ).fetchall()
    if not cols:
        return False
    row = con.execute("SELECT has_explore FROM runs").fetchone()
    return bool(row[0]) if row else False


def _connect_ro(path):
    """Open a run file read-only, replaying a crashed run's WAL if needed.

    A run that died mid-write may have a WAL that only a read-write open can
    replay; DuckDB then refuses the read-only open until it is. Try read-only
    first (the common case), and on failure do a throwaway read-write
    open/close to replay, then reopen read-only.
    """
    try:
        return duckdb.connect(path, read_only=True)
    except Exception:
        rw = duckdb.connect(path, read_only=False)
        rw.close()
        return duckdb.connect(path, read_only=True)


class RunStore:
    def __init__(self, runs_dir: str | None = None):
        self.runs_dir = runs_dir or DEFAULT_RUNS_DIR

    def _path(self, run_id: str) -> str:
        path = os.path.join(self.runs_dir, f"{run_id}.duckdb")
        if not os.path.exists(path):
            raise RunNotFoundError(run_id)
        return path

    # ---- listing ---------------------------------------------------------
    def list_runs(self, active_run_id: str | None = None) -> list:
        """All runs newest-first, EXCLUDING the active run's file.

        The active run holds a read-write connection, so its file can't be
        opened here; the server injects a synthesized row for it. A stale
        `running` file (no active sim) is reported as `aborted` (D17).
        """
        if not os.path.isdir(self.runs_dir):
            return []
        runs = []
        for path in glob.glob(os.path.join(self.runs_dir, "*.duckdb")):
            run_id = os.path.splitext(os.path.basename(path))[0]
            if active_run_id is not None and run_id == active_run_id:
                continue
            try:
                con = _connect_ro(path)
            except Exception:
                continue  # unreadable file - skip rather than crash the list
            try:
                row = con.execute(
                    "SELECT run_id, label, started_at, status, "
                    "frames_recorded, has_ilqr_history, config_json, "
                    "x_len, y_len, num_arms FROM runs"
                ).fetchone()
            finally:
                con.close()
            if row is None:
                continue
            (
                rid,
                label,
                started_at,
                status,
                frames_recorded,
                has_ilqr,
                config_json,
                x_len,
                y_len,
                num_arms,
            ) = row
            # A file left 'running' with no active sim is a crashed run.
            if status == "running":
                status = "aborted"
            flat = json.loads(config_json)
            runs.append(
                {
                    "run_id": rid,
                    "label": label or "",
                    "started_at": str(started_at),
                    "status": status,
                    "frames_recorded": frames_recorded,
                    "has_ilqr_history": bool(has_ilqr),
                    "config_summary": {
                        "x_len": x_len,
                        "y_len": y_len,
                        "octo_num_arms": num_arms,
                        "octo_movement_mode": flat.get("octo_movement_mode"),
                        "inference_mode": flat.get("inference_mode"),
                    },
                }
            )
        runs.sort(key=lambda r: (r["started_at"], r["run_id"]), reverse=True)
        return runs

    def rename_run(self, run_id: str, label: str):
        """Update the label of a completed run."""
        path = self._path(run_id)
        con = duckdb.connect(path, read_only=False)
        try:
            con.execute("UPDATE runs SET label = ?", [label])
        finally:
            con.close()

    # ---- metadata --------------------------------------------------------
    def run_meta(self, run_id: str) -> dict:
        con = _connect_ro(self._path(run_id))
        try:
            row = con.execute(
                "SELECT label, status, frames_recorded, config_json, "
                "x_len, y_len, num_arms, limb_rows, limb_cols, "
                "ilqr_horizon, ilqr_max_iters, has_ilqr_history "
                "FROM runs"
            ).fetchone()
            if row is None:
                raise RunNotFoundError(run_id)
            (
                label,
                status,
                frames_recorded,
                config_json,
                x_len,
                y_len,
                num_arms,
                limb_rows,
                limb_cols,
                horizon,
                max_iters,
                has_ilqr,
            ) = row
            if status == "running":
                status = "aborted"

            # Background grid, sent ONCE (not per frame): [y][x] -> [r,g,b].
            surf_rows = con.execute(
                "SELECT y, x, r, g, b FROM surface ORDER BY y, x"
            ).fetchall()
            background = [[[0.0, 0.0, 0.0] for _ in range(x_len)] for _ in range(y_len)]
            for y, x, r, g, b in surf_rows:
                background[y][x] = [_r4(r), _r4(g), _r4(b)]

            # Per-frame chart series.
            frames = con.execute(
                "SELECT visibility_before, visibility_after, "
                "prey_captured_total FROM frames ORDER BY frame"
            ).fetchall()
            vis_before = [_r4(f[0]) for f in frames]
            vis_after = [_r4(f[1]) for f in frames]
            prey = [int(f[2]) for f in frames]
            mean_cost_rows = con.execute(
                "SELECT frame, avg(final_cost) FROM limb_solves "
                "GROUP BY frame ORDER BY frame"
            ).fetchall()
            mean_final_cost = [_r4(r[1]) for r in mean_cost_rows]
            has_explore = _runs_has_explore(con)
        finally:
            con.close()

        return {
            "run_id": run_id,
            "label": label or "",
            "status": status,
            "frames_recorded": frames_recorded,
            "config": json.loads(config_json),
            "background": background,
            "anatomy": {
                "num_arms": num_arms,
                "limb_rows": limb_rows,
                "limb_cols": limb_cols,
                "n_free": max(limb_rows - 1, 0),
                "ilqr_horizon": horizon,
                "ilqr_max_iters": max_iters,
            },
            "summary": {
                "visibility_before": vis_before,
                "visibility_after": vis_after,
                "prey_captured": prey,
                "mean_final_cost": mean_final_cost,
            },
            "has_ilqr_history": bool(has_ilqr),
            "has_explore": has_explore,
        }

    # ---- frames ----------------------------------------------------------
    def get_frame(self, run_id: str, frame: int, include_ilqr: bool = False,
                  include_explore: bool = False) -> dict:
        con = _connect_ro(self._path(run_id))
        try:
            meta = con.execute(
                "SELECT frames_recorded, num_arms, limb_rows, ilqr_horizon, "
                "x_len, y_len FROM runs"
            ).fetchone()
            if meta is None:
                raise RunNotFoundError(run_id)
            frames_recorded = meta[0] or 0
            if frame < 0 or frame >= frames_recorded:
                raise FrameOutOfRangeError(frame, frames_recorded - 1)
            n_free = max(meta[2] - 1, 0)
            state = self._build_state(con, run_id, frame,
                                      _frames_has_theta(con))
            ilqr = (
                self._build_ilqr(con, run_id, frame, meta[3], n_free)
                if include_ilqr
                else []
            )
            explore = (self._build_explore(con, frame, meta[4], meta[5])
                       if include_explore else None)
        finally:
            con.close()
        return {"run_id": run_id, "frame": frame, "state": state,
                "ilqr": ilqr, "explore": explore}

    def _build_explore(self, con, frame, x_len, y_len):
        """The per-frame sucker visit-count grid as [y][x], or None if the run
        has no explore data (pre-exploration runs / exploration disabled)."""
        if not _table_exists(con, "explore_map"):
            return None
        row = con.execute(
            "SELECT counts FROM explore_map WHERE frame = ?", [frame]
        ).fetchone()
        if row is None or row[0] is None:
            return None
        flat = row[0]
        return [[_r4(flat[y * x_len + x]) for x in range(x_len)]
                for y in range(y_len)]

    def get_frames(self, run_id: str, start: int, count: int) -> dict:
        con = _connect_ro(self._path(run_id))
        try:
            meta = con.execute("SELECT frames_recorded FROM runs").fetchone()
            if meta is None:
                raise RunNotFoundError(run_id)
            frames_recorded = meta[0] or 0
            if start < 0 or start >= max(frames_recorded, 1):
                raise FrameOutOfRangeError(start, frames_recorded - 1)
            end = min(start + count, frames_recorded)
            has_theta = _frames_has_theta(con)
            states = [self._build_state(con, run_id, f, has_theta)
                      for f in range(start, end)]
        finally:
            con.close()
        return {"run_id": run_id, "start": start, "frames": states}

    # ---- builders --------------------------------------------------------
    def _build_state(self, con, run_id, frame, has_theta=True) -> dict:
        theta_col = ", head_theta" if has_theta else ""
        has_body_state = _col_exists(con, "frames", "body_state")
        bstate_col = ", body_state" if has_body_state else ""
        fr = con.execute(
            "SELECT head_x, head_y, visibility_before, visibility_after, "
            f"prey_captured_total, prey_captured_frame{theta_col}{bstate_col} "
            "FROM frames WHERE frame = ?",
            [frame],
        ).fetchone()
        if fr is None:
            raise FrameOutOfRangeError(frame, None)
        head_theta = _r4(fr[6]) if has_theta else 0.0
        # body_state is the last selected column when present (after the optional
        # theta col), so its index shifts with has_theta.
        body_state = int(fr[7 if has_theta else 6]) if has_body_state else 0

        # Limb centerlines, grouped by limb in node order.
        node_rows = con.execute(
            "SELECT limb_ix, node_ix, x, y FROM limb_nodes WHERE frame = ? "
            "ORDER BY limb_ix, node_ix",
            [frame],
        ).fetchall()
        limbs = {}
        for limb_ix, _node_ix, x, y in node_rows:
            limbs.setdefault(limb_ix, []).append({"x": _r4(x), "y": _r4(y)})
        limb_list = [limbs[k] for k in sorted(limbs)]

        # Per-limb behavior policy (pre-v3 runs lack the column -> all 0/idle).
        limb_states = []
        if _col_exists(con, "limb_frames", "motor_state"):
            ls_rows = con.execute(
                "SELECT motor_state FROM limb_frames WHERE frame = ? "
                "ORDER BY limb_ix",
                [frame],
            ).fetchall()
            limb_states = [int(r[0]) for r in ls_rows]

        # Suckers, flat in (limb, sucker) order.
        has_state = _suckers_has_state(con)
        state_col = ", motor_state" if has_state else ""
        suck_rows = con.execute(
            "SELECT limb_ix, x, y, r_after, g_after, b_after, r_before, "
            f"g_before, b_before, r_target, g_target, b_target{state_col} "
            "FROM suckers WHERE frame = ? ORDER BY limb_ix, sucker_ix",
            [frame],
        ).fetchall()
        suckers = [
            {
                "limb": int(s[0]),
                "x": _r4(s[1]),
                "y": _r4(s[2]),
                "color": [_r4(s[3]), _r4(s[4]), _r4(s[5])],
                "color_before": [_r4(s[6]), _r4(s[7]), _r4(s[8])],
                "target_color": [_r4(s[9]), _r4(s[10]), _r4(s[11])],
                "state": int(s[12]) if has_state else 0,
            }
            for s in suck_rows
        ]

        has_beh = _col_exists(con, "agents", "behavior")
        beh_col = ", behavior" if has_beh else ""
        agent_rows = con.execute(
            f"SELECT agent_id, agent_type, x, y, t, vx, vy{beh_col} FROM agents "
            "WHERE frame = ? ORDER BY agent_id",
            [frame],
        ).fetchall()
        agents = [
            {
                "id": int(a[0]),
                "x": _r4(a[2]),
                "y": _r4(a[3]),
                "type": "prey" if a[1] == 0 else "predator",
                "vx": _r4(a[5]),
                "vy": _r4(a[6]),
                "angle": _r4(a[4]),
                # Per-agent policy (pre-v5 runs lack it -> 0/idle).
                "behavior": int(a[7]) if has_beh else 0,
            }
            for a in agent_rows
        ]

        return {
            "octopus": {
                "head": {"x": _r4(fr[0]), "y": _r4(fr[1]), "theta": head_theta},
                "limbs": limb_list,
                "suckers": suckers,
                "limb_states": limb_states,
                "body_state": body_state,
            },
            "agents": agents,
            "metadata": {
                "iteration": frame,
                "visibility_score": _r4(fr[3]),
                "visibility_score_before": _r4(fr[2]),
                "prey_captured": int(fr[4]),
                "prey_captured_this_frame": int(fr[5]),
            },
        }

    def _build_ilqr(self, con, run_id, frame, horizon, n_free) -> list:
        has_ec = _col_exists(con, "limb_solves", "explore_cell")
        ec_col = ", explore_cell" if has_ec else ""
        solves = con.execute(
            "SELECT limb_ix, base_x, base_y, attract_tgt, attract_sw, "
            "repel_tgt, repel_sw, threat_active, iterations, "
            f"converged, final_cost{ec_col} FROM limb_solves WHERE frame = ? "
            "ORDER BY limb_ix",
            [frame],
        ).fetchall()
        out = []
        for s in solves:
            limb_ix = s[0]
            iters = con.execute(
                "SELECT iter, phase, accepted, cost, x_traj FROM ilqr_iters "
                "WHERE frame = ? AND limb_ix = ? ORDER BY iter",
                [frame, limb_ix],
            ).fetchall()
            iteration_list = []
            last_traj = None
            for it in iters:
                _iter, phase, accepted, cost, x_traj = it
                if x_traj is not None:
                    last_traj = _reshape_traj(x_traj, horizon, n_free)
                iteration_list.append(
                    {
                        "iter": int(_iter),
                        "phase": phase,
                        "accepted": bool(accepted),
                        "cost": _r4(cost),
                        "trajectory": last_traj,  # carry-forward on rejects
                    }
                )
            out.append(
                {
                    "limb": int(limb_ix),
                    "base": {"x": _r4(s[1]), "y": _r4(s[2])},
                    "final": {
                        "cost": _r4(s[10]),
                        "iterations": int(s[8]),
                        "converged": bool(s[9]),
                        # Per-node sensing: each free node's attract/repel target
                        # ([x, y] pairs) and its weight (0 = sensed nothing).
                        "attract_tgt": _pairs(s[3]),
                        "attract_sw": [_r4(v) for v in s[4]],
                        "repel_tgt": _pairs(s[5]),
                        "repel_sw": [_r4(v) for v in s[6]],
                        "threat_active": bool(s[7]),
                        # Per-node chosen frontier cell (None where not exploring
                        # or on pre-v4 runs) - the real explore target for the
                        # analyzer's hover box.
                        "explore_cell": (_pairs_nullable(s[11])
                                         if has_ec and len(s) > 11 else []),
                    },
                    "iterations": iteration_list,
                }
            )
        return out


def _pairs(flat) -> list:
    """Flat [x0,y0,x1,y1,...] -> [[x0,y0],[x1,y1],...] (per-node [x, y])."""
    if flat is None:
        return []
    return [[_r4(flat[i]), _r4(flat[i + 1])] for i in range(0, len(flat), 2)]


def _pairs_nullable(flat) -> list:
    """Like _pairs but a NaN pair (a node that wasn't exploring) -> None, so the
    result is JSON-safe (NaN is not valid JSON) and the client can skip it."""
    if flat is None:
        return []
    out = []
    for i in range(0, len(flat), 2):
        x, y = flat[i], flat[i + 1]
        # DuckDB reads a stored NaN back as either NaN or NULL(None); treat both
        # as "no cell" so the client falls back cleanly.
        blank = x is None or y is None or x != x or y != y  # x!=x: NaN
        out.append(None if blank else [_r4(x), _r4(y)])
    return out


def _reshape_traj(flat, horizon, n_free) -> list:
    """Flat (horizon+1)*2*n_free -> nested [[[x, y], ...n_free], ...horizon+1].

    Free-node positions only (unclamped solver space); the client prepends the
    base point to each step's chain.
    """
    steps = horizon + 1
    traj = []
    k = 0
    for _t in range(steps):
        node_list = []
        for _n in range(n_free):
            node_list.append([_r4(flat[k]), _r4(flat[k + 1])])
            k += 2
        traj.append(node_list)
    return traj
