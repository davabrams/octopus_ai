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
        }

    # ---- frames ----------------------------------------------------------
    def get_frame(self, run_id: str, frame: int, include_ilqr: bool = False) -> dict:
        con = _connect_ro(self._path(run_id))
        try:
            meta = con.execute(
                "SELECT frames_recorded, num_arms, limb_rows, ilqr_horizon FROM runs"
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
        finally:
            con.close()
        return {"run_id": run_id, "frame": frame, "state": state, "ilqr": ilqr}

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
        fr = con.execute(
            "SELECT head_x, head_y, visibility_before, visibility_after, "
            f"prey_captured_total, prey_captured_frame{theta_col} FROM frames "
            "WHERE frame = ?",
            [frame],
        ).fetchone()
        if fr is None:
            raise FrameOutOfRangeError(frame, None)
        head_theta = _r4(fr[6]) if has_theta else 0.0

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

        # Suckers, flat in (limb, sucker) order.
        suck_rows = con.execute(
            "SELECT limb_ix, x, y, r_after, g_after, b_after, r_before, "
            "g_before, b_before, r_target, g_target, b_target FROM suckers "
            "WHERE frame = ? ORDER BY limb_ix, sucker_ix",
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
            }
            for s in suck_rows
        ]

        agent_rows = con.execute(
            "SELECT agent_id, agent_type, x, y, t, vx, vy FROM agents "
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
            }
            for a in agent_rows
        ]

        return {
            "octopus": {
                "head": {"x": _r4(fr[0]), "y": _r4(fr[1]), "theta": head_theta},
                "limbs": limb_list,
                "suckers": suckers,
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
        solves = con.execute(
            "SELECT limb_ix, base_x, base_y, target_x, target_y, "
            "target_kind, threat_x, threat_y, threat_active, iterations, "
            "converged, final_cost FROM limb_solves WHERE frame = ? "
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
            threat = [_r4(s[6]), _r4(s[7])] if s[8] else None
            out.append(
                {
                    "limb": int(limb_ix),
                    "base": {"x": _r4(s[1]), "y": _r4(s[2])},
                    "final": {
                        "cost": _r4(s[11]),
                        "iterations": int(s[9]),
                        "converged": bool(s[10]),
                        "target": [_r4(s[3]), _r4(s[4])],
                        "target_kind": s[5],
                        "threat": threat,
                    },
                    "iterations": iteration_list,
                }
            )
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
