# Record & Replay Plan

Headless simulation recording + playback analyzer for the octopus simulator.
This plan was produced by a multi-agent research pass over the codebase
(7 subsystem readers + 4 independent designers, adversarially reviewed) and is
written to be implemented phase by phase. Line references are against the repo
as of commit `cb3a4c8` (2026-07-16).

---

## 1. Requirements

From the user, verbatim intent:

1. **Simulate 120 frames without visualizing them** — headless, no matplotlib.
2. **Log everything for offline analysis and playback:**
   - position of everything (agents, suckers, limbs, head),
   - position of limbs **at each iLQR iteration**, so each iteration can be
     viewed at each frame,
   - iLQR costs,
   - color error **before and after** the camouflage step.
3. **Storage in a columnar/embedded database**, not JSON files ("json is very
   verbose and file heavy, and not in a database") → **DuckDB**.
4. **Full planned horizon** recorded per iLQR iteration (not just the applied
   pose).
5. **The playback analyzer replaces the current visualizer.** Two modes:
   - **Simulate** button — server runs frame-by-frame headless, returns final
     state for display;
   - **Playback** button — full analysis: pick a saved run, scrub frames, and
     scrub iLQR iterations within a frame.
6. **Each simulation saved under a unique timestamped run id.**

Agreed process checkpoint: build the **data pipeline first** (Phases 0–3),
validate a real recorded run, then the server (Phase 4) and frontend (Phase 5).

### Non-goals

- No live free-running streaming mode in the new analyzer (the old
  play/pause/config-sidebar workflow is superseded; see §4 decision D8).
- No per-iteration `u_traj` recording (x_traj + per-solve `u_init` suffice; see
  D6).
- No cross-machine reproducibility guarantee (config snapshot is the flat dict,
  which is deliberately lossy; `background_image` is a local path; numpy global
  RNG seeding depends on construction order — recorded, caveated, not solved).
- No JS build toolchain — the analyzer stays a single self-contained HTML file.

---

## 2. Current state (what exists / what's missing)

**Exists and is reused:**

- The canonical sim-loop order in `visualizer/octo_viz.py:103-141`:
  `ag.increment_all(octo)` → `octo.move(ag)` → `ag.remove_captured_prey(octo)`
  → `color_matrix = octo.find_color(...)` → per-limb `force_color` →
  `octo.visibility(surf)`. This explicit find/force sequence (not `set_color`)
  is the only one exposing the **before/after color seam**.
- `websocket_server.py:152-214 get_simulation_state()` — the per-frame JSON
  shape (head/limbs/suckers/agents/metadata) the analyzer's draw code already
  understands; becomes `serialize_state()` in the headless runner.
- `ForceLogger` (`simulator/force_logger.py`) — the recorder pattern to copy:
  injectable db path (test seam), config snapshot as
  `json.dumps(config_to_flat(cfg))`, one-thread DB rule.
- The old page's canvas draw routine (`octopus-visualizer.html:315-431`) —
  including the **load-bearing half-cell entity shift**
  (`ctx.translate(cellSize/2, cellSize/2)`, fixed in commit `089a7ff`).

**Missing (the new work):**

- Per-iteration iLQR history — `solver.py`'s loop returns only the final
  `ILQRResult`. The loop is **eager Python** (only the six kernels — rollout,
  total_cost, dyn_jac, quad_run, quad_term, forward — are `@tf.function`), so
  opt-in capture is zero-cost when off.
- Any columnar storage, headless entry point, playback protocol, or analyzer
  UI.

---

## 3. Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │ Browser: visualizer/analyzer.html           │
                    │  [Simulate]  [Playback]                     │
                    └───────────────┬─────────────────────────────┘
                                    │ ws://localhost:8765  (v2 protocol)
                    ┌───────────────▼─────────────────────────────┐
                    │ visualizer/websocket_server.py  (rewritten) │
                    │  asyncio event loop — never runs the sim    │
                    │   simulate ──► asyncio.to_thread ──┐        │
                    │   list_runs/load_run/get_frame ──► │        │
                    │        asyncio.to_thread(RunStore) │        │
                    └────────────────────────────────────┼────────┘
                          worker thread                  │
        ┌────────────────────────────────────────────────▼──────┐
        │ simulator/headless_runner.py  HeadlessRunner.run()    │
        │   builds surface/octopus/agents/model, steps N frames │
        │   progress_cb ──► loop.call_soon_threadsafe(queue)    │
        └───────────────┬───────────────────────────────────────┘
                        │ SimRecorder (simulator/sim_recorder.py)
                        ▼
          logs/runs/<run_id>.duckdb      ◄── simulator/run_store.py
          (one file per run, columnar)       (read-only queries)
```

The browser never touches DuckDB; the server queries it and streams JSON. The
CLI (`python simulator/headless_runner.py`) drives the **same**
`HeadlessRunner` — one loop implementation, two drivers.

### Key design decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **One DuckDB file per run**: `logs/runs/<run_id>.duckdb` | Completed runs stay readable (server playback AND external `duckdb`/pandas) while a simulate is writing another file; deleting a run is `rm`; maps 1:1 to "unique timestamp per simulation". DuckDB is single-writer-per-file — a shared file would lock all playback during a simulate. Cross-run SQL uses `ATTACH`. |
| D2 | `run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]` | Sortable, human-readable, collision-safe. |
| D3 | **iLQR iteration 0 = the warm-start rollout** (pre-loop plan); solver iterations are 1..N; invariant `len(history) == iterations + 1` | The analyzer must show what the solve refined *from*. Failed iterations (Cholesky / line-search) get status-only entries with `x_traj=None` — otherwise the scrubber shows unexplained duplicates. |
| D4 | Recorded trajectories are **unclamped solver space**; the applied pose is stored separately in `limb_nodes` | `_move_ilqr` clamps `x_traj[1]` to world bounds (`octopus_generator.py:588-596`). Both are recorded; the UI labels plan (ghost) vs applied (solid). |
| D5 | `record_history` is a **per-call kwarg** on `solve()`, not a `make_solver`/dataclass field | `make_solver` returns a closure (no instance); one compiled controller serves both modes; zero retrace risk since no `@tf.function` signature changes. |
| D6 | No per-iteration `u_traj`; per-solve `u_init` IS recorded | u_traj would ~2× trajectory storage; warm-start coupling means a solve is only reproducible with its `u_init`, so that is kept (nullable, ~300 floats/solve). |
| D7 | v2 protocol **replaces** v1; old `config_update`/`simulation_control` return `error code="gone"` | The analyzer supersedes live streaming. Keeping a second live stepping path means two sim drivers and shared-mutable-objects across threads. Old clients fail loudly, not silently. |
| D8 | Old HTML page + its TSX twin are **deleted** in the same commit that lands the analyzer | Requirement 5. `/octopus-visualizer.html` remains a URL alias serving the new page. |
| D9 | Frontend keeps the existing single-file React 18 UMD + Babel + Tailwind CDN stack | Continuity, zero build step. Known cost: needs internet. Vendoring is a follow-up. |
| D10 | Wire floats rounded to 4 decimals server-side | Halves frame payloads (~50 KB vs ~100 KB). |
| D11 | Frame 0 = initial post-setup state, recorded through the **same explicit `find_color`/`force_color` seam** as frames 1..N (same inference mode); no movement, no iLQR data | Gives the scrubber a true "before anything moved" anchor with before/after colors consistent with every other frame. Deliberate deviation: the existing entry points run the initial pass as bare `set_color(surf)` (NO_MODEL heuristic even when a model is loaded). `frames_recorded == num_frames + 1`. |
| D12 | `EXPECTED_FLAT_KEYS` bumped **once**, 58 → 60, when both flags land in Phase 0 | Avoids cross-branch breakage between the two new config keys. |
| D13 | `HeadlessRunner.run()` calls **`np.random.seed(cfg.run.rand_seed)` explicitly** before constructing anything | The two existing entry points disagree on construction order (octo_viz: AgentGenerator→Octopus→Surface; websocket_server: Surface→Octopus→AgentGenerator) and the seed fires mid-construction inside `AgentGenerator.__init__` — a surface built before it draws from unseeded RNG state. Explicit seeding is a deliberate deviation that makes recorded runs reproducible regardless of construction order. |
| D14 | **Enums serialize as `.name`** in `config_json` and every wire config payload; `merge_flat_overrides` coerces by the baseline value's type **including Enum** (member lookup by name; unknown member ⇒ `bad_request`) | `config_to_flat` holds raw Enum objects (`MovementMode`, `MLMode`, ...): bare `json.dumps` raises TypeError (ForceLogger only works via its `_json_default`), and the old coercion loop only handles bool/int/float — a browser override `{"octo_movement_mode": "ILQR"}` would silently produce a Config holding the *string* `"ILQR"`. |
| D15 | `simulate_complete.final_state` is **single-sourced**: the handler reads the last frame back through `RunStore.get_frame` after the run file closes. `RunSummary.final_state` (live serialize_state) is CLI-only | Live float64 values vs DuckDB FLOAT (float32) round-trips diverge at the 4-decimal rounding boundary — an exact-equality contract between two float paths is flaky by construction. |
| D16 | Naming: **`num_frames`** = requested stepped frames (120) in `simulate`/`simulate_progress`; **`frames_recorded`** = recorded rows incl. frame 0 (121) in the `runs` table, `runs_list`, `run_meta`, `simulate_complete`. Frame index range is `[0, frames_recorded - 1]` | The off-by-one homonym is exactly the seam where a timeline slider and a progress bar diverge. |
| D17 | Run/status vocabulary is **one set everywhere**: `running \| complete \| cancelled \| aborted \| failed` (DB column, `RunSummary.status`, every protocol message; RunStore passes DB values through untranslated) | 'complete' vs 'completed' drift between run-list and simulate-complete would break frontend comparisons. |

---

## 4. Phases

Dependency order: **P0 → P1 → P2 → P3 (checkpoint) → P4 → P5 → P6**, strictly:
P2's recorder drains the `limb.last_ilqr_meta` / `last_ilqr_history` attributes
that P1 creates, and P2's iLQR-capture test cannot run without P1. (The drain
uses `getattr(limb, "last_ilqr_meta", None)` regardless, so the recorder
degrades gracefully for non-ILQR runs.) **Every phase boundary must leave
`bazel build //...` and the test suite green.**

---

### Phase 0 — Dependency + config plumbing

**Goal:** everything later phases need from config/deps, landed in one commit.

1. `pyproject.toml` `[project].dependencies` += `"duckdb>=1.0"` (runtime dep;
   there is no requirements.txt and none should be created). Reinstall:
   `pip install -e ".[dev]"`. Bazel needs nothing (ambient interpreter).
2. `octopus_ai/config_schema.py` — `OutputConfig` gains:
   ```python
   record_run: bool = False           # write a DuckDB run file (SimRecorder)
   record_ilqr_history: bool = False  # capture per-iteration iLQR history on
                                      # each limb; off = zero overhead
   ```
   Both default False per OutputConfig's contract ("a fresh checkout should not
   write databases or videos unprompted"). TEST profile constructs a fresh
   `OutputConfig()` → tests stay side-effect free automatically.
3. `octopus_ai/config.py` — register both keys in `config_to_flat` (bare names,
   matching `log_forces`/`track_performance` convention) and
   `config_from_flat`'s OutputConfig block. Add a **`RECORD` profile**:
   ```python
   RECORD = replace(VIZ_ILQR, output=replace(
       VIZ_ILQR.output, record_run=True, record_ilqr_history=True,
       show_forces=False))
   ```
   (Note: VIZ_ILQR carries a machine-specific `background_image` path; missing
   files degrade gracefully to random noise — acceptable, caveat documented.)
4. `tests/test_config_schema.py:145` — `EXPECTED_FLAT_KEYS` 58 → **60**.
   (`tests/helpers.VALID_KEYS` derives automatically; `test_config_provenance`
   compares lengths dynamically — no edits.)

**Tests:** config round-trip of both new keys; `make_config(record_run=True)`
works; DEFAULT/TEST profiles keep both False.

**Acceptance:** `python run_tests.py` green; `import duckdb` works in `.venv`.

---

### Phase 1 — iLQR per-iteration instrumentation

**Goal:** opt-in solve history with zero overhead when off.

#### `simulator/ilqr/solver.py`

New NamedTuple above `ILQRResult`:

```python
class ILQRIterationRecord(NamedTuple):
    iteration: int            # 0 = initial warm-start rollout; 1..N = solver iters
    phase: str                # 'init' | 'accepted' | 'cholesky_fail' | 'linesearch_fail'
    cost: float               # total cost in effect AFTER this iteration
    alpha: float | None       # accepted line-search step; None for init/rejected
    mu: float                 # regularization used in this iteration's backward
                              # pass; the iter-0 'init' entry records mu_init
                              # (the value the FIRST backward pass will use)
    rel_improve: float | None # None for init/rejected (loop inits it to 0.0 — recording
                              # that on failure would lie)
    x_traj: np.ndarray | None # float32 (horizon+1, 2*n_free), UNCLAMPED solver space;
                              # None for rejected iterations (no new trajectory exists)
```

`ILQRResult` gains a trailing defaulted field — safe: single keyword-only
construction site (`solver.py:242`), all consumers use attribute access
(grep-verified: `octopus_generator.py:588,599`, `tests/test_ilqr.py`), no
positional unpacking anywhere. **Field must stay last forever.**

```python
class ILQRResult(NamedTuple):
    x_traj: tf.Tensor
    u_traj: tf.Tensor
    cost: tf.Tensor
    iterations: int
    converged: bool
    history: list | None = None   # list[ILQRIterationRecord] when recording
```

`solve()` (the closure, `solver.py:148`) gains `record_history: bool = False`.
Capture points, all guarded by `if record_history:` and all **before** any
`mu` mutation (so `mu` reads as the value used in that backward pass):

- after the initial rollout (`solver.py:154-155`): append the `'init'` entry
  (`x_traj.numpy()`, `float(cost)`, `alpha=None`);
- in the `if not backward_ok:` branch (`:209`): `'cholesky_fail'`, `x_traj=None`;
- in the `if not improved:` branch (`:231`): `'linesearch_fail'`, `x_traj=None`;
- after line-search acceptance, before `mu = max(...)` (`:236-237`):
  `'accepted'` with `x_traj.numpy()`, `alpha`, `float(rel_improve)`.

Invariant: **`len(history) == iterations + 1` exactly** (the `mu_max` break
paths append before breaking).

Zero-cost-when-off: the loop is eager Python; `record_history` never reaches
any `@tf.function` kernel, so retracing is structurally excluded. All
`.numpy()`/`float()` device→host transfers sit strictly inside the guard. When
off the added work is one kwarg default + ≤4 falsy checks per iteration.

The six compiled kernels are closure-locals inside `make_solver` and otherwise
unreachable from tests — before returning, attach them as a test-only handle:
`solve._kernels = (rollout, total_cost, dyn_jac, quad_run, quad_term, forward)`
(plain attribute on the closure; no `@tf.function` signature change), so tests
can assert `experimental_get_tracing_count()` is unchanged by a recorded solve.

#### `simulator/ilqr/arm.py`

`ArmController.solve(..., record_history: bool = False)` → pass-through to
`self._solve(x0, params, u_init, record_history=record_history)`. No new
dataclass field (a per-controller flag would bake the mode into the compiled
controller).

#### `simulator/octopus_generator.py`

`Limb.__init__` (next to the existing iLQR slots at `:169-173`):

```python
self.record_ilqr_history = cfg.output.record_ilqr_history
self.last_ilqr_history = None   # list[ILQRIterationRecord] | None, per frame
self.last_ilqr_meta = None      # dict | None, per-solve metadata
```

`_move_ilqr`: snapshot `u_init = self._ilqr_u` **before** the solve (the
`np.roll` afterward builds a new array, so the reference is stable), pass
`record_history=self.record_ilqr_history`, and after the warm-start shift
stash:

```python
if self.record_ilqr_history:
    self.last_ilqr_history = res.history
    self.last_ilqr_meta = {
        "base_xy": (float(x_octo), float(y_octo)),
        "target": (float(solve_target[0]), float(solve_target[1])),
        "target_kind": "prey" if prey is not None else "hold",  # idle frames
                        # hold at the tip; without this every idle frame looks
                        # like the arm "reaching" its own tip
        "threat": None if threat is None else (float(threat[0]), float(threat[1])),
        "threat_active": threat is not None,
        "x0": x0,                     # (2*n_free,) float32
        "u_init": u_init,             # (horizon, 2*n_free) float32 | None (frame 1)
        "iterations": res.iterations,
        "converged": res.converged,
        "final_cost": float(res.cost),
    }
```

History references are rebound every frame — the recorder must drain them
between `octo.move()` and the next frame.

#### Bazel (closes an existing gap — fully this time)

`simulator/ilqr/BUILD` currently has no targets for `solver.py`/`arm.py`, and
`octopus_generator.py:19` does a **module-level**
`from simulator.ilqr.arm import ArmController` — so today no Bazel consumer of
`//simulator:octopus_generator` has `arm.py` in its runfiles at all. Fix all
of it:

- `simulator/ilqr/BUILD`: `py_library(name="solver", srcs=["solver.py"])` and
  `py_library(name="arm", srcs=["arm.py"], deps=[":solver"])` (arm.py imports
  only solver/numpy/tf — no simutil dep).
- `simulator/BUILD`: add `"//simulator/ilqr:arm"` to
  `//simulator:octopus_generator`'s deps (propagates through `:generators`).
- `tests/BUILD`: `py_test(name="test_ilqr", srcs=["helpers.py",
  "test_ilqr.py"], imports=["."], deps=["//octopus_ai:octo_util",
  "//simulator/ilqr:arm", "//simulator/ilqr:solver",
  "//simulator:octopus_generator", "//simulator:simutil"])` — the file is
  currently pytest-only.

#### Tests (extend `tests/test_ilqr.py`; small configs n_free=5, horizon=8)

1. **Off ⇒ identical + None**: same inputs solved with and without the flag
   produce identical `x_traj/u_traj/cost/iterations/converged` (deterministic
   solver, no RNG); `history is None`; kernel tracing counts unchanged.
2. **Length/indices/shapes**: `len(h) == res.iterations + 1`; `h[0].phase ==
   'init'`; iteration numbers are `0..len-1`; accepted/init entries have
   `(horizon+1, 2*n_free)` float32 `x_traj`; accepted `alpha` ∈ alphas tuple.
3. **Accepted costs strictly decreasing** (acceptance requires
   `cost_cand < cost`); last accepted cost == `float(res.cost)`; last accepted
   `x_traj` == `res.x_traj.numpy()`.
4. **Rejected entries carry no trajectory**: solver-level test via
   `make_solver` with identically-zero costs (no improvement possible) and
   `mu_init=1.0, mu_max=4.0` so escalation exhausts in ~2 iterations — every
   non-init entry is `linesearch_fail` with `x_traj/alpha/rel_improve` None and
   unchanged cost; mu non-decreasing.
5. **Limb integration**: `make_config(record_ilqr_history=True,
   octo_movement_mode=ILQR, limb_movement_mode=ILQR, octo_ilqr_horizon=4,
   octo_ilqr_max_iters=3, limb_rows=6)`; after `limb.move(...)`,
   `last_ilqr_history`/`last_ilqr_meta` populated, `target_kind == 'hold'` with
   no agents, first-frame `u_init is None`, second-frame shape correct.
   Counter-test: flag off ⇒ both stay None.

**Acceptance:** all tests green; a manual `ArmController(...).solve(...,
record_history=True)` in a REPL shows a plausible cost-decreasing history.

---

### Phase 2 — SimRecorder (DuckDB storage layer)

**Goal:** `simulator/sim_recorder.py`, writing one `.duckdb` file per run.

#### Schema (DDL executed idempotently at construction; `SCHEMA_VERSION = 1`)

```sql
CREATE TABLE IF NOT EXISTS schema_info (version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS runs (
    run_id          VARCHAR PRIMARY KEY,   -- '20260716_143512_a1b2c3'
    label           VARCHAR,
    started_at      TIMESTAMP DEFAULT current_timestamp,
    finished_at     TIMESTAMP,             -- set by close()
    status          VARCHAR NOT NULL DEFAULT 'running',  -- running|complete|cancelled|aborted|failed (D17)
    frames_recorded INTEGER,               -- count of recorded frames (rows in `frames`,
                                           -- incl. frame 0 = requested+1 when complete);
                                           -- updated at every flush → partial runs scrubbable
    config_json     VARCHAR NOT NULL,      -- json.dumps(config_to_flat(cfg), default=_json_default,
                                           -- sort_keys=True) — the flat dict holds raw Enum objects,
                                           -- bare json.dumps raises TypeError; extract ForceLogger's
                                           -- _json_default (enum → name) into a shared helper (D14)
    -- denormalized dims so consumers avoid JSON parsing:
    x_len INTEGER NOT NULL, y_len INTEGER NOT NULL,
    num_arms INTEGER NOT NULL, limb_rows INTEGER NOT NULL, limb_cols INTEGER NOT NULL,
    ilqr_horizon INTEGER NOT NULL, ilqr_max_iters INTEGER NOT NULL,
    has_ilqr_history BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS surface (      -- actual RGB values (may come from an image)
    run_id VARCHAR NOT NULL, y SMALLINT NOT NULL, x SMALLINT NOT NULL,
    r FLOAT NOT NULL, g FLOAT NOT NULL, b FLOAT NOT NULL,
    PRIMARY KEY (run_id, y, x)
);

CREATE TABLE IF NOT EXISTS frames (
    run_id VARCHAR NOT NULL, frame INTEGER NOT NULL,   -- 0 = initial state
    head_x FLOAT NOT NULL, head_y FLOAT NOT NULL,
    body_force_x FLOAT, body_force_y FLOAT,            -- octo.last_body_force
    body_drift_x FLOAT, body_drift_y FLOAT,            -- octo.last_body_drift
    prey_captured_frame INTEGER NOT NULL,              -- remove_captured_prey() return
    prey_captured_total INTEGER NOT NULL,              -- ag.prey_captured
    visibility_before FLOAT NOT NULL,   -- mean(err_before), range [0,3] — same formula
    visibility_after  FLOAT NOT NULL,   -- as Octopus.visibility, computed from stored colors
    wall_ms FLOAT,
    PRIMARY KEY (run_id, frame)
);

CREATE TABLE IF NOT EXISTS limb_nodes (   -- applied (post-clamp) centerline after octo.move
    run_id VARCHAR NOT NULL, frame INTEGER NOT NULL,
    limb_ix TINYINT NOT NULL, node_ix SMALLINT NOT NULL,  -- 0=base .. rows-1=tip
    x FLOAT NOT NULL, y FLOAT NOT NULL, t FLOAT NOT NULL, -- CenterPoint x/y/theta
    PRIMARY KEY (run_id, frame, limb_ix, node_ix)
);

CREATE TABLE IF NOT EXISTS suckers (
    run_id VARCHAR NOT NULL, frame INTEGER NOT NULL,
    limb_ix TINYINT NOT NULL, sucker_ix SMALLINT NOT NULL, -- index = row + rows*col
    x FLOAT NOT NULL, y FLOAT NOT NULL,                    -- post-move position
    r_before FLOAT NOT NULL, g_before FLOAT NOT NULL, b_before FLOAT NOT NULL,
    r_after  FLOAT NOT NULL, g_after  FLOAT NOT NULL, b_after  FLOAT NOT NULL,
    r_target FLOAT NOT NULL, g_target FLOAT NOT NULL, b_target FLOAT NOT NULL,
    err_before FLOAT NOT NULL,   -- sum of squared per-channel deltas, range [0,3]
    err_after  FLOAT NOT NULL,
    PRIMARY KEY (run_id, frame, limb_ix, sucker_ix)
);

CREATE TABLE IF NOT EXISTS agents (
    run_id VARCHAR NOT NULL, frame INTEGER NOT NULL,
    agent_id INTEGER NOT NULL,    -- recorder-stamped stable id (see below)
    agent_type TINYINT NOT NULL,  -- AgentType.value
    x FLOAT NOT NULL, y FLOAT NOT NULL, t FLOAT NOT NULL,
    vx FLOAT NOT NULL, vy FLOAT NOT NULL, w FLOAT NOT NULL,
    PRIMARY KEY (run_id, frame, agent_id)
);

CREATE TABLE IF NOT EXISTS limb_frames (  -- per-limb force summary, all movement modes
    run_id VARCHAR NOT NULL, frame INTEGER NOT NULL, limb_ix TINYINT NOT NULL,
    tension_x FLOAT, tension_y FLOAT, net_x FLOAT, net_y FLOAT, arm_length FLOAT,
    PRIMARY KEY (run_id, frame, limb_ix)
);

CREATE TABLE IF NOT EXISTS limb_solves (  -- one row per iLQR solve (ILQR mode only)
    run_id VARCHAR NOT NULL, frame INTEGER NOT NULL, limb_ix TINYINT NOT NULL,
    base_x FLOAT NOT NULL, base_y FLOAT NOT NULL,
    target_x FLOAT NOT NULL, target_y FLOAT NOT NULL,
    target_kind VARCHAR NOT NULL,          -- 'prey' | 'hold'
    threat_x FLOAT, threat_y FLOAT, threat_active BOOLEAN NOT NULL,
    x0 FLOAT[] NOT NULL,                   -- (2*n_free,) initial free-node positions
    u_init FLOAT[],                        -- warm-start controls, NULL on frame 1 —
                                           -- required for exact solve reproduction
    iterations INTEGER NOT NULL, converged BOOLEAN NOT NULL, final_cost DOUBLE NOT NULL,
    PRIMARY KEY (run_id, frame, limb_ix)
);

CREATE TABLE IF NOT EXISTS ilqr_iters (   -- one row per iLQR ITERATION
    run_id VARCHAR NOT NULL, frame INTEGER NOT NULL, limb_ix TINYINT NOT NULL,
    iter SMALLINT NOT NULL,               -- 0 = warm-start rollout ('init')
    phase VARCHAR NOT NULL,               -- init|accepted|cholesky_fail|linesearch_fail
    accepted BOOLEAN NOT NULL,            -- phase IN ('init','accepted')
    cost DOUBLE NOT NULL, alpha FLOAT, mu DOUBLE NOT NULL, rel_improve DOUBLE,
    x_traj FLOAT[],                       -- flattened float32 (horizon+1)*2*n_free = 330
                                          -- at defaults; NULL on rejected iterations
                                          -- (readers carry last non-NULL forward);
                                          -- reshape(horizon+1, n_free, 2); UNCLAMPED
    PRIMARY KEY (run_id, frame, limb_ix, iter)
);
```

No extra indexes — every query filters on a PK prefix; DuckDB zone maps cover
frame-range scans.

#### API

```python
DEFAULT_RUNS_DIR = os.path.join(ROOT_DIR, "logs", "runs")  # ROOT_DIR honors
                                                           # BUILD_WORKSPACE_DIRECTORY
def new_run_id() -> str: ...   # strftime + uuid suffix (D2)

class SimRecorder:
    def __init__(self, config, run_id: str | None = None,
                 db_path: str | None = None,        # test seam; default
                 run_label: str = "",               # <runs_dir>/<run_id>.duckdb
                 flush_every_frames: int = 10): ...
    def record_surface(self, surf) -> None: ...     # once, after RandomSurface
    def begin_frame(self, frame: int) -> None: ...
    def snapshot_state(self, octo, ag, surf, captured_this_frame: int) -> None: ...
        # CALL AFTER remove_captured_prey, BEFORE find_color. Captures head,
        # limb_nodes (one float() host-read per field), sucker positions +
        # BEFORE colors, target colors via ONE batched numpy gather over
        # surf.grid (float64 rounding! — float32 collapses x.5+eps onto the .5
        # boundary; mirror Sucker.get_surf_color_at_this_sucker), agents
        # (stamping a._rec_id = next(counter) on first sight — list index is
        # NOT stable across capture/respawn), limb_frames forces, and drains
        # limb.last_ilqr_meta/-history into limb_solves / ilqr_iters staging.
    def snapshot_colors(self, color_matrix) -> None: ...
        # CALL AFTER the force_color loop; completes staged sucker rows with
        # AFTER colors — each sucker row INSERTed once, complete (no UPDATEs,
        # which DuckDB executes as delete+insert).
    def end_frame(self, wall_ms: float | None = None) -> None: ...
        # computes err_before/err_after + visibility_before/after (numpy),
        # moves staging → buffers, flushes every flush_every_frames.
    def flush(self) -> None: ...   # BEGIN; executemany per table;
                                   # UPDATE runs.frames_recorded; COMMIT
    def close(self, status: str = "complete") -> None: ...
    # context manager: __exit__ closes with 'aborted' on exception
```

Batching: **buffered `executemany`** (DuckDB's Python client has no Appender
API; Arrow/pandas would add a dep for no benefit at ~55k rows/run). `FLOAT[]`
params bind as plain Python lists. All writes on the sim thread only
(ForceLogger's rule). Crash bound: ≤ `flush_every_frames` frames lost, run row
stays `status='running'` → listed as `aborted` by the RunStore.

#### Size / overhead budget (120-frame VIZ_ILQR default: 8 arms, 16 rows →
n_free=15, 256 suckers, horizon=10, max_iters=5, 5 agents, 30×30 world)

121 recorded frames (frame 0 + 120 stepped, D11); the solve/iter tables cover
frames 1..120 only (frame 0 has no solves):

| table | rows | est. |
|---|---|---|
| limb_nodes | 128 × 121 = 15,488 | 0.4 MB |
| suckers | 256 × 121 = 30,976 | 2.4 MB |
| limb_solves | 8 × 120 = 960 | 1.3 MB (x0 + u_init) |
| ilqr_iters | ≤ 960 × 6 = 5,760 (init + ≤5) | ≤7.8 MB worst, ~4 MB typical |
| everything else | ~2,600 | <0.1 MB |

**≈ 6–12 MB/run on disk with history; ~2–3 MB without.** Recording overhead
< 5 ms/frame (~500 `float(tf)` host reads + numpy gathers + list appends),
< 2 % of an iLQR frame; exactly zero when `record_run` is off.

#### Tests (`tests/test_sim_recorder.py`, ForceLogger-test pattern: tmp db
path, tiny octopus via `make_config`, fixed construction order)

(1a) constructing SimRecorder twice against the same db_path (reopen) executes
the DDL idempotently and preserves the existing run row; (1b) two recorders
with distinct run_ids produce two independent `.duckdb` files, each containing
only its own run_id; (2) row-count invariants over a 3-frame manual loop;
(3) before/after color math exact vs known surface; `visibility_after` ≈
`octo.visibility(surf)`; (4) target == `get_surf_color_at_this_sucker` for
sampled suckers; (5) agent identity stable across frames; respawn produces a
NEW id (guards list-index keying); (6) iLQR capture end-to-end (small ILQR
config): iter 0 'init', accepted costs non-increasing,
`len(x_traj)==(horizon+1)*2*(rows-1)`, `final_cost` == last accepted cost;
flag off ⇒ `ilqr_iters` empty; (7) flush/crash: `flush_every_frames=1`, no
close ⇒ rows visible via second connection, status 'running'; `__exit__` on
exception ⇒ 'aborted'; (8) config/surface round-trip (`config_json` keys ==
`config_to_flat(cfg)` keys; surface reassembles to `surf.grid`); (9) Bazel
`py_test` + `simulator/BUILD` `py_library :sim_recorder` (+ append to
`//simulator:generators`); tests/README.md line.

**Acceptance:** tests green; a hand-driven 3-frame loop produces a queryable
`.duckdb` file.

---

### Phase 3 — Headless runner + CLI  ★ user checkpoint

**Goal:** `simulator/headless_runner.py` — the ONE headless loop implementation
shared by CLI and server.

```python
class RunSummary(NamedTuple):
    status: str            # complete | cancelled | failed   (D17 vocabulary)
    frames_recorded: int   # num_frames + 1 when complete (frame 0 = initial)
    elapsed_s: float
    final_state: dict      # serialize_state(...) of the last recorded frame.
                           # CLI-only convenience — the server does NOT use it;
                           # simulate_complete.final_state is read back through
                           # RunStore.get_frame (D15)

class HeadlessRunner:
    def __init__(self, cfg, run_id=None, label="", db_path=None): ...
        # stores args only — cheap to construct anywhere; raises ValueError on
        # cfg.run.num_iterations <= 0 (the old -1=infinite convention is
        # rejected for recorded runs)
    def run(self, progress_cb=None, should_stop=None) -> RunSummary: ...
```

`run()` executes entirely on the calling thread:

1. **Seed first, then build** (D13): `np.random.seed(cfg.run.rand_seed)`, then
   `RandomSurface(cfg)` → `Octopus(cfg)` → `AgentGenerator(cfg)` +
   `.generate(...)`. The explicit seed is a deliberate deviation: the two
   existing entry points disagree on construction order (octo_viz.py:58-61
   builds agents first; websocket_server.py:111-114 builds the surface first)
   and `AgentGenerator.__init__` is what seeds the global RNG — without the
   explicit seed, a surface built first is drawn from unseeded RNG state and
   runs are irreproducible. Then the ModelLoader block moved verbatim from
   `websocket_server.py:118-136` (custom_objects =
   ConstraintLoss/ClampedTargetLoss/DeltaColorLayer, fall back to NO_MODEL on
   exception; record the effective mode). Keras load + frame-1 tf.function
   compiles thus happen off the event loop when the server drives this.
2. `SimRecorder` if `cfg.output.record_run` (DuckDB connection born and dying
   on this thread). **Frame 0** uses the same explicit seam as every other
   frame (D11 — `Octopus.set_color` returns None, so it cannot feed
   `snapshot_colors`): `rec.snapshot_state(...)` (colors = 0.5 defaults) →
   `color_matrix = octo.find_color(surf, mode, model)` → per-limb
   `force_color` → `rec.snapshot_colors(color_matrix)` → `rec.end_frame()`.
   No movement, no agents step, no iLQR rows. Note this runs frame 0's
   camouflage with the **configured** inference mode, unlike the old bare
   `set_color(surf)` heuristic-only initial pass.
3. Per frame 1..N: `should_stop()` check → `increment_all` → `octo.move` →
   `remove_captured_prey` → `rec.snapshot_state` → `find_color` →
   per-limb `force_color` → `rec.snapshot_colors` → `rec.end_frame` →
   `progress_cb({frame, num_frames, visibility_score, prey_captured,
   frame_ms, elapsed_s})`.
4. `finally:` recorder closed with the right status; returns `RunSummary`.

`serialize_state(octo, ag, iteration, visibility, ...)` — descendant of
`get_simulation_state` minus background/fps, plus `color_before`,
`visibility_score_before`, stable agent `id`, `vx`+`vy` (the old wire's
`velocity` was only vx), `prey_captured_this_frame`. Single source of the
`state` JSON shape; unit-tested for shape equality against RunStore output.

`main()` — octo_viz.py entry-point pattern (sys.path insert, `CFG = RECORD` at
top, profile imports with `noqa: F401`), argparse `--frames/--label/--no-ilqr-history`,
progress to stdout, **`MPLBACKEND=Agg` set before importing simutil**
(matplotlib imports at simutil module level and can grab a GUI backend).

**Tests** (`tests/test_headless_runner.py`): 3-frame run with a FakeRecorder —
call counts, before≠after colors, after == color_matrix; determinism (two runs
same cfg ⇒ identical recorded positions/colors); `should_stop` after frame 1 ⇒
'cancelled', recorder finalized; runner exception ⇒ 'failed', recorder still
closed; `serialize_state` golden-key shape test; `num_iterations<=0` raises;
slow-marked iLQR smoke (tiny ILQR config ⇒ recorder receives per-limb history).
CLI smoke: `--frames 2` via subprocess. Bazel targets.

**★ CHECKPOINT — deliver to the user:**
`python simulator/headless_runner.py --frames 120` produces
`logs/runs/<run_id>.duckdb` (~6–12 MB). Validate together with a few DuckDB
queries (cost curves, visibility before/after, one frame's iLQR iterations)
before building the server/UI on this schema.

---

### Phase 4 — WebSocket server v2 (simulate + playback)

**Goal:** rewrite `visualizer/websocket_server.py`; add
`simulator/run_store.py`.

#### Execution model (the one hard constraint)

Today `simulation_step` runs synchronously **on the asyncio event-loop
thread** (`websocket_server.py:269-288`) — a 120-frame iLQR run there would
freeze all socket I/O for minutes. Therefore:

- `simulate` → `await asyncio.to_thread(runner.run, progress_cb=...,
  should_stop=active.cancel.is_set)`;
- `progress_cb` (called on the worker thread) does **only**
  `loop.call_soon_threadsafe(active.progress_q.put_nowait, info)`;
- a drain task on the loop broadcasts `simulate_progress`, **coalescing**
  (latest wins) so a slow client can never back-pressure the sim;
- cancel = `threading.Event` polled once per frame;
- **exactly one simulate at a time** (`error code="busy"`); the worker touches
  no server state, no sockets, no `self.clients`;
- playback = `await asyncio.to_thread(RunStore.method, ...)`, each call opening
  `duckdb.connect(path, read_only=True)` (~1-5 ms) — per-run files (D1) mean
  completed runs are always readable, even mid-simulate; the active run is
  readable only via progress until `simulate_complete`;
- server shutdown sets `cancel` and joins with a timeout so the run row
  finalizes.

The old `simulation_loop`/`simulation_step`/`start|stop|reset_simulation`/
`update_config`/`setup_simulation` are **deleted** (D7). `update_config`'s
type-coercion loop is extracted as a pure, unit-testable
`merge_flat_overrides(flat, overrides) -> (merged, ignored_keys)` reused by the
simulate handler — **extended with an Enum branch** (D14): when the baseline
value is an Enum, coerce the override by member-name lookup, `bad_request` on
an unknown member. All outbound config payloads (`server_info.default_config`,
`simulate_started.config`, `run_meta.config`) serialize enums as `.name`.
`handle_message`'s if/elif chain becomes a dispatch dict; unknown types get
`error code="unknown_type"` (no more silent ignore); v1 types get
`code="gone"`.

**The simulate handler must turn recording on** — the VIZ_ILQR baseline has
`record_run=False`, and nothing else would write the run the protocol promises
(`db_path`, playback, unique run id). After merging overrides:

```python
cfg = replace(cfg, run=replace(cfg.run, num_iterations=num_frames),
              output=replace(cfg.output, record_run=True,
                             record_ilqr_history=data.get("record_ilqr_history", True)))
```

The message-level `record_ilqr_history` field **wins** over any flat-key
override of the same name. The handler constructs the runner through a
**factory seam** — `self.runner_factory = HeadlessRunner` (class attribute) —
so protocol tests inject a fake runner without monkeypatching modules.

After the worker completes (run file closed), the handler builds
`simulate_complete.final_state` by reading the last frame back through
`RunStore.get_frame` (D15) — never from `RunSummary.final_state`.

#### `simulator/run_store.py`

Read-only query layer over `logs/runs/*.duckdb`: `list_runs(active_run_id=None)`
(directory scan, newest first; stale `status='running'` files with no active
sim reported as `aborted`), `run_meta(run_id)`, `get_frame(run_id, frame,
include_ilqr)`, `get_frames(run_id, start, count)`. Pure sync; always called
via `asyncio.to_thread`.

**Active-run exclusion:** the recorder holds a read-write connection to the
active run's file, so opening it (even `read_only=True`) fails. The server
passes the active run_id into `list_runs`, which **skips opening that file**
and synthesizes its row from the in-memory `ActiveRun` (status `running`,
current frame). `run_meta`/`get_frame` on the active run return
`error code="run_in_progress"` without touching the file.

RunStore reshapes `ilqr_iters.x_traj` into **nested per-step `[x, y]` node
lists** (the wire's `trajectory` shape — the server owns the reshape, D-note in
Phase 5) and applies the **carry-forward rule** (rejected iterations reuse the
last accepted trajectory, flagged `accepted: false`) so every client renders
correctly without reimplementing it.

#### v2 wire protocol (canonical spec lives in the module docstring, per repo convention)

Envelope both directions: `{"type": str, "req_id": str?, "data": {...}}`.
Every request gets exactly one terminal reply (its response or `error`),
echoing `req_id`. Progress/complete are additionally broadcast. Floats rounded
to 4 decimals (D10).

```jsonc
// on connect
{"type":"server_info","data":{"protocol":2,
  "active_run": null | {"run_id":..., "frame":17, "num_frames":120},
  "default_config": { /* config_to_flat(VIZ_ILQR) */ }}}

// simulate
{"type":"simulate","req_id":"a1","data":{
  "num_frames":120,             // int in [1,10000]; default cfg.run.num_iterations
  "config":{...},               // optional flat-key overrides (merge_flat_overrides)
  "record_ilqr_history":true,   // default true
  "label":"baseline-120"}}
→ {"type":"simulate_started","req_id":"a1","data":{"run_id":...,"num_frames":120,
    "config":{...merged...},"ignored_keys":[...],"db_path":"logs/runs/<id>.duckdb"}}
→ {"type":"simulate_progress","data":{"run_id":...,"frame":17,"num_frames":120,
    "visibility_score":0.041,"prey_captured":1,"elapsed_s":33.2,"frame_ms":812.4}}
→ {"type":"simulate_complete","req_id":"a1","data":{"run_id":...,
    "status":"complete|cancelled|failed","frames_recorded":121,"elapsed_s":241.7,
    "error":null,
    "final_state":{ /* read back via RunStore.get_frame(last) — D15 */ }}}

{"type":"simulate_cancel","req_id":"a2","data":{}}   // → simulate_cancel_ack,
                                                     // then complete(cancelled)
// playback
{"type":"list_runs","req_id":"b1","data":{}}
→ {"type":"runs_list","req_id":"b1","data":{"runs":[{"run_id":...,"label":...,
    "started_at":...,"status":"running|complete|cancelled|aborted|failed",
    "frames_recorded":121,"has_ilqr_history":true,       // D16: recorded rows
    "config_summary":{"x_len":30,"y_len":30,"octo_num_arms":8,
                      "octo_movement_mode":"ILQR","inference_mode":"SUCKER"}}]}}

{"type":"load_run","req_id":"c1","data":{"run_id":...}}
→ {"type":"run_meta","req_id":"c1","data":{"run_id":...,"label":...,"status":...,
    "frames_recorded":121,                    // frame range [0, frames_recorded-1] (D16)
    "config":{...},                           // full recorded flat dict, enums as .name
    "background":[[[r,g,b],...x_len],...y_len],  // sent ONCE, not per frame
    "anatomy":{"num_arms":8,"limb_rows":16,"limb_cols":2,"n_free":15,
               "ilqr_horizon":10,"ilqr_max_iters":5},
    "summary":{"visibility_before":[...],"visibility_after":[...],  // per-frame
               "prey_captured":[...],"mean_final_cost":[...]},      // chart series
    "has_ilqr_history":true}}

{"type":"get_frame","req_id":"d7","data":{"run_id":...,"frame":42,
                                          "include_ilqr":true}}   // default false
→ {"type":"frame_data","req_id":"d7","data":{"run_id":...,"frame":42,
  "state":{
    "octopus":{"head":{"x":..,"y":..},
      "limbs":[[{"x","y"}, ...rows], ...num_arms],
      "suckers":[{"x","y","color":[r,g,b],"color_before":[r,g,b],
                  "target_color":[r,g,b]}, ...]},   // flat, limb order
    "agents":[{"id":3,"x","y","type":"prey|predator","vx","vy","angle"}],
    "metadata":{"iteration":42,"visibility_score":0.041,
                "visibility_score_before":0.093,
                "prey_captured":2,"prey_captured_this_frame":0}},
  "ilqr":[  // [] when include_ilqr=false / frame 0 / no history
    {"limb":0,"base":{"x","y"},
     "final":{"cost":12.31,"iterations":3,"converged":true,
              "target":[x,y],"target_kind":"prey|hold","threat":[x,y]|null},
     "iterations":[
       {"iter":0,"phase":"init","accepted":true,"cost":18.02,
        "trajectory":[[[x,y], ...n_free], ...horizon+1]},   // free nodes; client
       ...]}]}}                                             // prepends base

{"type":"get_frames","req_id":"e1","data":{"run_id":...,"start":0,"count":20}}
→ {"type":"frames_data","req_id":"e1","data":{"run_id":...,"start":0,
    "frames":[ /* state objects only, no ilqr */ ]}}       // batch prefetch

// errors — terminal reply for any failed request
{"type":"error","req_id":"d7","data":{
  "code":"busy|not_running|unknown_run|run_in_progress|frame_out_of_range|"
         "bad_request|sim_failed|gone|unknown_type",
  "message":"...","detail":{}}}
```

HTML serving in Phase 4: **unchanged** — keep serving
`octopus-visualizer.html` (it becomes a dead v1 client, which D7 accepts).
`analyzer.html` does not exist yet; flipping `HTML_PAGE` or the BUILD `data`
entry here would break `bazel build //visualizer:websocket_server` for the
whole P4→P5 window. The flip is Phase 5's atomic commit.

#### Tests (socket-free: call `await server.handle_message(fake_ws, msg)`
directly with a FakeWebSocket; async wrapped in `asyncio.run`)

`tests/test_websocket_protocol.py`: merge_flat_overrides coercion/drop **plus
enum round-trip** (`{"octo_movement_mode": "ILQR"}` → MovementMode.ILQR;
unknown member ⇒ `bad_request`); simulate happy path with a fake runner
injected via `server.runner_factory` (started → ≥1 progress → complete,
`_active` cleared, merged config has `record_run=True`); busy rule; cancel
path; progress coalescing (100 cb calls ⇒ far fewer broadcasts, last wins);
v1 tombstones ⇒ `gone`; unknown type; bad `num_frames` (0, -1, "abc") ⇒
`bad_request`; req_id echoed everywhere; `server_info` on connect (JSON-
serializable, enums as names); playback error codes (unknown_run,
frame_out_of_range with `detail.max_frame`, run_in_progress);
**list_runs during a stubbed in-flight simulate includes the active run
without opening its file**; process_request paths.
`tests/test_record_playback.py` (integration, tmp runs dir): simulate 2 frames
end-to-end with the real runner+recorder, then list_runs/load_run/get_frame
round-trip; assert `get_frame(last).state == simulate_complete.final_state` —
trivially exact since both come from RunStore (D15); it guards the read-back
path, not float fidelity. If a test binds port 8765 it needs Bazel
`tags=["local"]` (inference_server precedent) — but the design keeps all
protocol tests socket-free. Split the server into an importable form
(`server_lib` pattern from inference_server/BUILD) for Bazel testability.

**Acceptance:** protocol tests green; manual `websocat`/browser-console
session: simulate 5 frames, list, load, scrub.

---

### Phase 5 — Analyzer frontend (`visualizer/analyzer.html`)

**Goal:** the replacement page. Single self-contained file, React 18 UMD +
Babel + Tailwind CDN (D9), `ReactDOM.createRoot` (not the deprecated
`ReactDOM.render`).

**This phase's commit is atomic** (keeps every phase boundary green, per §4):
it adds `analyzer.html`, flips `HTML_PAGE = "analyzer.html"` + the
`process_request` whitelist (`"/", "/index.html", "/analyzer.html",
"/octopus-visualizer.html"` — old path serves the new page as a bookmark
alias), changes `visualizer/BUILD` `data = ["analyzer.html"]`, and **deletes**
`octopus-visualizer.html` + `octopus-ai-visualizer.tsx` (D8).

**Two script blocks:**
- `<script id="analyzer-core">` — **plain JS, no JSX**: `to255`, `asTriple`
  (triple-or-scalar color tolerance), `makeLru(cap)`, `prefetchPlan`,
  `chainsWithBase(trajectory, base)` (the server already sends nested per-step
  `[x, y]` node lists — RunStore owns the reshape; this helper ONLY prepends
  the base point to each step's chain, no pairing), `nearestSucker`,
  `colorErrorStats`, `clampIdx`, `playbackAdvance`. Attached to
  `window.AnalyzerCore` → unit-testable under node without a JS toolchain.
- `<script type="text/babel">` — React UI + canvas.

**Rendering (reuses the old page's draw code, upgraded):**
- `drawWorld(ctx, frame, view)` lifted from old lines 315-431 — RGB background,
  grid, **the half-cell entity shift** (`ctx.translate(cellSize/2, cellSize/2)`
  — do not drop; commit `089a7ff`), limb polylines `#ff4444`, head `#ff6666`,
  suckers r=0.15·cell, agents + optional sensing circles.
- Two-layer canvas: background+grid pre-rendered once per run into an offscreen
  canvas (constant per recorded run), blitted per frame; entities/overlay on
  top. `requestAnimationFrame` + dirty flag reading refs — **frame data must
  live in refs, not React state** (the old page's per-message `setState` churn
  must not be copied).
- `cellSize = clamp(floor(min(availW/x_len, availH/y_len)), 12, 30)` (30×30
  world must fit).
- **Do NOT copy** the old page's reconnect stale-closure bug (rewrite with
  refs) or its fallback random-data generator (an analyzer must never fabricate
  data — show an empty state).

**Layout:** top bar (ws url/connect, Simulate|Playback tabs, run/frame/
visibility/prey readouts) · canvas + right sidebar (simulate controls with
frames input default 120 + progress; run list + config summary; sucker
inspector with before/after/target swatches on hover) · transport bar
(timeline slider, step, play/pause, speed 0.5–4×) · iLQR strip (overlay
toggle, limb chips 0–7|All, **iteration slider**, cost readout) · charts row
(visibility before/after vs frame — click to seek; iLQR cost vs iteration for
the current frame, per-limb lines, log toggle). Hand-rolled inline SVG charts
(~120 points; no chart lib).

**iLQR overlay** (same translate block as entities; coords are world coords —
verified `_move_ilqr` builds x0 from `center_line[i].x/y`):
- ghost horizon poses: `chains[t]` polylines in amber, α fading with t; t=0
  dashed;
- tip path: cyan polyline through `chains[t][last]`, dot at terminal tip;
- applied pose stays solid red on top → plan-vs-applied = ghost-vs-solid;
- convergence mode (key `V`): each iteration's terminal pose color-graded
  red→green;
- iteration slider range = actual recorded count for that frame/limb (solves
  early-terminate); scrubbing iterations is pure client-side redraw (zero
  network).

**Data flow:** LRU frame cache (400 ≈ 20 MB) + LRU iLQR cache (40); on
`run_meta` request frame 0 then background bulk-fill via chunked
`get_frames` while `frames_recorded × ~50 KB ≤ 25 MB` (a 120-frame run is
fully local in <1 s); seek = cache hit (sync draw) or fetch + windowed prefetch
(ahead 15 / behind 3); play maintains ≥30-frame lookahead, pauses on
"buffering" rather than skipping; iLQR fetched lazily (`include_ilqr=true`)
only when the overlay/cost chart is open.

**Simulate completion (requirement 9's display half):** on
`simulate_complete`, the canvas **draws `final_state`** (same shape as
`frame_data.state`, so `drawWorld` consumes it unchanged) and the new run is
**auto-selected in Playback mode** ready to scrub. Manual checklist item #3
verifies this.

**Keyboard:** Space play/pause · ←/→ ∓1 frame · Shift+←/→ ∓10 · Home/End ·
`[`/`]` iLQR iteration · `I` overlay · `L` cycle limb · `A` all limbs · `V`
convergence · `C` sucker color mode (after/before/target/error-heatmap) · `G`
grid · `D` sensing circles · `?` cheat sheet. Suppressed while typing in
inputs.

**Tests:** `tests/test_analyzer_core.py` — regex-extract the plain-JS core
block, run assertions via `node -e` subprocess (`skipif` node missing):
asTriple/to255, LRU eviction, prefetchPlan windows, **chainsWithBase**
(base prepended per step, node lists passed through untouched), nearestSucker,
colorErrorStats, playbackAdvance. Static checks: file contains
`translate(cellSize / 2`, does not contain `ReactDOM.render(`.

**Acceptance:** the manual checklist (Appendix A) passes against a real
recorded 120-frame run.

---

### Phase 6 — Documentation + lint

(The page replacement, deletions, and BUILD `data` flip all happened in Phase
5's atomic commit — this phase is docs and lint only.)

1. **Docs** (from the docs audit):
   - `websocket_server.py` module docstring → the canonical v2 protocol spec
     (it is the documented source of truth; currently stale even for v1);
   - `ARCHITECTURE.md`: §2 repo map (new modules, logs/runs/, analyzer.html —
     also fix the stale ilqr listing that omits solver.py/arm.py), §4.5
     (instrumentation + zero-overhead-off guarantee), §8.2 rewrite
     (simulate/playback modes, v2 protocol pointer), §9 (duckdb dep, new Bazel
     targets), §10 (data-flow), §11.5 (recorder as host-side per-frame sink);
   - `CLAUDE.md`: project structure, Build & Run (headless runner command,
     analyzer URL), tech stack (duckdb), key concepts (record/replay), a
     gotcha for the per-run DuckDB lock behavior + `record flags default off`;
   - `README.md`: running instructions + the stale "Current limitations";
   - `TRAINING.md`: note `pip install -e ".[dev]"` picks up duckdb;
   - `tests/README.md`: describe the five new test files.
2. `make lint` clean on all new files (ruff: E,W,F,I,N,UP,B,SIM,RUF; line 88,
   double quotes).

**Acceptance:** full suite green (`python run_tests.py` and
`bazel test //...`), fresh-clone flow works: install → headless run → server →
analyzer.

---

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Frame-1 stall: 8 per-limb tf.function compiles (tens of seconds) looks like a hang between `simulate_started` and first progress | `frame_ms` in progress lets the UI message it; consider an early frame-0 ping; documented |
| DuckDB WAL of a crashed run may need a read-write open to replay before `read_only` works | RunStore attempts rw-open-then-close on `aborted` runs; explicit test |
| `x_traj.numpy()` may share the tensor buffer | Tensors are immutable and rebound each acceptance; recorder `.copy()`s on write (~1.3 KB) as belt-and-braces |
| `cholesky_fail` path nearly unreachable with Gauss-Newton PSD Hessians | Ships as recorded-but-lightly-tested; a NaN-injection solver test is optional (deferred) |
| Recorder drains `limb.last_ilqr_history` which is rebound every frame | Drain point is inside `snapshot_state`, guaranteed between `octo.move()` calls |
| VIZ_ILQR's machine-specific `background_image` → cross-machine runs differ | Surface grid is recorded (playback always faithful); reproduction caveat documented |
| `asyncio.to_thread` default executor shared by simulate + playback reads | Fine at localhost scale; dedicated executor is a follow-up if scrubbing lags mid-simulate |
| Two server processes on one logs/runs/ dir | Out of scope (localhost tool); run_id uuid suffix avoids collisions |
| Analyzer page needs internet (CDN React/Babel/Tailwind) | Accepted (D9); vendoring is a known follow-up |
| Old num_iterations=-1 (infinite) convention | Rejected for recorded runs: runner raises, server `bad_request`, client input min=1 |

## 6. Deferred / open questions

- `delete_run` message + retention policy for `logs/runs/` (manual `rm` for now).
- Batch `get_frame` for iLQR data (per-frame lazy fetch is the design; add if
  overlay scrubbing across frames feels slow).
- Vendoring the frontend libraries for offline use.
- Deprecating ForceLogger (the new `limb_frames` table duplicates its useful
  columns); both can run side by side for now.
- A `phase` field in `simulate_progress` ('compiling' vs 'running') for the
  frame-1 stall.
- Per-iteration line-search depth (count of alphas tried) — one int, free to
  add if wanted.

## 7. New-file summary

| File | Phase | Purpose |
|---|---|---|
| `simulator/sim_recorder.py` | 2 | DuckDB SimRecorder |
| `simulator/headless_runner.py` | 3 | HeadlessRunner + serialize_state + CLI |
| `simulator/run_store.py` | 4 | read-only playback query layer |
| `visualizer/analyzer.html` | 5 | the analyzer (replaces octopus-visualizer.html) |
| `tests/test_sim_recorder.py` | 2 | |
| `tests/test_headless_runner.py` | 3 | |
| `tests/test_websocket_protocol.py` | 4 | |
| `tests/test_record_playback.py` | 4 | integration |
| `tests/test_analyzer_core.py` | 5 | node-based core-logic tests |

Modified: `solver.py`, `arm.py`, `octopus_generator.py` (P1);
`config_schema.py`, `config.py`, `pyproject.toml`, `test_config_schema.py`
(P0); `websocket_server.py` (P4); BUILD files (P1/2/3/4/5); docs (P6).
Deleted: `octopus-visualizer.html`, `octopus-ai-visualizer.tsx` (P5, atomic
with the analyzer landing).

---

## Appendix A — Analyzer manual checklist (Phase 5 acceptance)

Prerequisites: server running (`python visualizer/websocket_server.py`), at
least one recorded run on disk.

1. Open `http://localhost:8765/` — page loads, auto-connects, status green.
2. Old path `/octopus-visualizer.html` serves the same (new) page.
3. Simulate with default 120: progress bar advances; on completion the canvas
   **draws `final_state`** and the new run is auto-selected in Playback.
4. Simulate input rejects 0 / -1 / non-numeric (button disabled + hint).
5. Kill the server mid-simulate: error banner appears; restart + reconnect
   works; repeatedly toggling Connect never produces duplicate sockets.
6. Playback: run list populates (newest first, status + has-iLQR badges);
   loading a run shows the background immediately and frame 0.
7. Timeline drag is smooth after ~1 s (bulk fill) — no visible fetch stalls;
   jump End then Home works.
8. Play/pause at each speed (0.5/1/2/4×); playing to the last frame stops
   cleanly.
9. Frame counter, visibility, prey_captured readouts update per frame.
10. Visibility chart shows both series (before dashed, after solid); vertical
    cursor tracks the current frame; clicking the chart seeks.
11. Enable iLQR overlay (`I`): ghost horizon poses + cyan tip path render for
    the selected limb; "All" shows all 8.
12. Iteration slider (`[`/`]`) walks iterations; the cost-chart dot follows;
    accepted costs are non-increasing across iterations (sanity of data AND
    indexing).
13. Convergence mode (`V`) shows per-iteration terminal poses grading
    red→green.
14. **The final iteration's t=0 dashed pose approximately matches the NEXT
    frame's solid centerline** — the end-to-end coordinate-frame and
    iteration-indexing validation.
15. Sucker color mode cycle (`C`): after / before / target / error-heatmap all
    render; before-vs-after visibly differs on early frames.
16. Hover a sucker: inspector shows three swatches (before/after/target) with
    RGB values and deltas; moving off clears it.
17. All keyboard shortcuts fire; all are ignored while typing in an input.
18. Empty runs list shows an empty state — never fabricated data.
19. Resize the window: canvas rescales; no horizontal page scroll.
20. An unknown ws message is logged to console; the page keeps working.
