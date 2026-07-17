# Architecture Reference

Deep, module-by-module documentation of the octopus_ai codebase. Written for
future maintainers (human or Claude) who need to understand how the pieces
actually connect. For quick orientation start with `CLAUDE.md`; for training
workflows see `TRAINING.md`. A July 2026 fix pass repaired every issue
found in a full-code audit (details live in that pass's commit
messages); the few deliberately-open gaps are called out inline below
and in CLAUDE.md's gotchas.

---

## 1. What this project is

A toy-but-serious sandbox that models an octopus in a 2D grid world:

- The **octopus** has a head at (x, y), 8 **limbs**, and each limb carries a
  grid of **suckers** (default 16 rows × 2 cols → 32 suckers per limb, 256
  total).
- The world has a random black/white **surface** and roaming **agents**
  (prey / threats).
- Suckers try to **camouflage**: match their color to the surface beneath
  them, subject to a physical constraint that color can only change by
  `octo_max_hue_change` (0.25) per timestep.
- Camouflage can be driven by a **heuristic** (clamped color step) or by
  trained **ML models** (TensorFlow/Keras).
- A separate **iLQR-flavored prototype** (`simulator/ilqr/`) explores
  gradient-based limb motion as a graph of nodes with attractor/repeller
  costs. It is standalone and not wired into the main simulator.

Everything is grayscale in practice: `Color` has r/g/b, but only `c.r` is
used as the signal and g/b are set equal to r.

---

## 2. Repository map

```
octopus_ai/                   # repo root
├── octopus_ai/              # core package (importable as octopus_ai.*)
│   ├── config_schema.py     # THE config: frozen Config dataclass tree (source of truth)
│   ├── config.py            # Profiles built from it + flat<->nested converters + path maps
│   ├── util.py              # erase_all_logs, octo_norm; re-exports split fns from training.data_utils
│   ├── datagen.py           # OctoDatagen class + standalone datagen entry point
│   └── model.py             # Entry point: datagen → train → save → (inference/eval)
├── visualizer/
│   ├── octo_viz.py           # Entry point: matplotlib visualizer loop
│   ├── websocket_server.py   # Analyzer server, v2 protocol (Simulate + Playback); ws://localhost:8765
│   └── analyzer.html         # Self-contained record & replay analyzer (React 18 UMD, no build step)
├── simulator/
│   ├── simutil.py           # State, Agent, Color, enums, matplotlib display helpers
│   ├── octopus_generator.py # Sucker, Limb, Octopus classes
│   ├── agent_generator.py   # AgentGenerator (prey/threat spawning + motion)
│   ├── surface_generator.py # RandomSurface (RGB grid)
│   ├── sim_recorder.py      # Record & replay: per-frame DuckDB writer (one file per run)
│   ├── headless_runner.py   # Record & replay: the ONE headless sim loop (CLI + server share it)
│   ├── run_store.py         # Record & replay: read-only playback query layer over logs/runs/*.duckdb
│   └── ilqr/
│       ├── solver.py        # Generic Gauss-Newton iLQR optimizer (+ opt-in per-iteration history)
│       ├── residuals.py     # Differentiable residual library (spring/bend/repel/reach/effort)
│       └── arm.py           # ArmController: single-limb model + compiled solver (composes residuals.py)
├── training/
│   ├── trainutil.py         # Trainer base class (raises RuntimeError on unimplemented)
│   ├── sucker.py            # SuckerTrainer: datagen/format/train/inference
│   ├── limb.py              # LimbTrainer: dual-input (fixed + ragged) model
│   ├── losses.py            # ConstraintLoss, WeightedSumLoss, plot_loss_functions
│   ├── data_utils.py        # Canonical train/test split + tf.data conversion
│   ├── datagen/
│   │   ├── data_loader.py   # DataLoader (pickle files); absolute defaults → training/datagen/*.pkl
│   │   └── sucker.pkl       # Generated dataset (pickle)
│   └── models/
│       ├── base_loader.py   # DefaultLoader ABC (canonical copy)
│       ├── model_loader.py  # Keras model loader wrapper
│       ├── sucker.keras     # Trained sucker model (working)
│       └── limb.keras       # Trained limb model (from Apr 2024)
├── inference_server/
│   ├── server.py            # Flask REST server on localhost:8080
│   ├── model_inference.py   # InferenceJob / InferenceQueue + watchdog thread
│   └── test_server.py       # unittest client/server tests (run from this dir)
├── tests/                   # pytest suite, ~2,400 lines / 136 tests across 8 files
├── models/logs/             # TensorBoard output (sucker/, limb/)
├── MODULE.bazel             # Bazel (Bzlmod) module; BUILD files are per-package
├── Makefile, run_tests.py   # Test/lint entry points
└── pyproject.toml           # pip deps, ruff config, pytest config
```

---

## 3. Configuration (`config_schema.py` + `config.py`)

A tree of **frozen dataclasses**, rooted at `Config` in
`config_schema.py`. Nesting follows the concern, not the old key prefix:

| Section | Holds |
|---------|-------|
| `cfg.run` | `num_iterations` (120; `-1` = forever), `rand_seed`, `threading` |
| `cfg.world` | `x_len` / `y_len` (15), `surface_grayscale` |
| `cfg.agents` | `count`, velocities, `movement_mode`, `sensing_radius`, prey capture |
| `cfg.octopus` | `num_arms`, `sensing_radius`, `.limb` (rows/cols, `.random` / `.lumped` / `.chain` knobs), `.sucker` (`max_hue_change` 0.25, `adjacency_radius`) |
| `cfg.inference` | `location`, `mode`, `model` |
| `cfg.output` | `debug_mode`, `show_forces`, `log_forces`, `save_images`, `video_fps` |
| `cfg.datagen` | `write_format`, `randomize_colors_interval`, disk toggles |
| `cfg.training` | `ml_mode`, `training_model`, `epochs`, `batch_size`, `test_size`, `constraint_loss_weight`, `sucker_delta_model`, run/save/tensorboard switches |
| `cfg.paths` | `model_paths` / `dataset_paths`, `MLMode`-keyed |

The hyperparams under `cfg.training` used to be duplicated across both flat
dicts with nothing keeping them in sync; there is one source now.

`config.py` builds the profiles — `DEFAULT`, `VIZ`, `DEBUG`, `TEST`,
`DATAGEN`, `TRAINING` — with `dataclasses.replace()`, which is also how you
derive an ad-hoc variant. Selecting a profile is what "experimenting" means;
there is no shared dict to edit and forget to revert.

Mode-specific limb knobs live under the mode that reads them
(`cfg.octopus.limb.chain.spring_k`), so "which knobs apply to SPRING_CHAIN?"
is an autocomplete question.

`config_to_flat` / `config_from_flat` convert to and from the flat dict form
for the three callers that legitimately speak it: the browser wire protocol,
the force-log config snapshot (flat keys suit SQLite's `json_extract`), and
test fixtures. `config_from_flat` is deliberately tolerant — an omitted key
falls back to `DEFAULT` — and `as_config()` normalizes a Config-or-dict.

Path maps (both `MLMode`-keyed, absolute paths under `ROOT_DIR`):

- `default_models` → `training/models/{sucker,limb}.keras`
- `default_datasets` → `training/datagen/{sucker,limb}.pkl` (added in the
  July 2026 fix pass; used by dataset save/restore)

Access is attribute-style: `cfg.world.x_len`. Configs are frozen;
in-place mutation raises `FrozenInstanceError`. Resolve model/dataset paths
via `cfg.inference_model_path` / `cfg.training_model_path` /
`cfg.training_dataset_path` rather than indexing the path maps by hand.

Key enums live in `simulator/simutil.py`:

- `MLMode`: `NO_MODEL` (heuristic), `SUCKER`, `LIMB`, `FULL` (placeholder)
- `MovementMode`: `RANDOM`, `ATTRACT_REPEL` (stubbed everywhere — only
  RANDOM works)
- `AgentType`: `PREY`, `THREAT`
- `InferenceLocation`: `LOCAL` (0), `REMOTE` (1). Distinct values, but
  nothing routes inference to the remote server yet — remote mode is
  still aspirational.

---

## 4. Simulator core

### 4.1 `simulator/simutil.py`

- **`State(KinematicPrimitive)`** — 6-DOF kinematic state stored as a
  `tf.Variable` of shape (6,): `[x, y, θ, vx, vy, ω]`. Float properties with
  setters wrap the tensor. `update_kinematics()` integrates position with
  `dt = 1.0` and wraps θ to [0, 2π). `apply_grad(grad)` sets `vel = grad`
  and adds grad to position (used by the iLQR prototype).
  `KinematicPrimitive._dims` is a type annotation only; instances create
  their own variable (no TF work at import time).
- **`Agent(State)`** — adds `agent_type`.
- **`CenterPoint(State)`** — marker subclass used for limb spline nodes.
- **`Color`** — dataclass r/g/b (floats in [0,1], default 0.5 gray);
  `to_rgb()` → np array.
- **`setup_display()` / `display_refresh(...)`** — matplotlib rendering of
  surface (imshow binary), suckers (dots colored by `sucker.c`), limb
  centerlines (debug mode), and agents (green=prey, violet=threat, keyed
  off `agent.agent_type`).
- **`convert_adjacents_to_ragged_tensor(adjacents)`** — turns
  `[(Sucker, dist), ...]` into a `tf.ragged.constant([[colors],[dists]])`
  for the limb model.

### 4.2 `simulator/octopus_generator.py`

Three-layer containment: `Octopus` → `Limb` → `Sucker`.

**`Sucker`** (plain class; x, y, `c: Color`, `max_hue_change`, `prev`)
- `get_surf_color_at_this_sucker(surf)` — rounds x/y to grid, reads the
  binary surface value, returns as gray Color.
- `_find_color_change(c_start, c_target)` — the heuristic: step toward the
  target, clamped to ±`max_hue_change` and to [0,1].
- `find_color(args_tuple)` — thread-pool-friendly signature; a single tuple
  `(surf, inference_mode, model, adjacents, ix)`. Dispatch:
  - `NO_MODEL` → heuristic;
  - `SUCKER` → `model.predict([[self.c.r, surf_color]])`;
  - `LIMB` → dual input: fixed `[[c, surf_color]]` plus ragged adjacency
    tensor.
  Model type is validated as `keras.Model` (both the Sequential sucker
  model and the functional limb model pass). Returns `(Color, ix)` so
  results can be re-sorted after unordered pooling. Only the red channel
  is predicted; g and b are copied from r.
- `set_loc(x, y)` — keeps a one-deep `prev` chain (memory-bounded).

**`Limb`**
- Holds `center_line: list[CenterPoint]` of length `limb_rows`, and a flat
  instance-level `suckers` list of length `rows * cols` (indexed
  `row + rows * col`).
- `_gen_centerline` lays rows out radially from the octopus head at
  `init_angle`; `_refresh_sucker_locations` places `cols` suckers
  perpendicular to each centerline point, offset by `sucker_distance`, and
  clamps positions to the grid ([-0.5, len-0.51]).
- `move(x_octo, y_octo)` — `RANDOM` mode jitters each centerline point's θ
  by ±`octo_max_arm_theta` and chains points outward from the head;
  `sucker_distance` random-walks within [min, max]. `ATTRACT_REPEL` raises
  NotImplementedError.
- `find_adjacents(s, radius)` — brute-force O(n) within-limb neighbor
  search returning `[(Sucker, dist), ...]` (includes the target sucker
  itself at dist 0).
- `find_color(surf, mode, model)` — ThreadPool `imap_unordered` over
  suckers, sorted back by index; returns `list[Color]`.
- `force_color(color_array)` — asserts length/type and assigns.

**`Octopus`**
- Head position starts at grid center; builds `octo_num_arms` limbs at
  evenly spaced angles.
- `move(ag)` — `RANDOM` jitters head by ±`octo_max_body_velocity`, then
  moves each limb. `ATTRACT_REPEL` is a stub that just prints.
- `find_color(surf, mode, model)` — ThreadPool over limbs (2 workers),
  each limb pools over suckers; returns `list[list[Color]]` sorted by limb
  index.
- `set_color(surf, mode, model)` — `find_color` followed by a per-limb
  `force_color` loop. (Was a no-op before July 2026. Datasets
  generated before the fix have sucker state pinned at the 0.5 default;
  regenerate them.)
- `visibility(surf)` — mean squared color error across all suckers
  (lower = better camouflaged). Shown live in both visualizers.

### 4.3 `simulator/agent_generator.py`

`AgentGenerator(params)` seeds numpy's **global** RNG from `rand_seed` on
construction (so object-creation order affects random sequences) and spawns
agents with random position/velocity and a coin-flip PREY/THREAT type (or a
fixed type). `agents` is an instance attribute. `increment_all(octo)`
advances agents: `RANDOM` mode integrates kinematics then re-rolls
velocities and ω; `ATTRACT_REPEL` is a stub that returns agents unchanged.

### 4.4 `simulator/surface_generator.py`

`RandomSurface(params)` — a `y_len × x_len` int8 grid of {0,1} from the
seeded RNG. `get_val(x, y)` rounds float coords and raises `ValueError` out
of bounds. Note the grid is indexed `[y][x]`.

### 4.5 `simulator/ilqr/` (limb motor control)

The built-out, TensorFlow-native iLQR that drives `MovementMode.ILQR`:

- `solver.py` — a **generic Gauss-Newton iLQR** optimizer. `make_solver(...)`
  builds a reusable solver from three differentiable callables (`dynamics`,
  `running_cost`, `terminal_cost`) plus a per-solve **params tensor** carrying
  frame-varying data (base position, reach target). Costs are squared-residual
  vectors so the Hessian is the Gauss-Newton `2·Jᵀ·J` (always PSD). The heavy
  per-step Jacobian/cost kernels are `@tf.function`-compiled once and reused,
  so calling the returned solver each frame does not retrace — the difference
  between a ~240 ms and a ~26 s solve (see §11.6). Standard iLQR loop: forward
  rollout → backward Riccati pass → line search with Levenberg-Marquardt
  regularization. Both passes are horizon recurrences → CPU work.
- `arm.py` — the single-limb model. `ArmController` is a **persistent per-limb
  controller**: chain of nodes (node 0 = base, pinned to the body; the rest are
  decision variables), single-integrator dynamics `x' = x + u·dt`, and
  squared-residual costs (neighbour springs toward `rest_length`, control
  effort, per-node attract/repel from each node's own sensed target/threat). It
  builds its compiled solver once and exposes `solve(base_xy, attract_tgt,
  attract_sw, repel_tgt, repel_sw, x0, u_init)` — the attract/repel arrays are
  per free node; `Limb._move_ilqr` fills them from per-node sensing and calls it
  receding-horizon (MPC): plan, apply the first step, warm-start next frame.
- **Per-iteration history (record & replay).** `solve(..., record_history=True)`
  captures an `ILQRIterationRecord` per iteration (iter 0 = warm-start rollout;
  then `accepted`/`cholesky_fail`/`linesearch_fail`, with the unclamped
  `x_traj`), returned on `ILQRResult.history`; invariant `len(history) ==
  iterations + 1`. It is a **per-call kwarg** (not a controller field) so one
  compiled controller serves both modes, and it never reaches a `@tf.function`
  — the loop is eager Python, so it is **zero-overhead when off**. `Limb`
  drains `last_ilqr_history`/`last_ilqr_meta` for `SimRecorder` when
  `output.record_ilqr_history` is set.
- `residuals.py` — the **cost library**. Each cost term (spring, bend, threat
  repel, tip reach, control effort) is a pure, differentiable TF function
  returning a **residual vector**; the solver squares and sums them, and its
  weight enters as `sqrt(w)` (so `‖√w·r‖² = w·‖r‖²`). `arm.py` composes these
  into its `running_cost`/`terminal_cost`. This is the one home for cost terms —
  a new drive (e.g. exploration) adds a residual here. Residuals must be
  shape-stable (one-sided barriers use `relu`, not `if`) so the compiled graph
  never retraces.
- Tests: `tests/test_ilqr.py` (reach convergence, multi-target graph reuse,
  effort-holds-still, and the history capture invariants).

Each limb owns its own `ArmController` and solves independently of the others
(§11.4).

**Base ring + body rotation (BODY_ROTATION_PLAN.md).** Each limb's base (node 0)
is pinned not to the body center but to its own point on a ring of radius
`octo_ring_radius` around the body, at a fixed angular slot `2π·i/N` rotated by
the body's orientation `theta`: `base_i = body + R·[cos(theta+φ_i),
sin(theta+φ_i)]`. So the arms fan out from *distinct* roots and can't collapse
into one line (the limbs stay independent — only the geometry changed).
`Octopus` gains an orientation `theta` (the angular twin of `x, y`):
`_drift_body_by_tension` integrates the summed arm *torque* (`Σ r_i × F_i`, each
arm's base reaction at its ring point) into `theta`, capped by
`octo_max_body_angular_velocity`, just as it integrates the linear tension sum
into position. The ring is the prerequisite for rotation — a center-pinned base
has no moment arm. `R = 0` reproduces the legacy single-point base.

> **Retired (July 2026):** an earlier standalone `costs.py` (`CostTemplate`/
> `AllCosts` gradient-relaxation classes) + `nodemesh.py` (a networkx node-graph
> animation) prototype was **deleted**. It emitted hand-written gradients rather
> than autodiff'd residuals — a different, incompatible paradigm — and was never
> wired into `Limb.move`. The active path is `solver.py` + `arm.py` +
> `residuals.py`.

iLQR is the **motor-control tier** of the compute hierarchy and is CPU
work with each limb solving independently — see §11.4 for the placement
rationale.

---

## 5. Data generation (`datagen.py`)

`OctoDatagen(game_parameters)`:

1. Loads the inference model via `ModelLoader(params['inference_model'])`
   (an `MLMode` resolved through `default_models`; `NO_MODEL` → loader
   returns with `object=None`).
2. `run_color_datagen()`:
   - Builds surface, agents, octopus; initial `set_color` with `NO_MODEL`.
   - Loops `num_iterations` times: move agents, move octopus, then for every
     sucker record **state** and **ground truth**:
     - `datagen_data_write_format == MLMode.SUCKER`: state = current sucker
       color scalar (`s.c.r`); gt = surface color under the sucker (`Color`).
     - `== MLMode.LIMB`: state = `{'color': s.c.r, 'adjacents':
       limb.find_adjacents(s, adjacency_radius)}`; gt same as above.
   - Runs inference with the configured mode (now genuinely updating colors
     each iteration, so state evolves) and logs outputs into `sucker_test`
     (currently unused).
   - Returns `{'metadata': {datetime,user,machine}, 'game_parameters',
     'state_data', 'gt_data'}`.

Data volume per run = `num_iterations × octo_num_arms × limb_rows ×
limb_cols` (default 120 × 8 × 16 × 2 = 30,720 points).

The `__main__` block uses its own hardcoded params (2 iterations,
`inference_mode=SUCKER`) and pickles the result to
`default_datasets[MLMode.SUCKER]` = `training/datagen/sucker.pkl`.

---

## 6. Training

### 6.1 Orchestration (`model.py`)

A script (module-level code, no `main()`) that selects the `TRAINING`
profile (`CFG` at the top of the file) and runs stages in order:

1. Optionally erase TensorBoard logs (`util.erase_all_logs`).
2. Pick trainer by `cfg.training.ml_mode`: `SuckerTrainer(CFG)` or
   `LimbTrainer(CFG)`.
3. **Data**: `datagen_mode` → `trainer.datagen(SAVE_DATA_TO_DISK)` then
   `trainer.data_format(data)`; else `restore_data_from_disk` → unpickle
   from `CFG.training_dataset_path`.
4. **Train**: `trainer.train(train_dataset, GENERATE_TENSORBOARD=...)`;
   save with `model.save(model_location)` if configured
   (`CFG.training_model_path`).
5. **Inference** (optional): reload via
   `ModelLoader(path, custom_objects=...).get_object()` when
   `restore_model_from_disk`, then run `trainer.inference(model)` sweep.
6. **Eval** (optional): `model.evaluate(test_dataset)`.

Don't import this module from library code — it executes at import.

### 6.2 SuckerTrainer (`training/sucker.py`)

- **Constructor** — `SuckerTrainer(cfg)`, one `Config`.
- **datagen** — wraps `OctoDatagen.run_color_datagen()`. When saving, the
  path comes from `cfg.paths.dataset_paths[MLMode.SUCKER]`, falling back to
  `default_datasets` if the config carries no paths table.
- **data_format** — model input is the pair *(current color, surface
  color)*; both the "x" and "y" of the tf.data pipeline are the same stacked
  `(state, gt)` tensor (the loss unpacks what it needs from `y_true`).
  Uses `train_test_split` from `training/data_utils.py` (train gets
  `1 - test_size` of the samples).
- **train / train_sucker_model** — Sequential MLP: input (2,) → Dense 5
  relu ×3 → Dense 1 linear. Custom training loop with `GradientTape`, SGD
  lr=1e-3, `WeightedSumLoss`. TensorBoard logs to
  `models/logs/sucker/fit/<timestamp>` when enabled.
- **inference** — 5×5 sweep over (previous color, surface color) ∈
  {0, .25, .5, .75, 1}², rendered as a seaborn heatmap. This is the sanity
  check for a trained model: predictions should move from the previous
  color toward the surface color but never jump more than ~0.25.

### 6.3 LimbTrainer (`training/limb.py`)

- **Model** (`_model_constructor`) — functional Keras with two inputs:
  - fixed branch: shape (2,) → Dense 5, 5, 4 (relu);
  - ragged branch: shape (None, 2) ragged → SimpleRNN 5 → Dense 5, 4;
  - concat → Dense 2, 2 → Dense 1 linear.
  The ragged input carries (color, distance) pairs of adjacent suckers.
- **data_format** — accepts data written in either SUCKER or LIMB format;
  for LIMB, unpacks adjacency lists into parallel ragged lists. Builds
  `tf.data.Dataset.from_generator` with `RaggedTensorSpec(shape=(4, None))`
  where the 4 rows are [state color], [gt], colors-of-adjacents,
  dists-of-adjacents. Uses `train_test_split_multiple_state_vectors`.
- **datagen** — saves to `cfg.training_dataset_path` (i.e.
  `paths.dataset_paths[cfg.training.ml_mode]`).
- **train** — same custom-loop pattern as sucker, batch semantics differ
  (each generator element is one sample; batching is effectively 1). Logs to
  `models/logs/limb/fit/<timestamp>`. Per-epoch ETA math is now
  divide-by-zero-safe. Calls `limb_model.reset_states()` per step
  (relevant to the RNN; may be a no-op or error under newer Keras — the
  limb pipeline is experimental and its saved artifact is from Apr 2024).

### 6.4 Losses (`training/losses.py`)

Both registered with `@keras.saving.register_keras_serializable(package="Octo")`
so saved models can round-trip.

- **`ConstraintLoss(threshold)`** — physical plausibility penalty:
  `sq(|pred − previous_color| − threshold)` where the diff exceeds the
  threshold, else 0. Ground truth is *not* used; `y_true` here is the
  previous state value. Extensive optional per-element TensorBoard logging.
- **`WeightedSumLoss(threshold, weight)`** —
  `w · ConstraintLoss(prev, pred) + (1 − w) · MAE(gt, pred)`, with
  `y_true` packed as `[state, gt]` rows (indices `f1_fields=0`,
  `f2_fields=1`). Default weight 0.95 heavily favors the constraint.
  Logging-step math guards a `None` (symbolic) batch dimension.
- **`plot_loss_functions(...)`** — diagnostic plots of the loss landscape
  over a prediction sweep; gated behind `if False:` at module bottom (flip
  to use).

### 6.5 Loaders and data utilities

- `training/models/base_loader.py` — `DefaultLoader` ABC: accepts a path
  string or an `MLMode` (mapped through the subclass `defaults` dict),
  resolves bare filenames relative to the subclass's directory, verifies
  existence, calls `_load`. `get_object()` returns the loaded thing.
- `training/models/model_loader.py` — `ModelLoader`: `defaults =
  default_models` from config; `_load` calls `keras.models.load_model`,
  passing `custom_objects` when provided as a keyword (e.g.
  `ModelLoader(path, custom_objects={"ConstraintLoss": ConstraintLoss})`).
- `training/datagen/data_loader.py` — `DataLoader`: absolute defaults →
  `training/datagen/{sucker,limb}.pkl`; `_load` unpickles.
- `training/data_utils.py` — canonical `train_test_split` (train gets
  `1 - test_size`; honors `random_state`),
  `train_test_split_multiple_state_vectors`, and
  `convert_pytype_to_tf_dataset`. `util.py` re-exports these for
  backwards compatibility and additionally provides `erase_all_logs`
  (clears both sucker and limb TensorBoard dirs) and `octo_norm`.

### 6.6 Trained artifacts on disk

- `training/models/sucker.keras` — current sucker model (regenerated
  Feb 2026); loads and produces a sensible inference heatmap.
- `training/models/limb.keras` — limb model from Apr 2024; compatibility
  with the current TF version is unverified.
- `training/models/sucker.keras has pretty good results` — a stray backup
  file (Dec 2023) whose filename is a note-to-self. Safe to delete/rename.
- `training/datagen/sucker.pkl` — pickled dataset. **If it predates July
  2026, regenerate it** (`python octopus_ai/datagen.py`): older pickles were
  produced while `set_color` was a no-op, pinning sucker state
  at 0.5.

---

## 7. Inference server (`inference_server/`)

A Flask REST server intended to let simulations offload model prediction.

- **Startup**: `model_inference.py` loads the sucker model **at import
  time** using `default_models[MLMode.SUCKER]`, and does
  `sys.path.insert(1, '..')`. Run it via `bazel run
  //inference_server:server` (the BUILD's `imports = ["."]` puts the
  package dir on the path so the in-directory bare imports resolve), or
  directly with `cd inference_server && python server.py`. It listens on
  `localhost:8080`.
- **Queueing** (`InferenceQueue`): all queue state is per-instance
  (a July 2026 fix; they were class-level and shared). Jobs live in
  `_q` (id → `InferenceJob`) and move through
  `_pending_queue` → `_execution_queue` → `_completion_queue` (lists of
  `(timestamp, job_id)`). A daemon **watchdog thread** polls every 0.1 s
  and kicks off pending jobs (executes newest-first, up to
  `thread_count=2` in flight) on dedicated executor threads.
  `clear_stale()` (30 s expiry for pending jobs; returns removed ids)
  exists but is deliberately not called by the watchdog: pending jobs
  are normally picked up within ~0.1 s, and auto-deleting queued jobs
  could surprise clients. Re-enable deliberately if queue growth ever
  becomes a problem.
- **Job payload** — POST `/jobs` expects:

  ```json
  {"job_id": 1, "data": {"c.r": 0.5, "c_val.r": 1.0}}
  ```

  where `c.r` is the sucker's current color and `c_val.r` the surface color.
  `ExecuteSuckerInference` runs `sucker_model.predict([[c, c_val]])` and
  stores the scalar result.
- **Endpoints**:
  - `POST /jobs` — enqueue (500 if job_id exists);
  - `GET /jobs/<id>` — status/result; deletes the job on terminal status
    (201 + delete on `Complete`, 500 + delete on `Failed`, 200 while
    pending/executing, 404 if unknown);
  - `GET /list_jobs` — all jobs + statuses;
  - `GET /show_queues` — raw queue contents;
  - `POST /collect_and_clear` — drain completed jobs;
  - `GET /shutdown` — stops the watchdog (process keeps running).
- **Status strings**: `JobStatus` StrEnum — Pending / Executing / Complete /
  Failed. `server.py` compares against the enum members.
- **Client + tests**: `inference_server/test_server.py` contains a small
  requests-based client and unittest suite; run from within the directory.
  `tests/test_inference_server.py` covers the queue/job classes from the
  main suite.

Nothing in the simulator automatically routes to this server yet;
`InferenceLocation.REMOTE` is a distinct enum value but unused.

---

## 8. Visualization

All visualization code lives under `visualizer/`. The two Python entry
points import top-level project modules, so each inserts the repo root onto
`sys.path` and is meant to be run from the repo root
(`python visualizer/octo_viz.py`).

### 8.1 Local matplotlib (`visualizer/octo_viz.py`)

Selects a profile (`CFG = DEFAULT` at the top; swap for `VIZ` or
`DEBUG`), builds surface/agents/octopus, optionally loads a
model (validated as `keras.Model`), then loops: `display_refresh`, show
`visibility` score, move agents/octopus, compute colors via
`octo.find_color(...)` and apply with `limb.force_color(...)` per limb.
`debug_mode` adds centerlines, sucker outlines, and agent range circles.
Requires a GUI backend and an initial button press
(`fig.waitforbuttonpress()`); run from a real display session.

### 8.2 WebSocket stack (record & replay analyzer)

The live-streaming v1 server was replaced (RECORD_REPLAY_PLAN.md) by a **v2
record & replay** stack. There is no live free-running mode anymore.

- `visualizer/websocket_server.py` — the analyzer server on
  `ws://localhost:8765`. It **never runs the sim on the event loop**: a
  `simulate` request hands off to `HeadlessRunner.run` on a worker thread
  (`asyncio.to_thread`), coalesced `simulate_progress` broadcasts flow back, and
  cancellation is a `threading.Event` polled once per frame. Playback requests
  (`list_runs`/`load_run`/`get_frame`/`get_frames`) go through `RunStore` on
  worker threads, each opening a run's `.duckdb` read-only. Exactly one simulate
  runs at a time (`busy`); the active run's file is write-locked, so its list
  row is synthesized in memory. The **v2 wire protocol is the module docstring**
  (envelope `{type, req_id, data}`, floats rounded to 4 decimals, enums as
  `.name`); v1 `config_update`/`simulation_control` now return
  `error code="gone"`. Split into `websocket_server_lib` (importable, tested)
  + a binary, mirroring `inference_server`.
- `visualizer/analyzer.html` — the self-contained analyzer (React 18 UMD +
  Babel + Tailwind CDN, no build step). Two modes: **Simulate** (run a fresh
  headless sim, watch it record) and **Playback** (scrub a saved run
  frame-by-frame, and iLQR iteration-by-iteration within a frame — ghost
  horizon poses, tip path, cost curves, before/after/target sucker colors). Pure
  logic lives in a `window.AnalyzerCore` block that is unit-tested under node.
  The server serves it (and the old `/octopus-visualizer.html` URL aliases to
  it) from next to itself, so it must stay beside `websocket_server.py`.

Storage: `simulator/sim_recorder.py` writes one `logs/runs/<run_id>.duckdb` per
run (schema in the module; DuckDB is single-writer-per-file, hence per-run
files). `simulator/headless_runner.py` is the one headless loop shared by the
CLI and the server. `simulator/run_store.py` is the read-only query layer and
owns the iLQR trajectory reshape + carry-forward so no client reimplements it.

---

## 9. Build, test, lint

- **Environment**: Python ≥3.10 (venv here is 3.12, ARM-native — see
  TRAINING.md for Apple Silicon setup). `pip install -e ".[dev]"`.
- **Bazel**: Bzlmod-era (`MODULE.bazel`, no `WORKSPACE`). Runnable targets:
  `//visualizer:octo_viz`, `//visualizer:websocket_server`,
  `//octopus_ai:datagen`, `//octopus_ai:model`, `//inference_server:server`,
  `//simulator:headless_runner_bin` (record & replay CLI), plus `py_test`
  targets in `tests/BUILD`. New
  record & replay libraries: `//simulator:sim_recorder`, `//simulator:run_store`,
  `//simulator:headless_runner`, `//simulator/ilqr:solver`, `//simulator/ilqr:arm`,
  and `//visualizer:websocket_server_lib` (importable, socket-free-testable).
  Bazel does not manage Python deps here — it relies on the ambient
  interpreter having tensorflow etc. `config.py` reads
  `BUILD_WORKSPACE_DIRECTORY` (set by `bazel run`) so datagen/training write
  models and datasets into the source `training/` tree rather than the
  runfiles sandbox; `bazel test` leaves it unset and stays hermetic.
- **Tests**: `python run_tests.py` (pytest under the hood; `--verbose`,
  `--coverage`, `--test <file>`, `--runner unittest|bazel`,
  `--check-deps`). Makefile wraps the same plus per-file targets. Suite is
  8 files / 136 tests; `tests/README.md` describes coverage per file.
- **Lint**: ruff configured in `pyproject.toml` (E, W, F, I, N, UP, B, SIM,
  RUF; several N-rules ignored to tolerate the legacy CamelCase argument
  names that remain in the trainers).
  `make lint` / `make format` wrap `ruff check .` / `ruff format .`.
  Pre-existing lint debt remains in older modules and tests.

---

## 10. Data flow summary (sucker pipeline, the one that works end-to-end)

```
config.DEFAULT (a Config)
        │
        ▼
OctoDatagen.run_color_datagen()          # simulator rollout
  state = sucker current color (c.r)
  gt    = surface color under sucker
        │
        ▼
SuckerTrainer.data_format()              # split (80/20) + tf.data (batch 32)
        │
        ▼
SuckerTrainer.train()                    # MLP(2→5→5→5→1), SGD 1e-3
  loss = 0.95·ConstraintLoss(prev, pred)
       + 0.05·MAE(gt, pred)
        │
        ▼
training/models/sucker.keras
        │
        ├──▶ visualizer/octo_viz.py      # live camouflage w/ model (matplotlib)
        ├──▶ visualizer/websocket_server.py  # live camouflage in the browser
        ├──▶ SuckerTrainer.inference()   # 5×5 heatmap sanity check
        └──▶ inference_server            # REST predictions
```

The limb pipeline (adjacency-aware model) exists and has a saved artifact,
but the artifact is old and the RNN training loop hasn't been re-verified
under current Keras; treat it as experimental.

---

## 11. Compute placement philosophy (the fast/slow hierarchy)

This is the guiding principle behind how the project *wants* to compute, and
it should shape new simulator/inference code. It is a design north star, not
a description of everything that is wired up today — see "Current status"
below for the gap.

### 11.1 The thesis

The project models biological control as a hierarchy of **timescales**:

| Tier | What | Model size | Cadence | Home device |
|------|------|-----------|---------|-------------|
| Reflex | **Suckers** (camouflage color) | tiny (~50-param MLP) | every tick | **CPU**, batched |
| Motor control | **Limb iLQR** (arm trajectory) | small analytic optimizer | mid-rate, per-arm | **CPU**, parallel across arms |
| Cognition | **Octopus brain** (high-level choices) | large | slow | **GPU** |

Small models make **quick** choices; large models make **slow** choices; and
the slow tier must **never block** the fast tier. That "never block" is the
core requirement, and it drives every decision below.

### 11.2 Parallelism comes from batching, not from threads/agents

The intuitive model — "each sucker is an independent agent, give it its own
thread/process so they don't block each other" — is the wrong *execution*
mapping, even though it is a fine *conceptual* model. A GPU is one serial
device with one command stream: N per-agent inference calls (whether from N
threads or N processes) serialize on the hardware and each pays kernel-launch
overhead. That is the pathology the batched color path replaced (§4.2).

The correct realization of "N independent agents computed in parallel" is a
single call over a batched `(N, …)` tensor: the device evaluates all N rows
at once across its ALUs — genuine data-parallelism, no per-agent dispatch.
**Map one agent to one row of a batch, not to one thread.** Keep "agent" as
simulation semantics; keep the batch axis as the execution reality. If agents
later diverge in behavior, group same-model agents into batches or express the
branch with vectorized ops (`tf.where`), not per-agent calls.

### 11.3 "Slow doesn't block fast" is temporal decoupling, not simultaneity

On a single GPU you cannot run the slow brain and the fast reflex literally at
the same instant. You don't need to. The only thing that actually *blocks* is
an `await`. The fast loop stays at rate as long as it **never waits** for the
slow tier: the slow model runs as a background actor and *publishes* its latest
decision to a size-1 slot; the fast loop *reads the most recent published
decision* (non-blocking) and acts on it. The slow result is consumed whenever
it is ready, a few fast-ticks stale.

This staleness is biologically faithful: octopus arms carry substantial
autonomous neural processing and act on delayed, coarse central commands. The
lag is a feature of the model, not an artifact.

The repo already has the process-separated form of this: `InferenceLocation.
REMOTE` + the Flask `inference_server/` (§7) is a persistent process holding a
model, called asynchronously and coalescing requests via its job queue — the
natural home for the slow cognition tier. It is scaffolded but unwired.

### 11.4 Device placement, and why it resolves the "one GPU" problem

Don't put everything on the GPU. The tiers split naturally across hardware:

- **Tiny fast models → CPU.** A ~50-param MLP over 256 rows is microseconds;
  GPU kernel-launch overhead makes the GPU *lose* at this size (measured). The
  batched sucker path runs best on CPU.
- **Large slow model → GPU.** This is what actually benefits from the device.
- Because TensorFlow releases the GIL during kernel execution, a CPU-op thread
  and a GPU-op thread **genuinely overlap**. Placing the fast tier on CPU and
  the slow tier on GPU means they run on different hardware and truly do not
  contend — which is how you get "async on one GPU" without the GPU running two
  things at once. Use explicit placement: `with tf.device('/CPU:0'):` for the
  fast tier, `with tf.device('/GPU:0'):` for the brain.

**Limb iLQR is CPU** (§4.5): iLQR is sequential along the horizon (forward
rollout + backward Riccati are recurrences), does tiny dense linear algebra
per step (small matrices underutilize a GPU), and is branchy/iterative (line
search, regularization) — all CPU-favorable, latency-bound work. Each limb is
an **independent controller**: it solves its own controls for its own arm,
functioning autonomously of the other seven. That independence is the parallel
axis — the eight per-limb solves genuinely parallelize across CPU cores (BLAS
releases the GIL), leaving the GPU free for cognition. (This is autonomy per
limb, not one coupled batched solve across arms — each arm owns its optimizer.) The one condition that flips iLQR to GPU: if its **dynamics or cost
model becomes a neural network**, the rollout turns inference-bound and wants
to be batched on the GPU (differentiable iLQR, batched across arms/horizon).
With today's analytic costs it stays on CPU.

### 11.5 Current status (the honest gap)

- **Reflex tier: done.** Color inference for `NO_MODEL`/`SUCKER` is a single
  batched TensorFlow pass over all 256 suckers (`Octopus._find_color_batched`,
  §4.2), ~1300× faster than the old per-sucker `model.predict` loop, bit-for-bit
  identical output. `LIMB` still uses the per-sucker path (its ragged stateful
  RNN doesn't batch trivially).
- **Motor tier: built.** A TensorFlow Gauss-Newton iLQR (`simulator/ilqr/
  solver.py` + `arm.py`) drives `MovementMode.ILQR`: each limb owns a
  persistent, compiled `ArmController` and re-plans receding-horizon (MPC).
  Attraction and repulsion are **PER-NODE** (node-autonomous sensing, not a limb
  policy): every free node attracts to ITS OWN nearest sensed prey/explore cell
  and flees ITS OWN nearest sensed threat, gated by the sense window (weight 0
  where a node senses nothing) and passed to the solver as per-node arrays in the
  params tensor. Arm costs = spring (rest spacing) + bending (anti-crumple) +
  effort (per-node velocity) + per-node **attract** + per-node **repel** (graded
  body>tip via `repel_tip_fraction`). The body drifts by the summed per-arm base
  reactions - so it chases prey and flees threats with no central negotiation.
  CPU work per §11.4. All the knobs (horizon, iters, body stiffness, and every
  cost weight) live in
  `LimbConfig.ilqr` (`ILQRConfig`), tunable per profile via `replace()`.
  Remaining polish: warm-tune the per-frame iteration budget, and (optionally)
  share one compiled graph across arms to avoid N first-frame compiles while
  keeping per-arm state.
- **Cognition tier: not built.** No large brain model or async actor yet.
- **No GPU on the dev machine.** `tf.config.list_physical_devices('GPU')` is
  empty here (ARM Mac, `tensorflow-metal` not installed). So *today everything
  runs on CPU* and the CPU/GPU split is architecture, not a live configuration.
  Keep code vectorized and device-agnostic; validate placement once a GPU is
  available. The win from batching (§11.2) is real on CPU regardless.

### 11.6 Rules of thumb

- One agent → one row of a batch. Never one agent → one thread/process for
  inference.
- Never `await` the slow tier from the fast loop; read its latest published
  decision instead.
- Tiny models on CPU, large models on GPU; let the GIL-releasing overlap give
  you concurrency across the two.
- Keep every device submission bounded; if one forward pass would monopolize
  the GPU, chunk it or move it off the fast path.
- Vectorize and compile (`@tf.function`) before choosing a device — an eager,
  per-element implementation is the worst case on *any* device, and its cost
  swamps the placement decision.
