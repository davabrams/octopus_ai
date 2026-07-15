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
│   ├── websocket_server.py   # WebSocket sim server for browser viz (ws://localhost:8765)
│   ├── octopus-visualizer.html  # Self-contained HTML/JS frontend
│   └── octopus-ai-visualizer.tsx # React version of the same frontend
├── simulator/
│   ├── simutil.py           # State, Agent, Color, enums, matplotlib display helpers
│   ├── octopus_generator.py # Sucker, Limb, Octopus classes
│   ├── agent_generator.py   # AgentGenerator (prey/threat spawning + motion)
│   ├── surface_generator.py # RandomSurface (binary grid)
│   └── ilqr/
│       ├── costs.py         # CostTemplate + ColocationRepeller/MaxDistanceRepeller/PointAttractor/AllCosts
│       └── nodemesh.py      # Interactive node-graph octopus prototype (main()-guarded)
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
├── Makefile, run_tests.py, test.sh   # Test/lint entry points
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

### 4.5 `simulator/ilqr/` (standalone prototype)

- `costs.py` defines a small gradient-cost framework:
  - `ColocationRepeller` — pushes nodes apart below `min_distance`;
  - `MaxDistanceRepeller` — pulls neighbors together beyond `max_distance`;
  - `PointAttractor` — exponential-falloff pull toward a point;
  - `AllCosts` — composes the above over all nodes/neighbors/attractors and
    implements a naive `line_search()` over candidate alphas, returning
    `(best_alpha, best_cost, best_grad)`;
  - `cost_heatmap()` / `plot_graph()` demo functions under `__main__`.
- `nodemesh.py` builds a 4-limb × 4-sucker node graph (networkx) and
  animates it with attractors (including mouse-following), applying
  line-searched gradients per frame. The animation lives in a
  `main()`-guarded entry point, so the module is safely
  importable. Run it via `python simulator/ilqr/nodemesh.py` or
  `bazel run //simulator/ilqr:nodemesh`.

This subsystem shares `State` with the simulator but nothing else; the main
octopus does not use iLQR motion yet (that's the intent behind the TODO in
`Limb.move`).

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
  `sys.path.insert(1, '..')` — so run the server from inside
  `inference_server/` (`cd inference_server && python server.py`); it
  listens on `localhost:8080`.
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

### 8.2 WebSocket stack

- `visualizer/websocket_server.py` (rewritten July 2026) streams simulation
  state as JSON at ~10 FPS on `ws://localhost:8765`, with play/pause/reset
  and config-update messages. It drives the **real** simulator classes:
  a flat mirror for the wire and a typed `Config` rebuilt from it on every
  update, real `Octopus` serialization (limb centerlines
  as `limbs`, flat `suckers` list with `color`/`target_color`, agents as
  `prey`/`predator`), visibility via `Octopus.visibility`, and optional ML
  inference per `cfg.inference.mode` with heuristic fallback
  if the model can't load. Config updates are type-coerced to the existing
  value's type and rebuild the simulation. Handler uses the websockets ≥13
  single-argument API. Wire format is documented in the module docstring.
  Run: `python visualizer/websocket_server.py` (no Bazel target).
- `visualizer/octopus-visualizer.html` — self-contained browser frontend
  (canvas rendering, config sliders, connect/play/pause/reset). Expects
  messages of type `simulation_state` with `{background,
  octopus{head,limbs,suckers}, agents, metadata{iteration,
  visibility_score, fps}}`. The server serves this file from next to
  itself, so it must stay beside `websocket_server.py`.
- `visualizer/octopus-ai-visualizer.tsx` — the same UI as a React component
  (lucide-react icons), for embedding elsewhere.

---

## 9. Build, test, lint

- **Environment**: Python ≥3.10 (venv here is 3.12, ARM-native — see
  TRAINING.md for Apple Silicon setup). `pip install -e ".[dev]"`.
- **Bazel**: Bzlmod-era (`MODULE.bazel`, empty `WORKSPACE`). Targets:
  `//visualizer:octo_viz`, `//octopus_ai:datagen`, `//octopus_ai:model`,
  `//simulator/ilqr:nodemesh`, plus `py_test` targets in `tests/BUILD`.
  Bazel does not manage Python deps here — it relies on the ambient
  interpreter having tensorflow etc.
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
