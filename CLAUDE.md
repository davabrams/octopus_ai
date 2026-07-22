# CLAUDE.md

## Project Overview

Octopus AI is a simulation and machine learning project that models octopus
behavior — locomotion, limb movement, and sucker-based camouflage (color
change). It includes a physics-ish simulator, ML training pipelines, a Flask
inference server, a WebSocket visualization server, and HTML/React visualizer
frontends.

## Documentation Index

| File | What it's for |
|------|---------------|
| `CLAUDE.md` (this file) | Orientation, conventions, commands |
| `ARCHITECTURE.md` | Deep module-by-module reference, data flow, APIs |
| `TRAINING.md` | Step-by-step training/inference workflows and env setup |
| `tests/README.md` | Per-file test coverage description |
| `README.md` | Public-facing intro |

## Tech Stack

- **Language:** Python 3.10+ (local venv is 3.12, ARM-native — see TRAINING.md)
- **ML Framework:** TensorFlow / Keras
- **Build System:** Bazel (Bzlmod / `MODULE.bazel`) — but Bazel does NOT
  manage Python deps; it uses the ambient interpreter. pip/venv is primary.
- **Testing:** pytest (primary, via `run_tests.py`), unittest, Bazel test
- **Linting:** ruff (configured in `pyproject.toml`; `make lint` / `make
  format` wrap it)
- **Visualization:** matplotlib (local) and WebSocket server + HTML/React
  frontends
- **Key Libraries:** numpy, tensorflow, matplotlib, seaborn, websockets (≥13
  single-arg handler API), flask, networkx, duckdb (record & replay storage)

## Project Structure (abridged — full map in ARCHITECTURE.md §2)

```
octopus_ai/                  # repo root
├── octopus_ai/             # core package: config, profiles, shared utils, datagen/train entry points
│   ├── config_schema.py    # The typed Config dataclasses — source of truth
│   ├── config.py           # Profiles (DEFAULT/VIZ/DEBUG/TEST/DATAGEN/TRAINING) + flat<->nested converters + path maps
│   ├── util.py             # erase_all_logs, octo_norm + re-exports from training.data_utils
│   ├── datagen.py          # OctoDatagen class + standalone pickle-writing main
│   └── model.py            # datagen → train → save → inference/eval pipeline
├── visualizer/             # matplotlib viz (octo_viz.py) + browser analyzer (websocket_server.py v2 protocol + analyzer.html)
├── simulator/              # simutil, octopus/agent/surface generators, ilqr/, record & replay (sim_recorder/headless_runner/run_store)
├── training/               # trainers (sucker, limb), losses, loaders, data_utils, saved .keras models
├── inference_server/       # Flask REST server (localhost:8080), job queue + watchdog
├── logs/runs/              # recorded runs, one <run_id>.duckdb per run (gitignored)
└── tests/                  # pytest suite
```

**Record & replay** (RECORD_REPLAY_PLAN.md): `simulator/headless_runner.py`
steps a headless sim and writes one `logs/runs/<run_id>.duckdb` per run via
`simulator/sim_recorder.py` (positions, before/after camouflage colors, iLQR
costs, and full per-iteration iLQR trajectories). `simulator/run_store.py` is
the read-only query layer. `visualizer/analyzer.html` is the browser analyzer
(Simulate + Playback) served by the rewritten `websocket_server.py` (v2
protocol; the old live-streaming `octopus-visualizer.html` is gone, its URL now
serves the analyzer). The analyzer also has a **live preview** (the server
streams each frame's geometry in the `simulate_progress` broadcast — same shape
as playback — so the canvas animates while recording), **state-colour** outlines
(behavior policy), a **measure** tool (`M`, click two points for distance +
angle), a **jet-radius** ring under the sensing overlay, and a **hover box** on
the cell a sucker's node is attracting toward (`explore_cell`/`attract_tgt`).

## Configuration — IMPORTANT

Config is a tree of **frozen dataclasses** defined in `config_schema.py`
(the source of truth). Access is **attribute-style**: `cfg.world.x_len`,
`cfg.octopus.num_arms`, `cfg.training.epochs`.

Note this is the reverse of what older code and history will show you. The
module-level mutable dicts `GameParameters` / `TrainingParameters` are
**gone**; so is dict-style `params['x_len']` access. If you find a branch
or a comment describing them as current, it predates the config reorg.

`config.py` builds the **profiles** — pick one, don't edit shared
state:

| Profile | For |
|---------|-----|
| `DEFAULT` | the shipped baseline; writes nothing to disk |
| `VIZ` | watching a run: force arrows on, disk quiet |
| `DEBUG` | everything on: arrows + force DB + PNG/MP4 capture |
| `TEST` | deterministic, side-effect free, needs no model on disk |
| `TRAINING` | the training pipeline; `model.py` selects it |
| `DATAGEN` | data generation. Defined and tested, but nothing selects it yet — `datagen.py`'s `__main__` still hand-builds a flat dict |
| `RECORD` | headless record & replay: VIZ_ILQR + `record_run` and `record_ilqr_history` on, force arrows off; `headless_runner.py` selects it |

Derive a variant with `dataclasses.replace()` — configs are frozen, so
in-place mutation raises:

```python
CFG = replace(VIZ, octopus=replace(VIZ.octopus,
                                   movement_mode=MovementMode.SPRING_CHAIN))
```

`config_to_flat` / `config_from_flat` convert to and from a flat dict. That
boundary exists for three callers only — the browser wire protocol, the
force-log snapshot, and test fixtures — and `as_config()` normalizes either
form. Don't reach for them in new simulator or training code; take a
`Config`.

Path maps in `config.py`, both `MLMode`-keyed and reachable as
`cfg.paths.model_paths` / `cfg.paths.dataset_paths`:
- `default_models` — `MLMode` → absolute `.keras` path
  (`training/models/{sucker,limb}.keras`).
- `default_datasets` — `MLMode` → absolute `.pkl` path
  (`training/datagen/{sucker,limb}.pkl`). Used by `save_data_to_disk` /
  `restore_data_from_disk`.

Resolve paths through the properties rather than indexing by hand:
`cfg.inference_model_path`, `cfg.training_model_path`,
`cfg.training_dataset_path`.

Tests build configs with `tests/helpers.make_config(**flat_overrides)`,
which is baselined on `TEST` and raises `UnknownConfigKey` on a typo.

## Key Concepts

- **State:** 6-DOF kinematic primitive `[x, y, θ, vx, vy, ω]` stored as a
  TensorFlow variable (`simulator/simutil.py`). Angles in radians, wrapped
  to [0, 2π). `dt = 1.0`.
- **Octopus model:** head at (x, y) with an orientation `theta` → 8 `Limb`s →
  each limb has `limb_rows × limb_cols` `Sucker`s (default 16×2 = 32; 256
  total). In `ILQR` mode each limb's base is pinned to its own point on a ring
  (`octo_ring_radius`) around the body — a fixed angular slot rotated by
  `theta` — so the arms fan out from distinct roots. The body integrates the
  arms' summed *torque* into `theta` (angular twin of the linear tension→drift),
  so the fan rotates. See BODY_ROTATION_PLAN.md.
- **Motor sensing is PER-NODE** (node-autonomous, not a limb policy): in `ILQR`
  mode every free centerline node independently **attracts** to the nearest prey
  within its sense window (strong) and **flees** the nearest threat within it
  (`repel`, graded so the body-adjacent node recoils hardest and the tip least,
  `octo_ilqr_repel_tip_fraction`). Both can act on different nodes of one arm at
  once — there is no single per-limb target or "target kind". A node sensing no
  prey is drawn gently to a stale explore cell (exploration). The body still
  emerges from the summed base tension. `simulator/ilqr/residuals.py`
  `attract_residual`/`repel_residual`; the per-node sensing loop is in
  `Limb._move_ilqr`. Two refinements to avoid boundary artifacts: (1) the sense
  weight **smoothstep-ramps** to 0 at the window edge (`sense_ramp_band`) instead
  of a hard on/off (`_sense_ramp`), so nodes don't bunch at the radius; (2) flee
  aims at a point one `repel_step` **toward the body** (not the body centre), so
  the retraction magnitude comes from the threat-proximity weight and is
  **independent of how far the node is from the body** — far tips no longer get
  yanked in explosively.
- **Propulsion** (`octopus.propulsion_mode`, `simutil.PropulsionMode`): how the
  body's centre of mass translates. `INTERNAL` = the legacy summed-arm-tension
  drift (`_drift_body_by_tension`), non-physical (internal forces can't move a
  free body). `REACTION` (`_propel_body`, what `VIZ_ILQR`/analyzer use) =
  external-reaction: **crawl** (a tip that's *planted*, i.e. world-stationary
  under `crawl_plant_speed`, and pulling, hauls the body toward its grip, capped
  by `crawl_grip_limit`) + **jet** (a threat within `jet_trigger_radius` fires a
  siphon burst away from it, decaying by `jet_decay`, capped at
  `max_jet_velocity`). Rotation (summed torque → θ) is shared by both.
- **Behavior policy** (analyzer colour-coding): a per-node/limb/body state code
  is computed each frame (0 idle · 1 exploring · 2 chasing prey · 3
  avoiding/fleeing · 4 gripping/crawling) — `Limb.last_node_state`/
  `last_limb_state`, `Octopus.last_body_state` — recorded (schema v3) and shown
  as outline/centerline/head colours. **Agents** have their own policy too
  (`Agent.behavior`: 0 idle/wandering · 1 pursuing · 2 fleeing), set in
  `AgentGenerator._increment_*` and recorded (schema v5); the analyzer rings an
  actively-pursuing/fleeing agent bright and dims an idle one.
- **Agent camouflage-gating:** `PURSUIT_FLEE` threats now pursue only when the
  octopus is visible enough (`agents.visibility_threshold`); below it the octopus
  reads as hidden and agents wander (the reactive spring modes already scale
  continuously with visibility). So camouflage is protective.
- **Exploration** (`octo_ilqr_explore_enabled`, off by default): a
  least-RECENTLY-visited map, `Octopus.visit_recency` — a cell is **set to 1.0**
  when a sucker is on it (not incremented, so dwell time doesn't matter) and
  decays by `octo_ilqr_explore_decay` < 1 each frame, so its value is
  `decay ** frames_since_last_visit`. A node that senses no prey is gently drawn
  (`w_explore ≪ w_reach_terminal`) to a stale cell chosen **lexicographically**
  (`_node_explore_target`): first the least-recently-visited set (minimum
  recency in `explore_node_radius`, plus a threat penalty in the primary key),
  then the **closest** of that set — recency always beats distance, so a node
  never prefers a near recent cell to a far stale one. The chosen cell is
  recorded (`explore_cell`, schema v4) for the analyzer's hover box. Prey
  outranks it per node. See EXPLORATION_PLAN.md.
- **Camouflage:** each sucker matches the surface color beneath it,
  constrained to change ≤ `octo_max_hue_change` per step **per channel**. Full
  **RGB**: the surface grid is `(y, x, 3)` and each of `Color.r/g/b` matches its
  channel independently (grayscale surfaces are the special case r=g=b). The
  `SUCKER`/`LIMB` models are single-channel and are applied per channel.
  `world.background_image` loads an image as the surface;
  `world.surface_grayscale` picks grayscale vs colour random noise.
  `max_hue_change` is enforced in **both** paths: `NO_MODEL` clamps the step
  directly, and the `SUCKER` model's output is **re-clamped** to it at inference
  (the trained model bakes in its *training-time* cap, so this is what lets a
  smaller config value actually slow camouflage without retraining). It is
  propagated to every `Sucker` in `_refresh_sucker_locations` — a `Sucker(0,0)`
  built without config keeps a 0.25 fallback, so before this the knob silently
  did nothing. **Opt-in:** `training.sucker_hue_change_conditioned` trains the
  sucker net on `max_hue_change` as a 3rd input (over a sampled range), so one
  model honours any budget passed at inference (`DeltaColorLayer` reads it from
  input column 2; inference detects a 3-input model and skips the re-clamp).
- **Setting colors:** `octo.set_color(surf, inference_mode, model)` computes
  colors in parallel and applies them (fixed July 2026). The
  equivalent explicit form, used where the caller wants the matrix too:

  ```python
  color_matrix = octo.find_color(surf, inference_mode, model)
  for ix, l in enumerate(octo.limbs):
      l.force_color(color_matrix[ix])
  ```

- **ML modes** (`MLMode` enum): `NO_MODEL` (clamped heuristic step),
  `SUCKER` (MLP on (current color, surface color)), `LIMB` (dual-input
  model that also sees adjacent suckers via ragged tensors), `FULL`
  (placeholder, no implementation).
- **Trainer pattern:** `training/trainutil.Trainer` base; `SuckerTrainer`
  and `LimbTrainer` implement `datagen`, `data_format`, `train`,
  `inference`. Orchestrated by `model.py`, which selects the
  `TRAINING` profile. Both trainers take a single `Config`.
- **Loss:** `WeightedSumLoss = 0.95·ConstraintLoss + 0.05·MAE`. The
  constraint loss penalizes color changes beyond the 0.25 threshold and
  ignores ground truth; MAE pulls toward the surface color. This is why
  trained models "drift toward the target slowly."
- **Loaders:** `ModelLoader` / `DataLoader` (subclasses of `DefaultLoader`
  in `training/models/base_loader.py`) accept a path or an `MLMode` and
  expose `get_object()`. Pass Keras custom objects as a keyword:
  `ModelLoader(path, custom_objects={...})`.
- **Data-splitting utilities** live canonically in
  `training/data_utils.py` (train gets `1 - test_size` of samples);
  `util.py` re-exports them for backwards compatibility.

## Build & Run

```bash
# Visualizer (matplotlib; needs a GUI session; press a key in the window to start)
python visualizer/octo_viz.py            # or: bazel run //visualizer:octo_viz

# Data generation (standalone; writes training/datagen/sucker.pkl)
python octopus_ai/datagen.py        # or: bazel run //octopus_ai:datagen

# Training pipeline (behavior driven by the TRAINING profile; CFG at the top of the file)
python octopus_ai/model.py          # or: bazel run //octopus_ai:model

# Inference server (localhost:8080; API in ARCHITECTURE.md §7)
bazel run //inference_server:server        # or: cd inference_server && python server.py

# Headless record & replay: step N frames, write logs/runs/<run_id>.duckdb
python simulator/headless_runner.py --frames 120   # or: bazel run //simulator:headless_runner_bin -- --frames 120

# Analyzer server (Simulate + Playback), then open http://localhost:8765/ in a browser
bazel run //visualizer:websocket_server  # or: python visualizer/websocket_server.py; ws://localhost:8765
```

Always activate the venv first: `source .venv/bin/activate` (ARM-native
Python 3.12; recreate per TRAINING.md if TensorFlow won't import).

## Testing

```bash
python run_tests.py                 # all tests (preferred)
python run_tests.py --verbose
python run_tests.py --coverage
python run_tests.py --test test_kinematics.py
make test / make test-kinematics / make test-training / ...
```

Hermetic alternative via Bazel (uses the `.bazelrc` defaults — failing tests
print their output instead of hiding it in a log file):

```bash
bazel test //...                              # all test targets
bazel test //tests:test_kinematics            # one target
bazel test --config=debug //tests:test_kinematics   # stream output live
```

`//inference_server:test_server` is a real integration test — it boots the
Flask app and talks to `localhost:8080`, so it is tagged `local` (runs
unsandboxed to bind the port). It has a pre-existing logic failure
(`test_communications`) that reproduces under plain pytest too; it is not a
Bazel wiring issue.

## Linting & Formatting

```bash
make lint       # ruff check .
make format     # ruff format .
```

There is pre-existing lint debt in older modules and tests (whitespace,
unused imports); clean opportunistically, don't let it block work.

## Code Conventions

- TensorFlow tensors are used for kinematic state even in the simulator,
  not just training.
- Color values are floats in [0, 1]; ML normalization to [-1, 1] via
  `util.octo_norm` where used.
- Grid coordinates: `x_len × y_len` (default 15×15); `RandomSurface.grid`
  is indexed `[y][x]`.
- Sucker/limb inference is parallelized with `multiprocessing.pool.ThreadPool`
  + `imap_unordered`, carrying an index through and re-sorting after.
- Config access is attribute-style on a frozen `Config`; derive
  variants with `dataclasses.replace()`, never mutate.
- Mutable containers are initialized in `__init__`, never as class
  attributes (class-level lists were shared across instances and bit us
  in July 2026).
- Prefer `ValueError`/`TypeError` over bare `assert` for runtime validation
  (older code still asserts; migrate when touching it).
- Entry scripts are mostly top-to-bottom module code (not `main()`-wrapped);
  `visualizer/websocket_server.py` and `simulator/headless_runner.py` have
  proper `main()` guards. The two `visualizer/` Python entry points insert
  the repo root onto `sys.path` so they run from the repo root despite
  importing top-level modules. Don't import `model.py` from library
  code — it executes the pipeline at import.

## Gotchas for future Claude instances

1. `restore_data_from_disk` unpickles from `cfg.training_dataset_path`
   — make sure a dataset was actually
   generated (e.g. `python octopus_ai/datagen.py`) and postdates July 2026
   (older pickles were generated with a bug that pinned sucker state at
   0.5 because the color feedback loop never ran).
2. Inference server payload is `{"job_id": N, "data": {"c.r": x, "c_val.r": y}}` —
   not `"input"`.
3. `AgentGenerator` seeds numpy's global RNG from `rand_seed` on every
   construction — order of object creation affects random sequences.
4. `MovementMode` has four working limb modes: `RANDOM`, `LUMPED_SPRING`,
   `SPRING_CHAIN`, and `ILQR` (per-limb TensorFlow iLQR reach; each limb owns
   its own compiled `ArmController`, `simulator/ilqr/` + `Limb._move_ilqr`,
   MPC-style). Agents only implement `RANDOM`/`REACTIVE`. The active iLQR path
   is `solver.py` + `arm.py` + `residuals.py` (the cost library — new cost terms
   go there). The old `nodemesh.py` + `costs.py` gradient-relaxation prototype
   was **deleted** July 2026 (incompatible paradigm, never wired in). See
   ARCHITECTURE.md §4.5 and §11 for the compute-placement rationale. Note
   `PropulsionMode` (how the *body* translates: `INTERNAL`/`REACTION`) is a
   SEPARATE axis from `MovementMode` (how a *limb* moves) — see Key Concepts.
5. `InferenceLocation.REMOTE` exists but nothing routes inference to the
   server yet — wiring it up means giving `Sucker.find_color` (or a layer
   above it) an HTTP client path.
6. `InferenceQueue.clear_stale()` is deliberately not called by the
   watchdog: pending jobs are picked up within ~0.1 s, and auto-deleting
   queued jobs after 30 s could surprise clients. Re-enable deliberately
   if queue growth becomes a problem.
7. `MLMode.FULL` is a placeholder with no model, dataset, or trainer.
8. `training/models/` contains a stray backup file literally named
   `sucker.keras has pretty good results` — it's a Dec 2023 model, not
   docs. Safe to delete or rename.
9. Record & replay: `output.record_run` / `output.record_ilqr_history`
   **default off** (a fresh checkout writes no databases). DuckDB is
   single-writer-per-file, which is why there is one `.duckdb` per run — a
   completed run stays readable (server playback and external `duckdb`/pandas)
   while another run is being recorded. `run_store.py` opens each run
   read-only; the server never opens the *active* run's file (it's write-locked
   by the recorder), synthesizing that row from memory instead. Enabling
   `record_ilqr_history` has zero overhead when off but is the bulk of a run's
   disk (~20 MB at 120 frames vs ~2–3 MB without). `SCHEMA_VERSION` is 5;
   `RunStore` reads newer columns **defensively** (column-existence checks, not a
   version gate), so older runs still open — they just read back the added fields
   as defaults (`motor_state`/`body_state`/`agents.behavior` = 0/idle,
   `explore_cell` = none).
