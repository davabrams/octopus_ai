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
  single-arg handler API), flask, networkx

## Project Structure (abridged — full map in ARCHITECTURE.md §2)

```
octopus_ai/                  # repo root
├── octopus_ai/             # core package: config, profiles, shared utils, datagen/train entry points
│   ├── config_schema.py    # The typed Config dataclasses — source of truth
│   ├── config.py           # Profiles (DEFAULT/VIZ/DEBUG/TEST/DATAGEN/TRAINING) + flat<->nested converters + path maps
│   ├── util.py             # erase_all_logs, octo_norm + re-exports from training.data_utils
│   ├── datagen.py          # OctoDatagen class + standalone pickle-writing main
│   └── model.py            # datagen → train → save → inference/eval pipeline
├── visualizer/             # matplotlib viz (octo_viz.py) + browser viz (websocket_server.py, HTML/React frontends)
├── simulator/              # simutil (State/Agent/Color/enums), octopus/agent/surface generators, ilqr/
├── training/               # trainers (sucker, limb), losses, loaders, data_utils, saved .keras models
├── inference_server/       # Flask REST server (localhost:8080), job queue + watchdog
└── tests/                  # pytest suite (~2,400 lines; 136 tests)
```

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
- **Octopus model:** head at (x, y) → 8 `Limb`s → each limb has
  `limb_rows × limb_cols` `Sucker`s (default 16×2 = 32; 256 total).
- **Camouflage:** each sucker matches the binary surface color beneath it,
  constrained to change ≤ `octo_max_hue_change` (0.25) per step. Grayscale
  in practice — only `Color.r` is the signal; g/b mirror it.
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

# WebSocket visualization server (then open visualizer/octopus-visualizer.html in a browser)
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
  `visualizer/websocket_server.py` and `simulator/ilqr/nodemesh.py` have
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
   MPC-style). Agents only implement `RANDOM`/`REACTIVE`. The old
   `simulator/ilqr/nodemesh.py` + `costs.py` gradient-relaxation prototype is
   superseded by `solver.py` + `arm.py` and is not wired in. See
   ARCHITECTURE.md §4.5 and §11 for the compute-placement rationale.
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
