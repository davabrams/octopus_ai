# Training Guide

(See `ARCHITECTURE.md` for how these pieces work internally, and
CLAUDE.md's gotchas for the short list of known limitations.)

## Prerequisites

Create an ARM-native Python venv (required on Apple Silicon):

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Verify TensorFlow loads:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

## Training Pipeline

Training has three stages: **data generation**, **model training**, and
**model saving**. All stages are controlled by the profile `octo_model.py`
selects (`CFG = TRAINING`, from `OctoConfig.py`) and executed by running it.

### Quick Start

Train a sucker (camouflage) model from scratch:

The `TRAINING` profile already enables datagen + training + model saving,
so this is just:

```bash
python octo_model.py
```

To vary it, edit `CFG` at the top of `octo_model.py` rather than a shared
dict — configs are frozen, so derive:

```python
CFG = replace(TRAINING,
              training=replace(TRAINING.training,
                               ml_mode=MLMode.SUCKER, epochs=5))
```

This generates data in memory, trains the model, and saves it to
`training/models/sucker.keras`.

### With Bazel

```bash
bazel run octo_model
```

## Detailed Steps

### 1. Data Generation

Data generation runs the simulator and captures sucker state/ground-truth
pairs.

**Generate data inline (no disk save):**
```bash
# In OctoConfig.py set:
#   datagen_mode = True
#   save_data_to_disk = False
#   restore_data_from_disk = False
python octo_model.py
```

**Save a dataset to disk / reuse a saved dataset:** the path comes from
`cfg.training_dataset_path` (→ `default_datasets` in `OctoConfig.py`,
resolving to `training/datagen/{sucker,limb}.pkl`).

```bash
# Save while generating:
#   datagen_mode = True
#   save_data_to_disk = True

# Later, train from the saved pickle without re-running the simulator:
#   datagen_mode = False
#   restore_data_from_disk = True
python octo_model.py
```

The standalone generator also works and pickles to the same sucker path:

```bash
python octo_datagen.py
```

**⚠️ Regenerate old datasets.** Any pickle generated before July 2026 was
produced while `Octopus.set_color` was a no-op: the recorded
"current color" state is pinned at the 0.5 default instead of evolving
between iterations. `training/datagen/sucker.pkl` predates the fix —
regenerate before training anything you care about.

**Control data volume** by adjusting `cfg.run.num_iterations`
(default: 120). Each iteration produces `num_arms × limb_rows × limb_cols`
data points (default: 8 × 16 × 2 = 256 per iteration → 30,720 total).

### 2. Model Training

**Train sucker model (camouflage/color-change):**
```bash
# OctoConfig.py:
#   ml_mode = MLMode.SUCKER
#   run_training = True
python octo_model.py
```

**Train limb model (movement/adjacency-aware color):**
```bash
# OctoConfig.py:
#   ml_mode = MLMode.LIMB
#   datagen.write_format = MLMode.LIMB   # so adjacents are captured
#   run_training = True
python octo_model.py
```
The limb pipeline is experimental: the saved `limb.keras` artifact dates
from Apr 2024 and the RNN training loop hasn't been re-verified under
current Keras.

### 3. Hyperparameters

Set under `cfg.training` in `config_schema.py` (one source — these used to
be duplicated across two dicts with nothing keeping them in sync):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 50 | Training epochs |
| `batch_size` | 32 | Batch size |
| `test_size` | 0.2 | Test fraction (train gets the remaining 80%) |
| `constraint_loss_weight` | 0.95 | Weight for constraint loss vs MAE |

The loss function (`WeightedSumLoss`) combines:
- **ConstraintLoss** (weight=0.95) — penalizes color changes exceeding
  `octo_max_hue_change` (0.25); compares prediction to the *previous* color,
  ignores ground truth
- **MAE** (weight=0.05) — penalizes distance from ground truth (surface color)

Optimizer: SGD, lr=1e-3, custom `GradientTape` loop (not `model.fit`).

### 4. TensorBoard

```bash
# Enable in OctoConfig.py:
#   generate_tensorboard = True
#   erase_old_tensorboard_logs = True  (optional, clears old runs)
python octo_model.py

# Then view logs:
tensorboard --logdir models/logs/sucker/fit/    # or models/logs/limb/fit/
```

### 5. Model Output

Trained models are saved as Keras files:

| Model | Path |
|-------|------|
| Sucker (camouflage) | `training/models/sucker.keras` |
| Limb (movement) | `training/models/limb.keras` |

(`training/models/` also contains a stray Dec-2023 backup whose filename is
a note-to-self; ignore it.)

## Running Inference

### Local (visualizer)

```bash
# Requires a trained model on disk
# Set CFG at the top of octo_viz.py, e.g.
#   CFG = replace(VIZ, inference=replace(VIZ.inference,
#                                        mode=MLMode.SUCKER,
#                                        model=MLMode.SUCKER))
#   mode  = how colors are computed; model = which .keras to load
#   (resolved via CFG.inference_model_path)
python octo_viz.py
# or
bazel run octo_viz
```
Click into the matplotlib window and press a key to start the loop. The
green number is the visibility score (mean squared color error; lower =
better camouflage).

### Browser (WebSocket visualizer)

```bash
python websocket_server.py     # ws://localhost:8765
# then open octopus-visualizer.html in a browser and click Connect
```
Uses the heuristic by default; set `inference.mode` on the server's
profile to drive it with a trained model (falls back to the heuristic if the model
can't load).

### Inference server

```bash
cd inference_server        # must run from this directory (sys.path tricks)
python server.py
# Runs on localhost:8080; loads training/models/sucker.keras at startup

# POST a job — payload is {"job_id", "data": {"c.r", "c_val.r"}}:
#   c.r     = sucker's current color, c_val.r = surface color under it
curl -X POST http://localhost:8080/jobs \
  -H "Content-Type: application/json" \
  -d '{"job_id": 1, "data": {"c.r": 0.5, "c_val.r": 1.0}}'

# GET result/status:
curl http://localhost:8080/jobs/1

# List all jobs:
curl http://localhost:8080/list_jobs

# Drain completed jobs:
curl -X POST http://localhost:8080/collect_and_clear
```

Full endpoint reference: `ARCHITECTURE.md` §7. The simulator does not yet
route inference to this server automatically (`InferenceLocation.REMOTE`
is a distinct enum value but nothing uses it yet).

### Standalone inference sweep

```bash
# OctoConfig.py:
#   restore_model_from_disk = True
#   run_inference = True
#   run_training = False
python octo_model.py
```
The sweep renders a 5×5 seaborn heatmap of predictions over (previous
color, surface color); a healthy sucker model shows values stepping from
the previous color toward the surface color by ≤ ~0.25.

## Testing

```bash
python run_tests.py              # all tests
python run_tests.py -v           # verbose
python run_tests.py -c           # with coverage
python run_tests.py -t test_kinematics.py   # single file
```

## Linting

```bash
make lint       # ruff check .
make format     # ruff format .
```
