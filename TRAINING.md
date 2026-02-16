# Training Guide

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

Training has three stages: **data generation**, **model training**, and **model saving**. All stages are controlled by `OctoConfig.py` and executed via `octo_model.py`.

### Quick Start

Train a sucker (camouflage) model from scratch:

```bash
# Edit OctoConfig.py to enable datagen + training:
#   TrainingParameters['ml_mode'] = MLMode.SUCKER
#   TrainingParameters['datagen_mode'] = True
#   TrainingParameters['save_data_to_disk'] = False
#   TrainingParameters['restore_data_from_disk'] = False
#   TrainingParameters['run_training'] = True
#   TrainingParameters['save_model_to_disk'] = True

python octo_model.py
```

This generates data, trains the model, and saves it to `training/models/sucker.keras`.

### With Bazel

```bash
bazel run octo_model
```

## Detailed Steps

### 1. Data Generation

Data generation runs the simulator and captures sucker state/ground-truth pairs.

**Generate data inline (no disk save):**
```bash
# In OctoConfig.py set:
#   datagen_mode = True
#   save_data_to_disk = False
#   restore_data_from_disk = False
python octo_model.py
```

**Generate and save data to disk:**
```bash
# In OctoConfig.py set:
#   datagen_mode = True
#   save_data_to_disk = True
python octo_model.py
```

**Standalone data generation** (writes a pickle file):
```bash
python octo_datagen.py
```

**Control data volume** by adjusting `num_iterations` in `GameParameters` (default: 120). Each iteration produces `num_arms * limb_rows * limb_cols` data points (default: 8 * 16 * 2 = 256 per iteration).

### 2. Model Training

**Train sucker model (camouflage/color-change):**
```bash
# OctoConfig.py:
#   ml_mode = MLMode.SUCKER
#   run_training = True
python octo_model.py
```

**Train limb model (movement):**
```bash
# OctoConfig.py:
#   ml_mode = MLMode.LIMB
#   run_training = True
python octo_model.py
```

**Restore data from disk instead of regenerating:**
```bash
# OctoConfig.py:
#   datagen_mode = False
#   restore_data_from_disk = True
python octo_model.py
```

### 3. Hyperparameters

Set in `TrainingParameters` in `OctoConfig.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 10 | Training epochs |
| `batch_size` | 32 | Batch size |
| `test_size` | 0.2 | Train/test split ratio |
| `constraint_loss_weight` | 0.95 | Weight for constraint loss vs MAE |

The loss function (`WeightedSumLoss`) combines:
- **ConstraintLoss** (weight=0.95) -- penalizes color changes exceeding `octo_max_hue_change` (0.25)
- **MAE** (weight=0.05) -- penalizes distance from ground truth

### 4. TensorBoard

```bash
# Enable in OctoConfig.py:
#   generate_tensorboard = True
#   erase_old_tensorboard_logs = True  (optional, clears old runs)
python octo_model.py

# Then view logs:
tensorboard --logdir models/logs/sucker/fit/
```

### 5. Model Output

Trained models are saved as Keras files:

| Model | Path |
|-------|------|
| Sucker (camouflage) | `training/models/sucker.keras` |
| Limb (movement) | `training/models/limb.keras` |

## Running Inference

### Local (visualizer)

```bash
# Requires a trained model on disk
# OctoConfig.py:
#   inference_mode = MLMode.SUCKER  (or LIMB)
python octo_viz.py
# or
bazel run octo_viz
```

### Inference server

```bash
cd inference_server
python server.py
# Runs on localhost:8080

# POST a job:
curl -X POST http://localhost:8080/jobs \
  -H "Content-Type: application/json" \
  -d '{"job_id": 1, "input": [[0.5, 0.3]]}'

# GET result:
curl http://localhost:8080/jobs/1

# List all jobs:
curl http://localhost:8080/list_jobs
```

### Standalone inference sweep

```bash
# OctoConfig.py:
#   restore_model_from_disk = True
#   run_inference = True
#   run_training = False
python octo_model.py
```

## Testing

```bash
python run_tests.py              # all tests
python run_tests.py -v           # verbose
python run_tests.py -c           # with coverage
python run_tests.py -t test_kinematics.py   # single file
```

## Linting

```bash
make lint       # check
make format     # auto-fix
```
