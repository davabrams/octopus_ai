# CLAUDE.md

## Project Overview

Octopus AI is a simulation and machine learning project that models octopus behavior — locomotion, limb movement, and sucker-based camouflage (color change). It includes a physics simulator, ML training pipelines, an inference server, a WebSocket-based visualization server, and a React/HTML visualizer frontend.

## Tech Stack

- **Language:** Python 3.10+
- **ML Framework:** TensorFlow / Keras
- **Build System:** Bazel (with `MODULE.bazel` / Bzlmod)
- **Testing:** pytest (primary), unittest, Bazel test
- **Linting:** ruff (configured in `pyproject.toml`)
- **Visualization:** matplotlib (local), WebSocket server + HTML/React frontend
- **Key Libraries:** numpy, tensorflow, matplotlib, websockets, asyncio, flask, networkx, seaborn
- **Dependency Management:** `pyproject.toml`

## Project Structure

```
octopus_ai/
├── pyproject.toml             # Dependencies, ruff config, pytest config
├── OctoConfig.py              # Typed config: GameConfig + TrainingConfig dataclasses
├── octo_viz.py                # Scenario synthesizer and visualizer (main())
├── octo_datagen.py            # Training data generator (main())
├── octo_model.py              # Model trainer entry point (main())
├── util.py                    # ML utilities (log cleanup, normalization, re-exports)
├── websocket_server.py        # WebSocket server for browser visualization
├── websocket-integration.py   # WebSocket integration helpers
├── octopus-visualizer.html    # HTML visualizer frontend
├── octopus-ai-visualizer.tsx  # React visualizer component
├── simulator/
│   ├── simutil.py             # Core types: State, Agent, Color, enums (MLMode, MovementMode, etc.)
│   ├── octopus_generator.py   # Octopus model (head, limbs, suckers)
│   ├── agent_generator.py     # Prey/threat agent generator
│   ├── surface_generator.py   # Random background surface patterns
│   └── ilqr/                  # iLQR control for limb movement
├── training/
│   ├── trainutil.py           # Trainer ABC (abstract base class)
│   ├── sucker.py              # Sucker color-change ML model + training
│   ├── limb.py                # Limb movement ML model + training
│   ├── losses.py              # Custom loss functions
│   ├── data_utils.py          # Train/test split, dataset conversion utilities
│   ├── datagen/               # Data generation and DataLoader
│   └── models/
│       ├── base_loader.py     # DefaultLoader ABC for models and data
│       └── model_loader.py    # Keras ModelLoader
├── inference_server/
│   ├── server.py              # Flask/HTTP inference server
│   ├── model_inference.py     # Model loading (lazy) and inference logic
│   └── test_server.py         # Server tests
├── tests/                     # Test suite (pytest)
│   ├── test_octopus_generator.py
│   ├── test_kinematics.py
│   ├── test_integration.py
│   ├── test_simulator.py
│   ├── test_trainers.py
│   ├── test_training_losses.py
│   ├── test_inference_server.py
│   └── test_utilities.py
├── BUILD                      # Root Bazel build file
├── MODULE.bazel               # Bazel module definition
└── Makefile                   # Make targets for testing, linting, formatting
```

## Configuration

All simulation and training parameters are typed dataclasses in `OctoConfig.py`:

- `GameConfig` — grid size, octopus arm count/physics, agent behavior, inference mode
- `TrainingConfig` — ML mode, epochs, batch size, loss weights, tensorboard, model save/restore
- `GameParameters` / `TrainingParameters` — default instances for convenience

Access config fields with attribute syntax: `params.x_len`, `params.octo_num_arms`, etc.

## Key Concepts

- **State:** 6-DOF kinematic primitive `[x, y, θ, vx, vy, ω]` stored as TensorFlow tensors (`simulator/simutil.py`).
- **Octopus model:** Head (CenterPoint) → 8 limbs → each limb has a grid of suckers (rows × cols) with color (RGB).
- **ML modes** (`MLMode` enum): `NO_MODEL` (heuristic), `SUCKER` (color camouflage), `LIMB` (movement), `FULL` (combined).
- **Inference:** Can run locally or on a remote inference server (`InferenceLocation` enum in config).
- **Agents:** Prey and threats that the octopus reacts to (`AgentType` enum).
- **Trainer:** Abstract base class in `training/trainutil.py`. Subclasses (`SuckerTrainer`, `LimbTrainer`) must implement `datagen`, `data_format`, `train`, `inference`.

## Build & Run

```bash
# Visualizer
bazel run octo_viz

# Data generation
bazel run octo_datagen

# Model training
bazel run octo_model

# WebSocket visualization server
bazel run websocket_server
```

## Testing

```bash
# Run all tests (preferred)
python run_tests.py

# Alternatives
make test
./test.sh

# Verbose / coverage
python run_tests.py --verbose
python run_tests.py --coverage

# Individual test files
make test-kinematics
make test-training
python run_tests.py --test test_octopus_generator.py
```

## Linting & Formatting

```bash
make lint        # Check with ruff
make format      # Auto-format with ruff
```

Ruff is configured in `pyproject.toml` with rules for pycodestyle, pyflakes, isort, pyupgrade, bugbear, and simplify.

## Code Conventions

- TensorFlow tensors are used throughout the simulator for kinematic state (not just in training).
- Color values are floats in [0, 1] range (RGB).
- Coordinates use a grid system defined by `x_len` × `y_len` in config.
- Angles are in radians, wrapped to [0, 2π).
- The project uses Bazel for builds but pytest for most testing workflows.
- Use proper exceptions (`ValueError`, `TypeError`) instead of `assert` for runtime validation.
- Thread pools use `concurrent.futures.ThreadPoolExecutor` as context managers.
- Entry point scripts (`octo_viz`, `octo_model`, `octo_datagen`) wrap logic in `main()` behind `if __name__ == "__main__"`.
