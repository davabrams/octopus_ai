# octopus_ai

A simulation + machine-learning sandbox for octopus camouflage. It models an
octopus (head → 8 limbs → a grid of suckers per limb) crawling around a 2D
surface among prey/threat agents, and trains ML models that let the suckers
**camouflage** — matching the surface color beneath them, one constrained
step at a time.

More rambling at https://davabrams.wordpress.com/

## What's in here

- **Simulator** — an octopus (head/limbs/suckers), a random background
  surface, and attractor/repeller agents (prey and threats).
- **Datagen** — rolls the simulator forward and records
  `(sucker color, surface color)` training pairs.
- **Training + inference** — Keras models for sucker color change
  (camouflage), plus TensorBoard output.
- **Inference server** — a Flask REST server so models can run off-box.
- **Visualization** — a local matplotlib viewer and a browser (WebSocket)
  viewer.

## Why

I don't know, but I can't stop.

---

## Setup

[Bazel](https://bazel.build) is the primary build/run/test interface. It's
the polyglot bet: simulator or training pieces can move to C++/Rust/JS behind
the same `bazel run` commands later without changing how you drive the repo.

Today all the code is Python, and **Bazel borrows your venv's interpreter and
packages** — it does *not* manage the Python deps — so you create the venv
first either way:

```bash
/opt/homebrew/bin/python3.12 -m venv .venv   # any Python ≥3.10 works
source .venv/bin/activate
pip install -e ".[dev]"

# sanity check
python -c "import tensorflow as tf; print(tf.__version__)"
```

Always `source .venv/bin/activate` before running anything — both `bazel`
and the raw `python` fallbacks use that interpreter. Full environment notes
(including Apple-Silicon TensorFlow gotchas) live in [TRAINING.md](TRAINING.md).

---

## Configuration

This is the knob you'll turn most, so it's worth 60 seconds.

Config is a tree of **frozen dataclasses** — the schema is the source of
truth in [octopus_ai/config_schema.py](octopus_ai/config_schema.py), and
[octopus_ai/config.py](octopus_ai/config.py) builds named **profiles** from
it. You never edit shared mutable state; you *select a profile* (or derive a
variant) and the entry scripts read it.

### Profiles

| Profile | For |
|---------|-----|
| `DEFAULT` | shipped baseline; writes nothing to disk |
| `VIZ` | watching a run — force arrows on, disk quiet |
| `DEBUG` | everything on — arrows + SQLite force log + PNG/MP4 capture |
| `TEST` | deterministic, side-effect free, needs no model on disk |
| `DATAGEN` | data generation (saves a dataset, no rendering) |
| `TRAINING` | the training pipeline |

### How to change the config

Each entry script picks a profile on a single line near the top, e.g.
`CFG = DEFAULT` in [octopus_ai/model.py](octopus_ai/model.py) and
[visualizer/octo_viz.py](visualizer/octo_viz.py). To change behavior you
either swap that profile or derive a one-off variant with
`dataclasses.replace()` (configs are frozen — in-place mutation raises):

```python
from dataclasses import replace
from octopus_ai.config import VIZ
from simulator.simutil import MLMode

# Watch it run, but drive the suckers with the trained SUCKER model:
CFG = replace(VIZ, inference=replace(VIZ.inference,
                                     mode=MLMode.SUCKER,
                                     model=MLMode.SUCKER))

# Or a quick training experiment: 5 epochs instead of 50
CFG = replace(TRAINING, training=replace(TRAINING.training, epochs=5))
```

Access is attribute-style everywhere: `cfg.world.x_len`,
`cfg.octopus.num_arms`, `cfg.training.epochs`.

### Knobs worth knowing

| Where | Knob | Does what |
|-------|------|-----------|
| `cfg.run` | `num_iterations` | frames to simulate (120; `-1` = forever) |
| | `rand_seed` | RNG seed for reproducible runs |
| `cfg.world` | `x_len` / `y_len` | surface grid size (15×15) |
| | `surface_grayscale` | grayscale surface vs. classic binary 0/1 |
| `cfg.agents` | `count` | number of prey/threat agents (5) |
| | `movement_mode` | only `RANDOM` works today (see Limitations) |
| `cfg.octopus` | `num_arms` | limbs (8) |
| `cfg.octopus.limb` | `rows` / `cols` | suckers per limb (16×2 = 32) |
| `cfg.octopus.sucker` | `max_hue_change` | max color change per step (0.25) |
| | `adjacency_radius` | neighbor distance for the LIMB model |
| `cfg.inference` | `mode` | how colors are computed: `NO_MODEL` (heuristic), `SUCKER`, `LIMB` |
| | `model` | which trained model to load |
| `cfg.datagen` | `datagen_mode` | generate fresh data this run |
| | `save_to_disk` / `restore_from_disk` | write / reuse the dataset pickle |
| | `write_format` | `SUCKER` or `LIMB` (LIMB also captures adjacents) |
| `cfg.training` | `ml_mode` | `SUCKER` or `LIMB` |
| | `epochs` / `batch_size` / `test_size` | standard hyperparams |
| | `run_training` / `run_inference` / `run_eval` | which pipeline stages fire |

> First run with no trained model? Use the heuristic: set
> `inference.mode = MLMode.NO_MODEL`. `DEFAULT` expects a `SUCKER` model on
> disk, so a fresh checkout should switch to `NO_MODEL` or train one first.

---

## Running things

Commands lead with Bazel; the raw `python …` form is the fallback and stays
interchangeable. Activate the venv first. `bazel run` reads and writes
artifacts (dataset pickles, saved models) in the repo's `training/` tree —
not the sandbox — so a dataset generated one way is picked up by the other.
(`bazel test` stays hermetic and touches neither.)

### Visualize (matplotlib)

Watch the octopus crawl and camouflage in a native window.

```bash
bazel run //visualizer:octo_viz      # or: python visualizer/octo_viz.py
```

Needs a GUI session; click the window and press a key to start. The green
number is the **visibility score** (mean squared color error — lower =
better camouflage). Pick the profile via `CFG` at the top of the file
(`VIZ` for arrows, `DEBUG` to also record a video).

### Visualize (browser)

```bash
bazel run //visualizer:websocket_server    # or: python visualizer/websocket_server.py
# then open visualizer/octopus-visualizer.html and click Connect  (ws://localhost:8765)
```

The browser UI has live config sliders (it speaks the flat config form), so
this is the easiest way to poke at parameters without editing code.

### Generate training data

```bash
bazel run //octopus_ai:datagen       # or: python octopus_ai/datagen.py
```

Rolls the simulator and writes `training/datagen/sucker.pkl`. The integrated
pipeline below can also generate data inline as its first stage.

### Train a model

Training is driven by the `TRAINING` profile in
[octopus_ai/model.py](octopus_ai/model.py), which chains **datagen → train →
save** (and optionally inference/eval):

```bash
bazel run //octopus_ai:model         # or: python octopus_ai/model.py
```

Common variations (set on `CFG` at the top of the file):

- **Sucker vs. limb model** — `training.ml_mode = MLMode.SUCKER` (default) or
  `MLMode.LIMB`. LIMB also needs `datagen.write_format = MLMode.LIMB` so
  adjacent-sucker context is captured. (LIMB training is experimental.)
- **Train from a saved dataset** instead of regenerating — set
  `datagen.datagen_mode = False`, `datagen.restore_from_disk = True`.
- **Inference sweep / eval** — flip `training.run_inference` /
  `training.run_eval` on.

The saved model goes to `training/models/{sucker,limb}.keras`. The
step-by-step walkthrough is in [TRAINING.md](TRAINING.md).

### TensorBoard

Training writes logs when `training.generate_tensorboard` is on:

```bash
tensorboard --logdir models/logs/sucker/fit/    # or .../limb/fit/
```

### Inference server

A standalone Flask server that loads the sucker model and answers prediction
requests (so inference can run off-box).

```bash
bazel run //inference_server:server        # http://localhost:8080
# fallback: cd inference_server && python server.py

# POST a job — c.r = sucker's current color, c_val.r = surface color under it
curl -X POST http://localhost:8080/jobs \
  -H "Content-Type: application/json" \
  -d '{"job_id": 1, "data": {"c.r": 0.5, "c_val.r": 1.0}}'

curl http://localhost:8080/jobs/1     # fetch result
```

Full API in [ARCHITECTURE.md](ARCHITECTURE.md) §7.

---

## Testing

Each test file is its own Bazel target under `//tests`:

```bash
bazel test //tests/...               # all tests
bazel test //tests:test_kinematics   # a single file
```

Or the Python runner (adds coverage and single-file selection conveniences):

```bash
python run_tests.py                  # all tests
python run_tests.py --coverage       # with coverage
python run_tests.py --test test_kinematics.py
python run_tests.py --runner bazel   # drive Bazel via the runner
```

Per-file coverage notes are in [tests/README.md](tests/README.md).

## Linting

```bash
make lint       # ruff check .
make format     # ruff format .
```

---

## Current limitations

- Only `MovementMode.RANDOM` is implemented; the attract/repel / spring
  movement modes are stubbed.
- The `LIMB` training pipeline is experimental; `SUCKER` is the solid path.
- `MLMode.FULL` is a placeholder — no model, dataset, or trainer yet.
- Inference always runs locally; the inference server exists but nothing
  routes the simulator to it automatically.

## Going deeper

| Doc | What's in it |
|-----|--------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | module-by-module reference, data flow, APIs |
| [TRAINING.md](TRAINING.md) | detailed training/inference workflows + env setup |
| [CLAUDE.md](CLAUDE.md) | conventions, commands, and known gotchas |
| [tests/README.md](tests/README.md) | per-file test coverage |
