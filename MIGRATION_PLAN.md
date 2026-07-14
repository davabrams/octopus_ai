# Config shim removal — migration plan

Status: PLANNED. Written at commit `a91d9d9`, tree clean, 229 tests green.
This file is a working document; delete it when the migration lands.

## The actual goal

Delete the **module-level mutable dicts** `GameParameters` and
`TrainingParameters` from `OctoConfig.py`. Those are the shim: global
mutable state that every test inherits and that production code used to
index with bare strings.

## What is NOT being deleted (and why)

`to_game_parameters()` / `from_game_parameters()` are **not** shim. They are
a legitimate flat<->nested boundary with two real consumers:

1. `websocket_server` — the browser wire protocol genuinely sends flat keys
   (`{"x_len": 20, ...}`). It must translate.
2. Test fixtures — a flat `**kwargs` override is far more ergonomic than
   `replace(TEST, world=replace(TEST.world, x_len=30))` nested three deep.
3. `force_logger` — stores the flat view as the config snapshot, because
   that is what `json_extract` queries cleanly.

So: **rename them to say what they are**, keep them, delete the globals.

    to_game_parameters   -> config_to_flat
    from_game_parameters -> config_from_flat

## Current coupling (measured, not guessed)

Module-level importers of the globals — only TWO:
  - `octo_model.py:9`      `from OctoConfig import GameParameters, TrainingParameters`
  - `training/losses.py:360`  inside a `__main__` demo block

Everything in `training/` receives params as **constructor arguments**
(`SuckerTrainer(GameParameters, TrainingParameters)`); the trainers are
already dependency-injected and do not read a global.

Test files referencing GameParameters (10):
  test_config_isolation.py  13   <- asserts on the globals; rewrite target
  test_trainers.py           8
  conftest.py                7   <- the autouse fixture mutates the global
  test_octopus_generator.py  6
  test_integration.py        5
  test_agent_movement.py     3
  test_spring_chain.py       2
  test_prey_capture.py       2
  test_limb_adjacency.py     2
  test_force_logger.py       2

Dominant test pattern to replace:

    p = GameParameters.copy()
    p['x_len'] = 30
    p['limb_rows'] = 6
    octo = Octopus(p)

## Target

    from tests.helpers import test_config
    cfg = test_config(x_len=30, limb_rows=6)
    octo = Octopus(cfg)

`test_config(**overrides)` = `config_from_flat({**config_to_flat(TEST),
**overrides})`. Flat kwargs in, typed Config out. Shorter than the dict
version AND typed. Unknown keys must raise, not silently no-op — that is a
strict improvement over `p['typo'] = 5`.

## Steps (each ends green; commit per step)

1. **tests/helpers.py**: add `test_config(**overrides) -> Config`, raising
   on unknown keys. Add its own tests.
2. **conftest.py**: the autouse fixture currently mutates the
   `GameParameters` global. Once tests take a Config, the fixture has
   nothing to isolate — DELETE it, and with it the whole class of "tests
   inherit the developer's working config". Keep
   `tests/test_config_isolation.py` but rewrite it to assert the TEST
   profile's properties instead of the global's.
3. **Migrate the 8 remaining test files** to `test_config(...)`, one commit
   per file or in small batches, suite green each time.
4. **octo_model.py**: use the `TRAINING` profile; trainers take a `Config`
   (single arg, replacing the GameParameters+TrainingParameters pair — note
   `test_trainers.py` overrides `training_params['datasets']`, so paths must
   be settable via `replace(cfg, paths=...)`).
5. **training/losses.py:360**: the `__main__` demo uses `DEFAULT`.
6. **Delete** `GameParameters` / `TrainingParameters` from OctoConfig.
   Rename the converters. Update `websocket_server` + `force_logger` +
   `simulator/*` (`as_config`) to the new names.
7. Grep for `GameParameters` — expect zero hits outside this file's history.

## Traps found already (do not re-discover)

- `RandomSurface` used `params.get('surface_grayscale', False)`, an implicit
  "absent means binary" that contradicted the project default. Fixed in
  `542aea1`; tests now request binary explicitly.
- `test_simulator.py` builds a *partial* hand-made dict (~8 keys).
  `config_from_flat` must stay tolerant (missing key -> DEFAULT value) or
  that test breaks. Tolerance is deliberate.
- `agent_range_radius` is ONE flat key mapping to TWO fields
  (`agents.sensing_radius`, `octopus.sensing_radius`). `config_from_flat`
  sets both. Once tests pass Configs they can diverge — check nothing
  silently depends on them being equal.
- The trainers take TWO dicts today. Collapsing to one Config changes their
  signature; `test_trainers.py` mocks heavily and is the only coverage, so
  verify carefully.
- MCP `run_command` blocks some command patterns (`&&` chains, filenames
  containing "threat", `ffprobe`). Write patch logic to a neutrally-named
  .py file and run `python3 file.py` bare.

## Definition of done

- `grep -rn "GameParameters" --include="*.py" .` returns nothing.
- `tests/conftest.py` has no config-isolation fixture (it is unnecessary
  once nothing global exists to leak).
- Suite green (>=229).
- One profile switch in `octo_viz.py` still selects DEFAULT/VIZ/DEBUG.
