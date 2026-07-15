"""
Octopus game and ML parameters.

The typed schema in config_schema.py is the source of truth. This module
builds the concrete profiles and converts between a Config and its flat
dict form.

Pick a profile (DEFAULT, VIZ, DEBUG, TEST, DATAGEN, TRAINING) and derive
variants with dataclasses.replace(). There is no module-level params dict
to edit: that was the shim, and it is gone.

config_to_flat / config_from_flat are NOT a shim. They are the boundary
between the nested Config and the three places that legitimately speak
flat key/value pairs:
  - the browser wire protocol (websocket_server.py), which sends and
    receives {"x_len": 20, ...}
  - the force-log config snapshot (simulator/force_logger.py), where flat
    keys are what SQLite's json_extract queries most easily
  - test fixtures (tests/helpers.py), which take flat overrides so a test
    can say make_config(x_len=30) instead of a nest of replace() calls
"""
import os
from dataclasses import replace

from octopus_ai.config_schema import (  # noqa: F401  (re-exported for convenience)
    AgentConfig,
    Config,
    DatagenConfig,
    InferenceConfig,
    LimbConfig,
    LumpedSpringConfig,
    OctopusConfig,
    OutputConfig,
    PathsConfig,
    RandomDriftConfig,
    RunConfig,
    SpringChainConfig,
    SuckerConfig,
    TrainingConfig,
    WorldConfig,
)
from simulator.simutil import MLMode, InferenceLocation, MovementMode

# Under `bazel run`, __file__ points into the sandboxed runfiles tree, so
# artifact paths (datasets, saved models) would resolve there instead of the
# source tree. Bazel sets BUILD_WORKSPACE_DIRECTORY to the repo root for
# `bazel run`; honor it so datagen/training read and write the real
# training/ dir. It is unset for plain `python` and for `bazel test` (which
# should stay hermetic), where we fall back to this file's location — the repo
# root is two levels up (octopus_ai/ is the parent, its parent is the root).
ROOT_DIR = (
    os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

default_models = {
    MLMode.SUCKER: os.path.join(ROOT_DIR, 'training/models/sucker.keras'),
    MLMode.LIMB: os.path.join(ROOT_DIR, 'training/models/limb.keras'),
    MLMode.FULL: None,  # Placeholder for future full model
    MLMode.NO_MODEL: None
}

default_datasets = {
    MLMode.SUCKER: os.path.join(ROOT_DIR, 'training/datagen/sucker.pkl'),
    MLMode.LIMB: os.path.join(ROOT_DIR, 'training/datagen/limb.pkl'),
    MLMode.FULL: None,  # Placeholder for future full model
    MLMode.NO_MODEL: None
}

DEFAULT_PATHS = PathsConfig(
    model_paths=default_models,
    dataset_paths=default_datasets,
)


# =========================================================== PROFILES ===
# Profiles are derived with dataclasses.replace(). This is what the config
# reorg buys: "the settings I use to watch it run" is now a named thing you
# select, not working state you edit and have to remember to revert before
# committing.

DEFAULT = Config(paths=DEFAULT_PATHS)

# Watching the simulation: draw the force arrows, keep the disk quiet.
VIZ = replace(
    DEFAULT,
    output=replace(DEFAULT.output, show_forces=True),
)

# Everything on: force arrows, the SQLite force log, and PNG frames stitched
# to an MP4. This is the state that used to live (and get committed) in the
# flat dict by accident.
DEBUG = replace(
    DEFAULT,
    output=replace(
        DEFAULT.output,
        debug_mode=True,
        show_forces=True,
        log_forces=True,
        save_images=True,
    ),
)

# Deterministic and side-effect free. Nothing written to disk, no dependence
# on a trained model existing, and RANDOM movement (the only mode where
# move() works without an agent). tests/conftest.py applies this.
TEST = replace(
    DEFAULT,
    output=OutputConfig(),  # every flag already defaults off
    inference=replace(
        DEFAULT.inference,
        mode=MLMode.NO_MODEL,
        model=MLMode.NO_MODEL,
    ),
    agents=replace(DEFAULT.agents, movement_mode=MovementMode.RANDOM),
    octopus=replace(
        DEFAULT.octopus,
        movement_mode=MovementMode.RANDOM,
        limb=replace(DEFAULT.octopus.limb,
                     movement_mode=MovementMode.RANDOM),
    ),
)

# Generating training data: no rendering, no force logging.
DATAGEN = replace(
    DEFAULT,
    output=OutputConfig(),
    datagen=replace(DEFAULT.datagen, save_to_disk=True),
)

# Training a model: the simulator only matters insofar as datagen needs it.
TRAINING = replace(
    DEFAULT,
    output=OutputConfig(),
    training=replace(DEFAULT.training, run_training=True),
)


# ==================================================== FLAT CONVERSION ===
# The flat key names are the ones the browser and the force log already
# speak, so they are preserved verbatim. The mapping doubles as the
# old -> new correspondence for anything still reading the flat form.
#
# Note on agent_range_radius: it used to serve two distinct concepts - how
# far an AGENT senses the octopus, and how far an ARM senses agents. The
# schema splits them (agents.sensing_radius / octopus.sensing_radius), but a
# single flat key cannot express the split, so the flat view emits the
# agent's value and both readers see it. Anything that needs them to differ
# has to say so on the Config.

def config_to_flat(cfg: Config) -> dict:
    """Flat dict view of a Config.

    Lossy by design: the flat surface is the browser's vocabulary, not the
    schema's. Paths and the training run flags have no flat key.
    """
    return {
        # General game parameters
        'num_iterations': cfg.run.num_iterations,
        'x_len': cfg.world.x_len,
        'y_len': cfg.world.y_len,
        'surface_grayscale': cfg.world.surface_grayscale,
        'rand_seed': cfg.run.rand_seed,
        'debug_mode': cfg.output.debug_mode,
        'log_forces': cfg.output.log_forces,
        'show_forces': cfg.output.show_forces,
        'save_images': cfg.output.save_images,
        'adjacency_radius': cfg.octopus.sucker.adjacency_radius,
        'inference_location': cfg.inference.location,
        'inference_mode': cfg.inference.mode,
        'inference_model': cfg.inference.model,
        'datagen_data_write_format': cfg.datagen.write_format,

        # Agent parameters
        'agent_number_of_agents': cfg.agents.count,
        'agent_max_velocity': cfg.agents.max_velocity,
        'agent_max_theta': cfg.agents.max_theta,
        'agent_movement_mode': cfg.agents.movement_mode,
        'agent_range_radius': cfg.agents.sensing_radius,  # see note above
        'agent_prey_capture_radius': cfg.agents.prey_capture_radius,
        'agent_respawn_captured_prey': cfg.agents.respawn_captured_prey,

        # Octopus parameters
        'octo_max_body_velocity': cfg.octopus.max_body_velocity,
        'octo_max_arm_theta': cfg.octopus.limb.random.max_arm_theta,
        'octo_max_arm_reach_theta':
            cfg.octopus.limb.lumped.max_arm_reach_theta,
        'octo_max_limb_offset': cfg.octopus.limb.lumped.max_limb_offset,
        'octo_arm_stiffness': cfg.octopus.limb.lumped.arm_stiffness,
        'octo_arm_rest_fraction': cfg.octopus.limb.lumped.arm_rest_fraction,
        'octo_chain_spring_k': cfg.octopus.limb.chain.spring_k,
        'octo_chain_agent_k': cfg.octopus.limb.chain.agent_k,
        'octo_chain_move_k': cfg.octopus.limb.chain.move_k,
        'octo_num_arms': cfg.octopus.num_arms,
        'octo_max_sucker_distance': cfg.octopus.limb.max_sucker_distance,
        'octo_min_sucker_distance': cfg.octopus.limb.min_sucker_distance,
        'octo_movement_mode': cfg.octopus.movement_mode,
        'octo_threading': cfg.run.threading,

        # Limb parameters
        'limb_rows': cfg.octopus.limb.rows,
        'limb_cols': cfg.octopus.limb.cols,
        'limb_movement_mode': cfg.octopus.limb.movement_mode,

        # Sucker parameters
        'octo_max_hue_change': cfg.octopus.sucker.max_hue_change,
        'sucker_delta_model': cfg.training.sucker_delta_model,
        'datagen_randomize_colors_interval':
            cfg.datagen.randomize_colors_interval,

        # Training hyperparams. These used to be duplicated across two
        # flat dicts with nothing syncing them; cfg.training is now the
        # single source.
        'test_size': cfg.training.test_size,
        'epochs': cfg.training.epochs,
        'batch_size': cfg.training.batch_size,
        'constraint_loss_weight': cfg.training.constraint_loss_weight,
    }


def config_from_flat(d: dict) -> Config:
    """Build a Config from a flat dict.

    The inverse of config_to_flat, and deliberately TOLERANT: any key the
    dict omits falls back to the DEFAULT profile. That is what lets
    partially-specified dicts (e.g. the hand-built one in
    tests/test_simulator.py, or a browser sending only what it changed)
    keep working - previously, adding a parameter broke them with a
    KeyError.

    Note agent_range_radius: one flat key, two concepts. It sets BOTH
    agents.sensing_radius and octopus.sensing_radius, preserving the
    pre-split behaviour exactly for anything still passing a dict.
    """
    g = d.get
    D = DEFAULT
    sensing = g('agent_range_radius', D.agents.sensing_radius)

    return Config(
        run=RunConfig(
            num_iterations=g('num_iterations', D.run.num_iterations),
            rand_seed=g('rand_seed', D.run.rand_seed),
            threading=g('octo_threading', D.run.threading),
        ),
        world=WorldConfig(
            x_len=g('x_len', D.world.x_len),
            y_len=g('y_len', D.world.y_len),
            surface_grayscale=g('surface_grayscale',
                                D.world.surface_grayscale),
        ),
        agents=AgentConfig(
            count=g('agent_number_of_agents', D.agents.count),
            max_velocity=g('agent_max_velocity', D.agents.max_velocity),
            max_theta=g('agent_max_theta', D.agents.max_theta),
            movement_mode=g('agent_movement_mode', D.agents.movement_mode),
            sensing_radius=sensing,
            prey_capture_radius=g('agent_prey_capture_radius',
                                  D.agents.prey_capture_radius),
            respawn_captured_prey=g('agent_respawn_captured_prey',
                                    D.agents.respawn_captured_prey),
        ),
        octopus=OctopusConfig(
            num_arms=g('octo_num_arms', D.octopus.num_arms),
            max_body_velocity=g('octo_max_body_velocity',
                                D.octopus.max_body_velocity),
            movement_mode=g('octo_movement_mode', D.octopus.movement_mode),
            sensing_radius=sensing,  # see docstring
            limb=LimbConfig(
                rows=g('limb_rows', D.octopus.limb.rows),
                cols=g('limb_cols', D.octopus.limb.cols),
                min_sucker_distance=g('octo_min_sucker_distance',
                                      D.octopus.limb.min_sucker_distance),
                max_sucker_distance=g('octo_max_sucker_distance',
                                      D.octopus.limb.max_sucker_distance),
                movement_mode=g('limb_movement_mode',
                                D.octopus.limb.movement_mode),
                random=RandomDriftConfig(
                    max_arm_theta=g('octo_max_arm_theta',
                                    D.octopus.limb.random.max_arm_theta),
                ),
                lumped=LumpedSpringConfig(
                    max_arm_reach_theta=g(
                        'octo_max_arm_reach_theta',
                        D.octopus.limb.lumped.max_arm_reach_theta),
                    max_limb_offset=g(
                        'octo_max_limb_offset',
                        D.octopus.limb.lumped.max_limb_offset),
                    arm_stiffness=g('octo_arm_stiffness',
                                    D.octopus.limb.lumped.arm_stiffness),
                    arm_rest_fraction=g(
                        'octo_arm_rest_fraction',
                        D.octopus.limb.lumped.arm_rest_fraction),
                ),
                chain=SpringChainConfig(
                    spring_k=g('octo_chain_spring_k',
                               D.octopus.limb.chain.spring_k),
                    agent_k=g('octo_chain_agent_k',
                              D.octopus.limb.chain.agent_k),
                    move_k=g('octo_chain_move_k',
                             D.octopus.limb.chain.move_k),
                ),
            ),
            sucker=SuckerConfig(
                max_hue_change=g('octo_max_hue_change',
                                 D.octopus.sucker.max_hue_change),
                adjacency_radius=g('adjacency_radius',
                                   D.octopus.sucker.adjacency_radius),
            ),
        ),
        inference=InferenceConfig(
            location=g('inference_location', D.inference.location),
            mode=g('inference_mode', D.inference.mode),
            model=g('inference_model', D.inference.model),
        ),
        output=OutputConfig(
            debug_mode=g('debug_mode', D.output.debug_mode),
            show_forces=g('show_forces', D.output.show_forces),
            log_forces=g('log_forces', D.output.log_forces),
            save_images=g('save_images', D.output.save_images),
            video_fps=g('video_fps', D.output.video_fps),
        ),
        datagen=DatagenConfig(
            write_format=g('datagen_data_write_format',
                           D.datagen.write_format),
            randomize_colors_interval=g(
                'datagen_randomize_colors_interval',
                D.datagen.randomize_colors_interval),
            save_to_disk=g('save_data_to_disk', D.datagen.save_to_disk),
            restore_from_disk=g('restore_data_from_disk',
                                D.datagen.restore_from_disk),
            datagen_mode=g('datagen_mode', D.datagen.datagen_mode),
        ),
        training=replace(
            D.training,
            epochs=g('epochs', D.training.epochs),
            batch_size=g('batch_size', D.training.batch_size),
            test_size=g('test_size', D.training.test_size),
            constraint_loss_weight=g('constraint_loss_weight',
                                     D.training.constraint_loss_weight),
            sucker_delta_model=g('sucker_delta_model',
                                 D.training.sucker_delta_model),
        ),
        paths=DEFAULT_PATHS,
    )


def as_config(params) -> Config:
    """Normalize a Config-or-flat-dict into a Config.

    Lets constructors take typed config internally while the callers that
    legitimately speak flat (the wire protocol, old pickles) keep passing
    dicts.
    """
    if isinstance(params, Config):
        return params
    return config_from_flat(params)
