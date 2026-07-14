"""
Octopus game and ML parameters.

The typed schema in config_schema.py is the source of truth. This module
builds the concrete profiles and, for now, derives the legacy flat
GameParameters / TrainingParameters dicts from the DEFAULT profile so the
~38 existing `params['some_key']` call sites keep working untouched while
they are migrated.

Once every call site takes a Config object, LEGACY DERIVATION below (and
this docstring) can go.
"""
import os
from dataclasses import replace

from config_schema import (  # noqa: F401  (re-exported for convenience)
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

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

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


# =================================================== LEGACY DERIVATION ===
# Flat dicts derived from DEFAULT. Every key below is exactly what it was
# before the reorg, so no call site changes in this commit. The mapping also
# documents the old -> new correspondence for the migration.
#
# Note on agent_range_radius: it used to serve two distinct concepts - how
# far an AGENT senses the octopus, and how far an ARM senses agents. The
# schema splits them (agents.sensing_radius / octopus.sensing_radius), but a
# single flat key cannot express the split, so the legacy view emits the
# agent's value and both readers see it, exactly as before. The split only
# becomes real once Limb reads octopus.sensing_radius directly.

def to_game_parameters(cfg: Config) -> dict:
    """Legacy flat GameParameters view of a Config."""
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

        # Training hyperparams. These used to be duplicated here AND in
        # TrainingParameters with nothing syncing them; now both views read
        # the single cfg.training.* source.
        'test_size': cfg.training.test_size,
        'epochs': cfg.training.epochs,
        'batch_size': cfg.training.batch_size,
        'constraint_loss_weight': cfg.training.constraint_loss_weight,
    }


def to_training_parameters(cfg: Config) -> dict:
    """Legacy flat TrainingParameters view of a Config."""
    return {
        'ml_mode': cfg.training.ml_mode,

        # Datagen
        'save_data_to_disk': cfg.datagen.save_to_disk,
        'restore_data_from_disk': cfg.datagen.restore_from_disk,
        'datagen_mode': cfg.datagen.datagen_mode,

        # Tensorboard
        'erase_old_tensorboard_logs': cfg.training.erase_old_tensorboard_logs,
        'generate_tensorboard': cfg.training.generate_tensorboard,

        # Model save and restore
        'training_model': cfg.training.training_model,
        'save_model_to_disk': cfg.training.save_model_to_disk,
        'restore_model_from_disk': cfg.training.restore_model_from_disk,

        'run_training': cfg.training.run_training,

        # Test & Eval
        'run_inference': cfg.training.run_inference,
        'run_eval': cfg.training.run_eval,

        # Paths
        'models': cfg.paths.model_paths,
        'datasets': cfg.paths.dataset_paths,

        # Training hyperparams (single source: cfg.training)
        'test_size': cfg.training.test_size,
        'epochs': cfg.training.epochs,
        'batch_size': cfg.training.batch_size,
        'constraint_loss_weight': cfg.training.constraint_loss_weight,
    }


GameParameters: dict = to_game_parameters(DEFAULT)
TrainingParameters: dict = to_training_parameters(DEFAULT)
