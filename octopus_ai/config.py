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
from dataclasses import fields, is_dataclass, replace
from enum import Enum

import numpy as np

from octopus_ai.config_schema import (  # noqa: F401  (re-exported for convenience)
    AgentConfig,
    Config,
    DatagenConfig,
    ILQRConfig,
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
from simulator.simutil import MLMode, MovementMode, PropulsionMode

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


def json_default(o):
    """`default=` for json.dumps over a flat config.

    config_to_flat holds raw types json can't serialize on its own: Enum
    members (MovementMode, MLMode, ...) and numpy scalars. Enums serialize as
    their .name (D14); numpy scalars unwrap to their Python value; anything
    else falls back to str. Shared by force_logger and the record & replay
    recorder so the on-disk config snapshot is one consistent format.
    """
    if isinstance(o, Enum):
        return o.name
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


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

# Watching the iLQR octopus. The shared profile for BOTH front ends - the
# matplotlib visualizer (octo_viz) and the browser/websocket server - so they
# show the same simulation: iLQR motor control on the body and every limb,
# camouflage via the fast NO_MODEL heuristic (no trained model needed), the
# octopus outlined so it stays visible, and per-step performance tracking.
# To camouflage against a picture instead of random noise, set
# BACKGROUND_IMAGE to an image path (any format Pillow reads); it is
# grayscaled and resized to the grid. None = the usual random surface. A
# missing/unreadable file degrades gracefully back to random. The grid is
# bumped to 30x30 so the picture is legible (the octopus is then smaller
# relative to the world); drop it back to 15 for a chunkier, octopus-prominent
# view.
BACKGROUND_IMAGE = "/Users/davabrams/Pictures/ucfN3bl.jpg"

VIZ_ILQR = replace(
    VIZ,
    # Camouflage via the trained per-channel SUCKER model (falls back to the
    # NO_MODEL heuristic if you swap this to MLMode.NO_MODEL).
    inference=replace(VIZ.inference, mode=MLMode.SUCKER),
    output=replace(VIZ.output, highlight_octopus=True, track_performance=True),
    world=replace(VIZ.world, background_image=BACKGROUND_IMAGE,
                  x_len=30, y_len=30),
    octopus=replace(
        VIZ.octopus,
        movement_mode=MovementMode.ILQR,
        # The body translates by the external-reaction propulsion model
        # (friction-limited crawl + siphon jet on threat), not the legacy
        # internal summed-tension drift: watch a threat appear and the body jet
        # straight away from it. See Octopus._propel_body / PropulsionMode.
        propulsion_mode=PropulsionMode.REACTION,
        # Exploration ON so idle arms reach for the least-explored cells instead
        # of hanging slack - that reach is what plants a tip and lets the CRAWL
        # haul the body, so the octopus forages/wanders even with no prey or
        # threat nearby (without it a quiescent scene shows no locomotion).
        limb=replace(VIZ.octopus.limb, movement_mode=MovementMode.ILQR,
                     ilqr=replace(VIZ.octopus.limb.ilqr, explore_enabled=True)),
    ),
    # Agents react to the octopus: THREATs pursue it, PREY flee, inside their
    # sense window (ignoring camouflage). Makes the hunt/flee dynamics visible
    # instead of RANDOM wandering.
    agents=replace(VIZ.agents, movement_mode=MovementMode.PURSUIT_FLEE),
)

# Headless record-and-replay. Same simulation as VIZ_ILQR, but writes a
# DuckDB run file with per-iteration iLQR history for the analyzer to play
# back. Force arrows off (there is no live matplotlib window).
RECORD = replace(
    VIZ_ILQR,
    output=replace(VIZ_ILQR.output, record_run=True,
                   record_ilqr_history=True, show_forces=False),
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
        'background_image': cfg.world.background_image,
        'rand_seed': cfg.run.rand_seed,
        'debug_mode': cfg.output.debug_mode,
        'log_forces': cfg.output.log_forces,
        'show_forces': cfg.output.show_forces,
        'highlight_octopus': cfg.output.highlight_octopus,
        'track_performance': cfg.output.track_performance,
        'save_images': cfg.output.save_images,
        'record_run': cfg.output.record_run,
        'record_ilqr_history': cfg.output.record_ilqr_history,
        'adjacency_radius': cfg.octopus.sucker.adjacency_radius,
        'inference_location': cfg.inference.location,
        'inference_mode': cfg.inference.mode,
        'inference_model': cfg.inference.model,
        'datagen_data_write_format': cfg.datagen.write_format,

        # Agent parameters
        'agent_number_of_agents': cfg.agents.count,
        'agent_max_velocity': cfg.agents.max_velocity,
        'agent_max_theta': cfg.agents.max_theta,
        'agent_wander_persistence': cfg.agents.wander_persistence,
        'agent_movement_mode': cfg.agents.movement_mode,
        'agent_range_radius': cfg.agents.sensing_radius,  # see note above
        'agent_prey_capture_radius': cfg.agents.prey_capture_radius,
        'agent_respawn_captured_prey': cfg.agents.respawn_captured_prey,
        'agent_visibility_threshold': cfg.agents.visibility_threshold,

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
        'octo_ilqr_horizon': cfg.octopus.limb.ilqr.horizon,
        'octo_ilqr_max_iters': cfg.octopus.limb.ilqr.max_iters,
        'octo_ilqr_body_stiffness': cfg.octopus.limb.ilqr.body_stiffness,
        'octo_ilqr_w_spring': cfg.octopus.limb.ilqr.w_spring,
        'octo_ilqr_w_spring_stiffen':
            cfg.octopus.limb.ilqr.w_spring_stiffen,
        'octo_ilqr_spring_slack': cfg.octopus.limb.ilqr.spring_slack,
        'octo_ilqr_bend_deadzone_deg':
            cfg.octopus.limb.ilqr.bend_deadzone_deg,
        'octo_ilqr_w_effort_stiffen':
            cfg.octopus.limb.ilqr.w_effort_stiffen,
        'octo_ilqr_w_bend': cfg.octopus.limb.ilqr.w_bend,
        'octo_ilqr_w_effort': cfg.octopus.limb.ilqr.w_effort,
        'octo_ilqr_w_reach_run': cfg.octopus.limb.ilqr.w_reach_run,
        'octo_ilqr_w_reach_terminal': cfg.octopus.limb.ilqr.w_reach_terminal,
        'octo_ilqr_sense_ramp_band': cfg.octopus.limb.ilqr.sense_ramp_band,
        'octo_ilqr_w_repel': cfg.octopus.limb.ilqr.w_repel,
        'octo_ilqr_repel_step': cfg.octopus.limb.ilqr.repel_step,
        'octo_ilqr_repel_radius': cfg.octopus.limb.ilqr.repel_radius,
        'octo_ilqr_repel_tip_fraction':
            cfg.octopus.limb.ilqr.repel_tip_fraction,
        'octo_ilqr_explore_enabled': cfg.octopus.limb.ilqr.explore_enabled,
        'octo_ilqr_w_explore': cfg.octopus.limb.ilqr.w_explore,
        'octo_ilqr_explore_node_radius':
            cfg.octopus.limb.ilqr.explore_node_radius,
        'octo_ilqr_explore_target_smooth':
            cfg.octopus.limb.ilqr.explore_target_smooth,
        'octo_ilqr_explore_ticks': cfg.octopus.limb.ilqr.explore_ticks,
        'octo_ilqr_w_explore_threat_avoid':
            cfg.octopus.limb.ilqr.w_explore_threat_avoid,
        'octo_ilqr_explore_threat_radius':
            cfg.octopus.limb.ilqr.explore_threat_radius,
        'octo_ilqr_body_torque_gain': cfg.octopus.limb.ilqr.body_torque_gain,
        'octo_num_arms': cfg.octopus.num_arms,
        'octo_ring_radius': cfg.octopus.ring_radius,
        'octo_max_body_angular_velocity':
            cfg.octopus.max_body_angular_velocity,
        'octo_propulsion_mode': cfg.octopus.propulsion_mode,
        'octo_crawl_grip_limit': cfg.octopus.crawl_grip_limit,
        'octo_crawl_plant_speed': cfg.octopus.crawl_plant_speed,
        'octo_jet_enabled': cfg.octopus.jet_enabled,
        'octo_jet_trigger_radius': cfg.octopus.jet_trigger_radius,
        'octo_jet_impulse': cfg.octopus.jet_impulse,
        'octo_jet_decay': cfg.octopus.jet_decay,
        'octo_max_jet_velocity': cfg.octopus.max_jet_velocity,
        'octo_max_sucker_distance': cfg.octopus.limb.max_sucker_distance,
        'octo_min_sucker_distance': cfg.octopus.limb.min_sucker_distance,
        'octo_sucker_col_spacing': cfg.octopus.limb.sucker_col_spacing,
        'octo_movement_mode': cfg.octopus.movement_mode,
        'octo_threading': cfg.run.threading,

        # Limb parameters
        'limb_rows': cfg.octopus.limb.rows,
        'limb_cols': cfg.octopus.limb.cols,
        'limb_movement_mode': cfg.octopus.limb.movement_mode,

        # Sucker parameters
        'octo_max_hue_change': cfg.octopus.sucker.max_hue_change,
        'sucker_delta_model': cfg.training.sucker_delta_model,
        'sucker_hue_change_conditioned':
            cfg.training.sucker_hue_change_conditioned,
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
            background_image=g('background_image', D.world.background_image),
        ),
        agents=AgentConfig(
            count=g('agent_number_of_agents', D.agents.count),
            max_velocity=g('agent_max_velocity', D.agents.max_velocity),
            max_theta=g('agent_max_theta', D.agents.max_theta),
            wander_persistence=g('agent_wander_persistence',
                                 D.agents.wander_persistence),
            movement_mode=g('agent_movement_mode', D.agents.movement_mode),
            sensing_radius=sensing,
            prey_capture_radius=g('agent_prey_capture_radius',
                                  D.agents.prey_capture_radius),
            respawn_captured_prey=g('agent_respawn_captured_prey',
                                    D.agents.respawn_captured_prey),
            visibility_threshold=g('agent_visibility_threshold',
                                   D.agents.visibility_threshold),
        ),
        octopus=OctopusConfig(
            num_arms=g('octo_num_arms', D.octopus.num_arms),
            max_body_velocity=g('octo_max_body_velocity',
                                D.octopus.max_body_velocity),
            movement_mode=g('octo_movement_mode', D.octopus.movement_mode),
            sensing_radius=sensing,  # see docstring
            ring_radius=g('octo_ring_radius', D.octopus.ring_radius),
            max_body_angular_velocity=g('octo_max_body_angular_velocity',
                                        D.octopus.max_body_angular_velocity),
            propulsion_mode=g('octo_propulsion_mode',
                              D.octopus.propulsion_mode),
            crawl_grip_limit=g('octo_crawl_grip_limit',
                               D.octopus.crawl_grip_limit),
            crawl_plant_speed=g('octo_crawl_plant_speed',
                                D.octopus.crawl_plant_speed),
            jet_enabled=g('octo_jet_enabled', D.octopus.jet_enabled),
            jet_trigger_radius=g('octo_jet_trigger_radius',
                                 D.octopus.jet_trigger_radius),
            jet_impulse=g('octo_jet_impulse', D.octopus.jet_impulse),
            jet_decay=g('octo_jet_decay', D.octopus.jet_decay),
            max_jet_velocity=g('octo_max_jet_velocity',
                               D.octopus.max_jet_velocity),
            limb=LimbConfig(
                rows=g('limb_rows', D.octopus.limb.rows),
                cols=g('limb_cols', D.octopus.limb.cols),
                min_sucker_distance=g('octo_min_sucker_distance',
                                      D.octopus.limb.min_sucker_distance),
                max_sucker_distance=g('octo_max_sucker_distance',
                                      D.octopus.limb.max_sucker_distance),
                sucker_col_spacing=g('octo_sucker_col_spacing',
                                     D.octopus.limb.sucker_col_spacing),
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
                ilqr=ILQRConfig(
                    horizon=g('octo_ilqr_horizon',
                              D.octopus.limb.ilqr.horizon),
                    max_iters=g('octo_ilqr_max_iters',
                                D.octopus.limb.ilqr.max_iters),
                    body_stiffness=g('octo_ilqr_body_stiffness',
                                     D.octopus.limb.ilqr.body_stiffness),
                    body_torque_gain=g('octo_ilqr_body_torque_gain',
                                       D.octopus.limb.ilqr.body_torque_gain),
                    w_spring_stiffen=g(
                        'octo_ilqr_w_spring_stiffen',
                        D.octopus.limb.ilqr.w_spring_stiffen),
                    spring_slack=g('octo_ilqr_spring_slack',
                                   D.octopus.limb.ilqr.spring_slack),
                    bend_deadzone_deg=g(
                        'octo_ilqr_bend_deadzone_deg',
                        D.octopus.limb.ilqr.bend_deadzone_deg),
                    w_effort_stiffen=g(
                        'octo_ilqr_w_effort_stiffen',
                        D.octopus.limb.ilqr.w_effort_stiffen),
                    w_spring=g('octo_ilqr_w_spring',
                               D.octopus.limb.ilqr.w_spring),
                    w_bend=g('octo_ilqr_w_bend', D.octopus.limb.ilqr.w_bend),
                    w_effort=g('octo_ilqr_w_effort',
                               D.octopus.limb.ilqr.w_effort),
                    w_reach_run=g('octo_ilqr_w_reach_run',
                                  D.octopus.limb.ilqr.w_reach_run),
                    w_reach_terminal=g('octo_ilqr_w_reach_terminal',
                                       D.octopus.limb.ilqr.w_reach_terminal),
                    sense_ramp_band=g('octo_ilqr_sense_ramp_band',
                                      D.octopus.limb.ilqr.sense_ramp_band),
                    w_repel=g('octo_ilqr_w_repel',
                              D.octopus.limb.ilqr.w_repel),
                    repel_step=g('octo_ilqr_repel_step',
                                 D.octopus.limb.ilqr.repel_step),
                    repel_radius=g('octo_ilqr_repel_radius',
                                   D.octopus.limb.ilqr.repel_radius),
                    repel_tip_fraction=g(
                        'octo_ilqr_repel_tip_fraction',
                        D.octopus.limb.ilqr.repel_tip_fraction),
                    explore_enabled=g('octo_ilqr_explore_enabled',
                                      D.octopus.limb.ilqr.explore_enabled),
                    w_explore=g('octo_ilqr_w_explore',
                                D.octopus.limb.ilqr.w_explore),
                    explore_node_radius=g(
                        'octo_ilqr_explore_node_radius',
                        D.octopus.limb.ilqr.explore_node_radius),
                    explore_target_smooth=g(
                        'octo_ilqr_explore_target_smooth',
                        D.octopus.limb.ilqr.explore_target_smooth),
                    explore_ticks=g('octo_ilqr_explore_ticks',
                                    D.octopus.limb.ilqr.explore_ticks),
                    w_explore_threat_avoid=g(
                        'octo_ilqr_w_explore_threat_avoid',
                        D.octopus.limb.ilqr.w_explore_threat_avoid),
                    explore_threat_radius=g(
                        'octo_ilqr_explore_threat_radius',
                        D.octopus.limb.ilqr.explore_threat_radius),
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
            highlight_octopus=g('highlight_octopus',
                                D.output.highlight_octopus),
            track_performance=g('track_performance',
                                D.output.track_performance),
            log_forces=g('log_forces', D.output.log_forces),
            save_images=g('save_images', D.output.save_images),
            video_fps=g('video_fps', D.output.video_fps),
            record_run=g('record_run', D.output.record_run),
            record_ilqr_history=g('record_ilqr_history',
                                  D.output.record_ilqr_history),
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
            sucker_hue_change_conditioned=g(
                'sucker_hue_change_conditioned',
                D.training.sucker_hue_change_conditioned),
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


def format_config(cfg: Config, title: str = "CONFIG") -> str:
    """Render a Config (nested frozen dataclasses) as an indented tree.

    Meant for a startup dump so a run's full configuration is visible in the
    console. Enums print by name, nested dataclasses are indented, and dict
    fields (e.g. the path maps) are expanded one entry per line.
    """
    def fmt_scalar(v):
        if isinstance(v, Enum):
            return v.name
        if isinstance(v, str):
            return repr(v)
        return str(v)

    lines = [f"===== {title} ====="]

    def render(obj, prefix):
        for f in fields(obj):
            val = getattr(obj, f.name)
            if is_dataclass(val) and not isinstance(val, type):
                lines.append(f"{prefix}{f.name}:")
                render(val, prefix + "  ")
            elif isinstance(val, dict):
                lines.append(f"{prefix}{f.name}:")
                for k, v in val.items():
                    lines.append(f"{prefix}  {fmt_scalar(k)}: {fmt_scalar(v)}")
            else:
                lines.append(f"{prefix}{f.name}: {fmt_scalar(val)}")

    render(cfg, "  ")
    lines.append("=" * (len(title) + 12))
    return "\n".join(lines)


def print_config(cfg: Config, title: str = "CONFIG") -> None:
    """Print the full config tree to stdout (startup dump)."""
    print(format_config(cfg, title))
