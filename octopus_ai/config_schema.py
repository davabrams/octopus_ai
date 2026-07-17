"""Typed configuration schema for the octopus simulator.

This is the source of truth for every tunable. config.py builds the
concrete profiles from it and converts to/from the flat dict form that the
browser, the force log, and the test fixtures speak.

WHY DATACLASSES
---------------
The old flat dicts had 45 + 18 keys with four duplicated across both, no
type checking, and prefixes (octo_, agent_, limb_, datagen_) that encoded a
hierarchy without enforcing it - so params drifted into the wrong section
and, in one case, the wrong parameter was read entirely (find_adjacents was
passed agent_range_radius instead of adjacency_radius; see commit 99db724).

Everything here is frozen: a config cannot be mutated in place. Derive
variants with dataclasses.replace(), which is also how profiles are built.

STRUCTURE
---------
Nesting follows the concern, not the prefix. Mode-specific knobs live under
the mode that uses them (limb.random / limb.lumped / limb.chain), so
"which knobs apply to SPRING_CHAIN?" is answerable by autocomplete rather
than by reading comments.
"""
from dataclasses import dataclass, field, replace
from typing import Dict, Optional

from simulator.simutil import InferenceLocation, MLMode, MovementMode


# ---------------------------------------------------------------- paths ---
@dataclass(frozen=True)
class PathsConfig:
    """Where models and generated datasets live on disk.

    Keyed by MLMode because callers select a model by mode
    (training/sucker.py, training/limb.py, model.py all index these).
    """
    model_paths: Dict[MLMode, Optional[str]] = field(default_factory=dict)
    dataset_paths: Dict[MLMode, Optional[str]] = field(default_factory=dict)


# ------------------------------------------------------------------ run ---
@dataclass(frozen=True)
class RunConfig:
    """How a run executes. Not part of the simulated world."""
    num_iterations: int = 120  # -1 for an infinite loop
    rand_seed: int = 0
    threading: bool = True  # was octo_threading; an execution concern, not
                            # octopus anatomy


# ---------------------------------------------------------------- world ---
@dataclass(frozen=True)
class WorldConfig:
    x_len: int = 15
    y_len: int = 15
    surface_grayscale: bool = True  # random grayscale values in [0, 1);
                                    # False for the classic binary 0/1 grid
    background_image: Optional[str] = None  # path to an image file to use as
                                            # the surface (grayscaled + resized
                                            # to x_len x y_len); None = random


# --------------------------------------------------------------- agents ---
@dataclass(frozen=True)
class AgentConfig:
    count: int = 5  # number of AGENTS in the world (ie prey and threat); 0 disables
    max_velocity: float = 0.1
    max_theta: float = 0.1
    movement_mode: MovementMode = MovementMode.RANDOM
    sensing_radius: float = 5.0  # how far an AGENT senses the octopus.
                                 # Previously agent_range_radius, which also
                                 # doubled as the octopus's sensing radius -
                                 # see OctopusConfig.sensing_radius.
    prey_capture_radius: float = 0.3  # a PREY within this distance of any
                                      # sucker is captured; 0 disables
    respawn_captured_prey: bool = False


# ------------------------------------------------- limb movement models ---
@dataclass(frozen=True)
class RandomDriftConfig:
    """Knobs for MovementMode.RANDOM."""
    max_arm_theta: float = 0.1


@dataclass(frozen=True)
class LumpedSpringConfig:
    """Knobs for MovementMode.LUMPED_SPRING.

    One spring spans base->tip; intermediate suckers are placed by a
    geometric reflow rather than by forces.
    """
    max_arm_reach_theta: float = 0.3  # per-joint bend cap while reaching
    max_limb_offset: float = 0.5  # cap on tip movement per step
    arm_stiffness: float = 0.5  # higher = stubbier arms that yank the body
                                # harder; lower = longer reach, gentler pull
    arm_rest_fraction: float = 0.3  # neutral length between fully tucked
                                    # (0.0) and fully extended (1.0). Sits
                                    # above min so a threat can COMPRESS the
                                    # arm below rest, producing outward
                                    # tension that flees the threat.


@dataclass(frozen=True)
class SpringChainConfig:
    """Knobs for MovementMode.SPRING_CHAIN.

    Every sucker is a node; the arm's shape emerges from a direct linear
    solve of the force balance.
    """
    spring_k: float = 1.0  # neighbour spring stiffness along the arm
    agent_k: float = 0.5  # prey spring-to-target stiffness, before the tip
                          # weighting; threats enter as a constant push
    move_k: float = 2.0  # uniform per-sucker movement cost: an anchor spring
                         # to the sucker's position at the START of the
                         # frame, so a sucker must pay to move rather than
                         # teleporting to its balance point. Higher = more
                         # sluggish. Doubles as a trust region once forces
                         # go nonlinear.


@dataclass(frozen=True)
class ILQRConfig:
    """Knobs for MovementMode.ILQR (simulator/ilqr; ARCHITECTURE.md §11.4).

    Each limb runs its own compiled iLQR controller. `horizon`/`max_iters`
    size the per-frame receding-horizon solve; the `w_*` weights balance the
    arm's competing wants (see ARCHITECTURE.md §4.5). `body_stiffness` scales
    how hard each arm's attract/repel influence tugs the shared body.
    """
    horizon: int = 10            # trajectory length planned each frame
    max_iters: int = 5           # iLQR iterations per frame (MPC, warm-started)
    body_stiffness: float = 3.0  # gain from the arm's base-segment spring
                                 # tension to the body's drift (the body only
                                 # ever moves via this tension - never sensing
                                 # agents directly)
    body_torque_gain: float = 3.0  # gain from each arm's base reaction (applied
                                   # at its ring point) to the body's ROTATION.
                                   # Torque and linear tension share the same
                                   # base-reaction source, so they live together.
    w_spring: float = 2.0        # segment spacing toward rest length
    w_bend: float = 1.0          # straightness (anti-crumple)
    w_effort: float = 0.5        # per-node per-step VELOCITY penalty (the
                                 # "translation" cost): |u|^2 keeps motion
                                 # bounded/smooth so the arm can't teleport
                                 # (no unrealistic accel/jerk). Summed over every
                                 # free node x horizon step, so it weighs on the
                                 # total far more than its scalar suggests - push
                                 # it too high and the gentle explore pull loses.
    w_reach_run: float = 0.1     # weak per-step tip pull toward the target
    w_reach_terminal: float = 6.0  # strong terminal tip pull (dominates)
    w_repel: float = 8.0         # threat avoidance strength
    repel_radius: float = 2.5    # threat keep-out radius
    # Exploration (off by default): when an arm senses no prey, its tip seeks
    # the least-explored cell within reach (cells are marked explored by the
    # SUCKERS, not the body). A weak drive: w_explore << w_reach_terminal, and
    # prey always preempts it, so exploration never outranks hunting/fleeing.
    explore_enabled: bool = False
    w_explore: float = 0.5       # gentle terminal tip pull toward unexplored
                                 # (vs 6.0 for prey) - the "much less reward"
    explore_decay: float = 0.95  # per-frame decay of visit counts; < 1.0 turns
                                 # "least visited" into "least RECENTLY visited"
                                 # and makes the overlay read as recency (recent
                                 # cells bright, old ones fade). 1.0 = cumulative.
    # The explore-target picker is otherwise threat-blind, so the least-visited
    # frontier is exactly the region the threat has kept the octopus out of -
    # the arm reaches for it while the repel barrier shoves back, and the two
    # cancel into a stall. Penalize explore cells near a sensed threat so the
    # goal lands on the SAFE side (the solver's repel term still guards the
    # crossing). Zero weight reproduces the old threat-blind behaviour.
    w_explore_threat_avoid: float = 10.0  # penalty per unit inside the radius
    explore_threat_radius: float = 5.0    # cells within this of a threat pay it


# ---------------------------------------------------------------- limbs ---
@dataclass(frozen=True)
class SuckerConfig:
    max_hue_change: float = 0.25  # max rgb change per step; the constraint
                                  # threshold (was octo_max_hue_change)
    adjacency_radius: float = 1.0  # how close two suckers must be to count
                                   # as neighbours for the LIMB model


@dataclass(frozen=True)
class LimbConfig:
    rows: int = 16
    cols: int = 2
    min_sucker_distance: float = 0.1  # was octo_min_sucker_distance
    max_sucker_distance: float = 0.3  # was octo_max_sucker_distance
    movement_mode: MovementMode = MovementMode.RANDOM
    # Mode-specific knobs, nested under the mode that reads them.
    random: RandomDriftConfig = field(default_factory=RandomDriftConfig)
    lumped: LumpedSpringConfig = field(default_factory=LumpedSpringConfig)
    chain: SpringChainConfig = field(default_factory=SpringChainConfig)
    ilqr: ILQRConfig = field(default_factory=ILQRConfig)


# -------------------------------------------------------------- octopus ---
@dataclass(frozen=True)
class OctopusConfig:
    num_arms: int = 8
    max_body_velocity: float = 0.25
    movement_mode: MovementMode = MovementMode.RANDOM
    sensing_radius: float = 5.0  # how far an ARM senses agents. Split from
                                 # the agent's own sensing radius, which it
                                 # was accidentally forced to equal.
    ring_radius: float = 0.5  # each limb's base is pinned to a point on a ring
                              # of this radius around the body center (at the
                              # limb's fixed angular slot, rotated by the body's
                              # orientation) - so arms fan out from DISTINCT
                              # roots instead of all sharing the body center.
                              # 0.0 reproduces the legacy single-point base.
    max_body_angular_velocity: float = 0.1  # per-frame cap on body rotation
                                            # (rad); the angular twin of
                                            # max_body_velocity.
    limb: LimbConfig = field(default_factory=LimbConfig)
    sucker: SuckerConfig = field(default_factory=SuckerConfig)


# ------------------------------------------------------------ inference ---
@dataclass(frozen=True)
class InferenceConfig:
    location: InferenceLocation = InferenceLocation.LOCAL
    mode: MLMode = MLMode.SUCKER  # how colours are computed
    model: MLMode = MLMode.SUCKER  # WHICH model to load; resolve via
                                   # Config.inference_model_path rather than
                                   # indexing a path table at the call site


# --------------------------------------------------------------- output ---
@dataclass(frozen=True)
class OutputConfig:
    """Observability and side effects. All off by default: a fresh checkout
    should not write databases or videos unprompted."""
    debug_mode: bool = False  # draws agent sensing regions, etc.
    show_forces: bool = False  # force-vector arrows on the visualizer
    highlight_octopus: bool = False  # encircle suckers + draw the arm
                                     # centerlines, so the octopus stays visible
                                     # even when camouflaged into the surface
    log_forces: bool = False  # per-frame forces -> logs/forces.db
    save_images: bool = False  # per-frame PNGs -> stitched to MP4
    video_fps: int = 5  # playback rate of the stitched video
    track_performance: bool = False  # time each sim step + report memory
                                     # stats, printed as a summary at run end
                                     # (octopus_ai/perf.PerfTracker)
    record_run: bool = False  # write a DuckDB run file (SimRecorder)
    record_ilqr_history: bool = False  # capture per-iteration iLQR history on
                                       # each limb; off = zero overhead


# -------------------------------------------------------------- datagen ---
@dataclass(frozen=True)
class DatagenConfig:
    write_format: MLMode = MLMode.SUCKER  # was datagen_data_write_format
    randomize_colors_interval: int = 2  # re-randomize sucker colours every N
                                        # iterations so training data covers
                                        # mismatched (colour, surface) pairs;
                                        # 0 disables
    save_to_disk: bool = True
    restore_from_disk: bool = False
    datagen_mode: bool = True


# ------------------------------------------------------------- training ---
@dataclass(frozen=True)
class TrainingConfig:
    ml_mode: MLMode = MLMode.SUCKER
    training_model: MLMode = MLMode.SUCKER

    # Hyperparams. These lived in BOTH GameParameters and
    # TrainingParameters, with nothing keeping the two in sync.
    epochs: int = 50
    batch_size: int = 32  # tf default
    test_size: float = 0.2
    constraint_loss_weight: float = 0.95

    sucker_delta_model: bool = True  # sucker net predicts a tanh-bounded
                                     # colour DELTA (constraint enforced by
                                     # architecture); False = legacy direct
                                     # prediction with WeightedSumLoss

    save_model_to_disk: bool = True
    restore_model_from_disk: bool = False
    run_training: bool = True
    run_inference: bool = False
    run_eval: bool = False

    erase_old_tensorboard_logs: bool = True
    generate_tensorboard: bool = True


# ----------------------------------------------------------------- root ---
@dataclass(frozen=True)
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    octopus: OctopusConfig = field(default_factory=OctopusConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    datagen: DatagenConfig = field(default_factory=DatagenConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @property
    def inference_model_path(self) -> Optional[str]:
        """Resolved path of the model to run inference with.

        Centralizes what call sites used to do by hand:
            default_models[params['inference_model']]
        """
        return self.paths.model_paths.get(self.inference.model)

    @property
    def training_model_path(self) -> Optional[str]:
        return self.paths.model_paths.get(self.training.training_model)

    @property
    def training_dataset_path(self) -> Optional[str]:
        return self.paths.dataset_paths.get(self.training.ml_mode)
