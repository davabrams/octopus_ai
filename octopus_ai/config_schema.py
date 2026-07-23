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

from simulator.simutil import InferenceLocation, MLMode, MovementMode, PropulsionMode


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
    max_velocity: float = 0.25
    max_theta: float = 0.1
    wander_persistence: int = 12  # RANDOM/idle only: hold a wander heading this
                                  # many frames before re-rolling it. Re-rolling
                                  # velocity EVERY frame is symmetric about zero,
                                  # so agents jitter in place instead of roaming;
                                  # holding a heading lets them actually travel.
                                  # 1 = re-roll every frame (old behavior).
    movement_mode: MovementMode = MovementMode.RANDOM
    sensing_radius: float = 5.0  # how far an AGENT senses the octopus.
                                 # Previously agent_range_radius, which also
                                 # doubled as the octopus's sensing radius -
                                 # see OctopusConfig.sensing_radius.
    prey_capture_radius: float = 0.3  # a PREY within this distance of any
                                      # sucker is captured; 0 disables
    respawn_captured_prey: bool = False
    visibility_threshold: float = 0.05  # PURSUIT_FLEE only: the octopus must be
                                        # at least this VISIBLE (mean squared
                                        # camouflage error, ~0 = perfectly hidden)
                                        # before threats notice and commit to
                                        # pursuit; below it the octopus is
                                        # effectively invisible and agents wander.
                                        # So camouflage is protective - a hidden
                                        # octopus is left alone, a poorly-matched
                                        # (or moving-into-new-terrain) one gets
                                        # hunted. The visibility-gated REACTIVE
                                        # modes ignore this (they already scale
                                        # continuously with visibility).


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
    max_iters: int = 30          # iLQR iteration CAP per frame (MPC, warm-started).
                                 # The solver early-exits when rel_improve < tol, so
                                 # this is headroom, not a fixed count - typical
                                 # frames converge in ~6-8 iters, so raising the cap
                                 # is nearly free. Raised from 5: against the stiff
                                 # spring/effort landscape the line search takes
                                 # small steps (alpha ~0.25-0.5), so a cap of 5 cut
                                 # the descent off mid-solve - solves finished
                                 # UNconverged every frame (~8% converged) and the
                                 # arm only crept ~0.04 units/frame, unable to act
                                 # on a threat before the solve was truncated. At 30
                                 # it converges ~100% of frames
    compiled_backward: bool = True   # use the graph-compiled backward pass
                                     # (simulator/ilqr/solver_parallel.py) instead
                                     # of the eager Riccati loop. Same arm, same
                                     # math, but the whole backward pass is ONE
                                     # tf.function call/iter instead of ~2*horizon
                                     # eager dispatches - measured ~3.3x faster per
                                     # solve (~1.7x whole sim). Set False for the
                                     # eager reference path; results differ only by
                                     # float32 op-reorder noise (~1e-3)
    body_stiffness: float = 3.0  # gain from the arm's base-segment spring
                                 # tension to the body's drift (the body only
                                 # ever moves via this tension - never sensing
                                 # agents directly)
    body_torque_gain: float = 3.0  # gain from each arm's base reaction (applied
                                   # at its ring point) to the body's ROTATION.
                                   # Torque and linear tension share the same
                                   # base-reaction source, so they live together.
    w_spring: float = 4.0        # segment spacing toward rest length (linear).
                                 # Raised from 2.0 to hold sucker spacing more
                                 # firmly so explore-reaching nodes can't ball up
                                 # at the arm tip (they crowded when the spring
                                 # was soft; see the analyzer's per-node costs)
    w_spring_stiffen: float = 30.0  # super-linear (cubic-force) spring term: a
                                    # soft wall against large stretch/compression
                                    # so a strong attractor can't ball up or
                                    # stretch the chain (min seg ~= rest). Also acts
                                    # as ROTATIONAL DAMPING - softening it (tried 12)
                                    # let the reaching arm oscillate, feeding
                                    # residual torque into the body so its rotation
                                    # never settled (~half the cap rate). Keep stiff
    spring_slack: float = 0.00   # deadband: a segment may deviate +-this from
                                 # rest length for FREE (nodes move paying only
                                 # effort within the slack). Narrowed from 0.25:
                                 # the wide free zone let tip nodes compress
                                 # together at zero cost and bunch up
    w_bend: float = 2.0          # straightness (anti-crumple). Raised from 1.0 to
                                 # keep the arm shaft straighter and the suckers
                                 # more evenly spread from base to tip
    bend_deadzone_deg: float = 15.0  # free bend per node before bending engages
    w_effort_stiffen: float = 5.0  # super-linear (cubic-force) velocity penalty:
                                   # forbids a node teleporting across the field
                                   # in one frame; gentle for normal moves
    w_effort: float = 0.5        # per-node per-step VELOCITY penalty (the
                                 # "translation" cost): |u|^2 keeps motion
                                 # bounded/smooth so the arm can't teleport
                                 # (no unrealistic accel/jerk). Summed over every
                                 # free node x horizon step, so it weighs on the
                                 # total far more than its scalar suggests - push
                                 # it too high and the gentle explore pull loses.
    w_reach_run: float = 0.1     # weak per-step tip pull toward the target
    w_reach_terminal: float = 3.0  # terminal tip pull toward prey. Halved from
                                   # 6.0: at 6 (vs w_spring=4) the reach cost
                                   # (weight x distance^2) dwarfed the spring, so
                                   # iLQR bunched nodes onto the prey/sense edge
                                   # rather than letting the chain hold spacing
    sense_ramp_band: float = 0.3  # smooth the HARD edge at the agent sense radius
                                  # R. Prey attract used to switch on at full
                                  # weight the instant a node crossed R (and 0 just
                                  # outside), so boundary nodes felt a huge pull
                                  # while their neighbours felt none -> bunching at
                                  # the threshold. Instead the sense weight is full
                                  # for d <= R*(1-band) and smoothstep-ramps to 0 at
                                  # d = R (applied to BOTH prey attract and threat
                                  # repel). This is the fraction of R that ramps;
                                  # 0 restores the hard cutoff, 1 ramps over all R.
    w_repel: float = 1.0         # threat avoidance strength (flee URGENCY). With
                                 # the repel-target fix below the cost no longer
                                 # scales with |node-body|^2, so this is now the
                                 # clean priority of fleeing vs the other terms.
    repel_step: float = 1.0      # flee retraction: a fleeing node aims at a point
                                 # this far TOWARD the body (capped at the body),
                                 # NOT at the body centre. So the repel residual
                                 # magnitude is `repel_step` (constant), and the
                                 # force MAGNITUDE comes from the threat-proximity
                                 # WEIGHT (w_repel * sense-ramp), decoupled from how
                                 # far the node is from the body. Fixes the old
                                 # cost ~ w*|node-body|^2, which yanked far tips
                                 # explosively (and let far nodes catch up to near
                                 # ones -> bunching); now every threatened node
                                 # retracts toward the body at a body-distance-
                                 # INDEPENDENT rate, preserving spacing. Same
                                 # aim-a-short-step trick the explore target uses.
    repel_radius: float = 2.5    # threat keep-out radius
    repel_tip_fraction: float = 0.3  # tip's threat-avoidance vs the body-
                                     # adjacent node's: protect the body, not the
                                     # limb tip - base end recoils hardest (1.0),
                                     # tip least (this fraction)
    # Exploration (off by default): a node sensing no prey is drawn GENTLY to its
    # OWN nearest least-recently-visited cell (cells are marked explored by the SUCKERS;
    # the visit map is whole-body/shared, the attraction is PER-NODE so the arm
    # spreads to cover ground). w_explore << w_reach_terminal and prey preempts
    # it per node, so exploration never outranks hunting.
    explore_enabled: bool = False
    w_explore: float = 1.5       # gentle per-node pull toward its nearest
                                 # unexplored cell (vs 6.0 for prey)
    explore_node_radius: float = 6.0  # how far each node looks for its own
                                      # least-recently-visited cell; small = local search
                                      # = arm stays spread (no balling on one cell)
    explore_target_smooth: float = 0.85  # EMA smoothing of each node's explore
                                      # target, in [0, 1). The least-recently-visited cell
                                      # is a strict argmin over a visit map that
                                      # changes every frame, so it flip-flops
                                      # between adjacent cells frame-to-frame and
                                      # the free arm TIP snaps back and forth
                                      # (high-frequency flicker). Low-pass the
                                      # target: tgt = s*prev + (1-s)*raw. 0
                                      # disables (legacy, flickers); ->1 is very
                                      # smooth but laggier. Only affects explore
                                      # targets, never prey/threat.
    explore_ticks: int = 1000  # lifetime of the recency map, in frames. A cell
                               # is SET to 1.0 when a sucker is on it (not
                               # incremented - dwell doesn't matter), then ticks
                               # down LINEARLY by 1/explore_ticks each frame to 0,
                               # so its value is max(1 - frames_since_visit /
                               # explore_ticks, 0) and it fully reopens exactly
                               # explore_ticks frames after the last visit. Nodes
                               # seek the LOWEST (least recently visited). Larger
                               # = the octopus remembers where it has been for
                               # longer; <=0 means "never fades".
    # The explore-target picker is otherwise threat-blind, so the least-recently-visited
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
    max_hue_change: float = 0.01  # max rgb change per step; the constraint
                                  # threshold (was octo_max_hue_change)
    adjacency_radius: float = 1.0  # how close two suckers must be to count
                                   # as neighbours for the LIMB model


@dataclass(frozen=True)
class LimbConfig:
    rows: int = 16
    cols: int = 2
    min_sucker_distance: float = 0.1  # was octo_min_sucker_distance
    max_sucker_distance: float = 0.3  # was octo_max_sucker_distance
    sucker_col_spacing: float = 0.3  # LATERAL gap between the `cols` suckers on
                                     # ONE node, perpendicular to the centerline
                                     # (the width of the arm). Decoupled from the
                                     # along-arm segment spacing: raise this to
                                     # spread the suckers on a node apart WITHOUT
                                     # changing arm length. Was implicitly
                                     # sucker_distance (pinned at min_sucker_
                                     # distance in ILQR), so 0.1 keeps the look.
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
    # --- propulsion (how the body's CoM translates) ---------------------
    # INTERNAL reproduces the legacy summed-tension drift exactly; REACTION
    # switches to the external-reaction model (crawl + jet) in _propel_body.
    propulsion_mode: PropulsionMode = PropulsionMode.INTERNAL
    crawl_grip_limit: float = 0.15  # REACTION only: per-arm friction/adhesion
                                    # thrust budget. An anchored arm hauls the
                                    # body toward its grip up to this; beyond it
                                    # the sucker SLIPS and adds no more thrust
                                    # (the external-reaction cap the internal
                                    # model lacks).
    crawl_plant_speed: float = 0.08  # REACTION only: a tip that moved less than
                                     # this between frames counts as PLANTED
                                     # (gripping); a faster tip is swinging/
                                     # reaching and provides no crawl reaction,
                                     # so a lone reach no longer glides the body.
    jet_enabled: bool = True         # REACTION only: fire a siphon jet on threat
    jet_trigger_radius: float = 8.0  # a threat within this of the body CENTRE
                                     # fires the escape jet. Kept SEPARATE from
                                     # (and wider than) the arm sensing_radius:
                                     # the arms sense from their extended tips, so
                                     # they recoil from a threat well before it is
                                     # within sensing_radius of the body centre -
                                     # the body should flee just as early, not
                                     # wait for the threat to reach point-blank.
    jet_impulse: float = 0.9         # velocity added to the jet burst each frame
                                     # a threat is within jet_trigger_radius of
                                     # the body (accumulates up to max_jet_velocity)
    jet_decay: float = 0.6           # per-frame decay of jet velocity as the
                                     # mantle refills (0 = one-shot, 1 = never)
    max_jet_velocity: float = 1.2    # escape-speed cap; >> max_body_velocity so
                                     # a jet burst clearly outruns a crawl
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
    sucker_hue_change_conditioned: bool = False  # (delta model only) CONDITION
                                     # the sucker net on max_hue_change: it takes
                                     # (current, surface, max_hue_change) and is
                                     # trained across a range of budgets (each
                                     # sample drawn from [min,max]), so ONE model
                                     # honours any per-step cap passed at
                                     # inference - the network learns to spend a
                                     # small budget greedily vs a large one
                                     # smoothly. Off = the fixed-budget model
                                     # (inference re-clamps to the config value,
                                     # see Octopus._find_color_batched).
    sucker_hue_change_train_range: tuple = (0.01, 0.5)  # (min, max) budget range
                                     # sampled per training example when
                                     # sucker_hue_change_conditioned is on.

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
