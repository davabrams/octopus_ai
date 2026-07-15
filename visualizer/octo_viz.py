""" Octopus visualizer """
import os
import sys
import time
from dataclasses import replace

# This lives in visualizer/ but imports top-level project modules; put the repo
# root on sys.path so `python visualizer/octo_viz.py` works from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from tensorflow import keras

from octopus_ai.config import (  # noqa: F401  (DEBUG/DEFAULT are profiles)
    DEBUG,
    DEFAULT,
    VIZ,
    print_config,
)
from octopus_ai.perf import PerfTracker
from simulator.agent_generator import AgentGenerator
from simulator.octopus_generator import Octopus
from simulator.simutil import (
    Color,
    MLMode,
    MovementMode,
    display_refresh,
    setup_display,
)
from simulator.surface_generator import RandomSurface
from simulator.force_logger import ForceLogger
from simulator.frame_recorder import FrameRecorder
from training.losses import (
    ClampedTargetLoss,
    ConstraintLoss,
    DeltaColorLayer,
)
from training.models.model_loader import ModelLoader

# %% Pick a profile %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Swap this one line instead of editing (and forgetting to revert) a pile of
# flags. To watch the locomotion with force arrows use VIZ; to also record a
# force DB and stitch an MP4 use DEBUG. Derive an ad-hoc variant with
# dataclasses.replace, e.g.
#     CFG = replace(VIZ, octopus=replace(VIZ.octopus,
#                                        movement_mode=MovementMode.SPRING_CHAIN))
#     CFG = replace(VIZ, octopus=replace(VIZ.octopus,
#                                        movement_mode=MovementMode.SPRING_CHAIN))
#
# iLQR limb motor control (ARCHITECTURE.md §11.4): each arm reaches toward
# nearby prey via its own compiled TensorFlow iLQR controller, receding-horizon
# (MPC). The BODY (octopus movement_mode=ILQR too) drifts by the summed base
# reactions of all eight arms - it follows the collective reach, one frame
# lagged. Color inference is pinned to the fast NO_MODEL heuristic so no
# trained model is needed.
#
# Heads up on speed: every arm compiles its own graph on the FIRST frame (a
# multi-second pause), then each frame runs 8 independent solves (~1-2 s/frame).
# For snappier playback, thin the octopus - fewer/shorter arms - e.g. add
#     num_arms=4, limb=replace(..., rows=8, ...)
CFG = replace(
    VIZ,
    inference=replace(VIZ.inference, mode=MLMode.NO_MODEL),
    output=replace(VIZ.output, highlight_octopus=True,   # outline the camouflaged octopus
                   track_performance=True),               # per-step timing + memory summary
    octopus=replace(
        VIZ.octopus,
        movement_mode=MovementMode.ILQR,   # body follows the arms' pull
        limb=replace(VIZ.octopus.limb, movement_mode=MovementMode.ILQR),
    ),
)


# %% Generate game scenario %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print_config(CFG, "octo_viz CONFIG")
INFERENCE_MODE = CFG.inference.mode

ag = AgentGenerator(CFG)
octo = Octopus(CFG)
surf = RandomSurface(CFG)
ag.generate(num_agents=CFG.agents.count)
octo.set_color(surf)

model = None

if INFERENCE_MODE is not MLMode.NO_MODEL:
    # Override `model` with the model from disk. The config resolves the
    # path, so no MLMode-keyed table lookup here.
    custom_objects = {
        "ConstraintLoss": ConstraintLoss,
        "ClampedTargetLoss": ClampedTargetLoss,
        "DeltaColorLayer": DeltaColorLayer,
    }
    model = ModelLoader(
        file_name_or_ml_mode=CFG.inference_model_path,
        custom_objects=custom_objects,
    ).get_object()
    assert isinstance(model, keras.Model), (
        f"Expected keras model, got {type(model)}"
    )


# %% Visualizer %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NUM_ITERATIONS = CFG.run.num_iterations
DEBUG_MODE = CFG.output.debug_mode
SHOW_FORCES = CFG.output.show_forces
HIGHLIGHT_OCTOPUS = CFG.output.highlight_octopus

force_logger = (ForceLogger(run_label="octo_viz", config=CFG)
                if CFG.output.log_forces else None)
frame_recorder = (FrameRecorder(fps=CFG.output.video_fps)
                  if CFG.output.save_images else None)
perf = PerfTracker(enabled=CFG.output.track_performance, label="octo_viz")

fig, ax = setup_display()
fig.show()

font = {'family': 'monospace', 'color': 'green', 'size': 20}
y = fig.text(.1, .025, "", fontdict=font)
fig.waitforbuttonpress()

i: int = 0
while i != NUM_ITERATIONS:
    t_start: int = time.time_ns()
    i += 1

    # 1) advance the simulation
    with perf.track("agents.increment"):
        ag.increment_all(octo)
    with perf.track("octopus.move"):
        octo.move(ag)

    # 2) any prey touched by a sucker is caught and removed
    with perf.track("capture"):
        captured = ag.remove_captured_prey(octo)
    if captured:
        print(f"  caught {captured} prey (total {ag.prey_captured})")

    if force_logger is not None:
        force_logger.log_frame(i, octo)

    # 2) run inference / camouflage for the new positions
    with perf.track("find_color"):
        color_matrix = octo.find_color(surf, INFERENCE_MODE, model)
    assert isinstance(color_matrix, list)
    assert isinstance(color_matrix[0], list)
    assert isinstance(color_matrix[0][0], Color)
    for ix, l in enumerate(octo.limbs):
        l.force_color(color_matrix[ix])

    # 3) draw the updated state and flush it to the window
    with perf.track("render"):
        display_refresh(ax, octo, ag, surf, debug_mode=DEBUG_MODE,
                        show_forces=SHOW_FORCES,
                        highlight_octopus=HIGHLIGHT_OCTOPUS)
    with perf.track("visibility"):
        y.set_text(f"Visibility = {octo.visibility(surf):.4f}   "
                   f"Prey caught = {ag.prey_captured}")

    if frame_recorder is not None:
        frame_recorder.save_frame(fig)

    # plt.pause both renders the canvas and yields to the GUI event loop;
    # without it the window never repaints (this was the blank-window bug).
    with perf.track("gui.pause"):
        plt.pause(0.001)

    perf.end_frame()
    t_end: int = time.time_ns()
    print(
        f"Iteration: {i} complete with "
        f"Δt = {(t_end - t_start)/1000000000} sec"
    )

if force_logger is not None:
    force_logger.close()
    print(f"Force log written to {force_logger.db_path} "
          f"(run_id={force_logger.run_id})")

if frame_recorder is not None:
    video_path = frame_recorder.stitch_video(keep_frames=True)
    if video_path:
        print(f"Video written to {video_path} "
              f"({frame_recorder.frame_count} frames @ "
              f"{frame_recorder.fps} fps)")

perf.print_summary()
