"""Headless simulation runner + CLI (record & replay, Phase 3).

The ONE headless loop implementation. Both drivers share it: the CLI
(`python simulator/headless_runner.py --frames 120`) and the websocket
server's `simulate` handler (which calls `HeadlessRunner.run` on a worker
thread). There is exactly one sim-loop order; the recorder is an optional
per-frame sink.

Deliberate deviations from the two legacy entry points, both documented in
RECORD_REPLAY_PLAN.md:

- **Seed-first, then build (D13).** octo_viz and websocket_server disagree on
  construction order, and `AgentGenerator.__init__` is what seeds numpy's
  global RNG - so a surface built before the agents is drawn from unseeded
  state. `run()` calls `np.random.seed(cfg.run.rand_seed)` explicitly before
  constructing anything, making recorded runs reproducible regardless.
- **Frame 0 is the initial post-setup state (D11),** recorded through the same
  explicit find_color/force_color seam as every other frame (same configured
  inference mode) - not the legacy bare `set_color(surf)` heuristic pass. No
  movement, no agents step, no iLQR rows. `frames_recorded == num_frames + 1`.
"""

import os

# Headless: matplotlib is imported at simutil module level and can otherwise
# grab a GUI backend. Set before importing any simulator module.
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import logging
import sys
import time
from typing import NamedTuple

import numpy as np

from simulator.agent_generator import AgentGenerator
from simulator.octopus_generator import Octopus
from simulator.profiling import PROFILER, span
from simulator.sim_recorder import SimRecorder, new_run_id
from simulator.simutil import AgentType, Color, MLMode
from simulator.surface_generator import RandomSurface


def _load_model(cfg):
    """Load the configured inference model, or fall back to NO_MODEL.

    Returns (inference_mode, model). The Keras load and the frame-1 tf.function
    compiles thus happen on the calling thread (off the event loop when the
    server drives this). Moved verbatim from websocket_server's setup.
    """
    inference_mode = cfg.inference.mode
    model = None
    if inference_mode is MLMode.NO_MODEL:
        return inference_mode, None
    try:
        from training.losses import (
            ClampedTargetLoss,
            ConstraintLoss,
            DeltaColorLayer,
        )
        from training.models.model_loader import ModelLoader

        model = ModelLoader(
            cfg.inference_model_path,
            custom_objects={
                "ConstraintLoss": ConstraintLoss,
                "ClampedTargetLoss": ClampedTargetLoss,
                "DeltaColorLayer": DeltaColorLayer,
            },
        ).get_object()
    except Exception as e:
        logging.warning(
            "Could not load model for %s (%s); falling back to heuristic",
            inference_mode,
            e,
        )
        return MLMode.NO_MODEL, None
    return inference_mode, model


def serialize_state(
    octo,
    ag,
    surf,
    iteration,
    visibility_after,
    visibility_before=None,
    color_before=None,
    prey_captured=0,
    prey_captured_this_frame=0,
) -> dict:
    """The per-frame `state` JSON shape (single source of truth).

    Descendant of the old `get_simulation_state`, minus background/fps (the
    background is sent once by the playback layer), plus: per-sucker
    `color_before`, `visibility_score_before`, a stable agent `id`, both `vx`
    and `vy` (the old wire's `velocity` was only vx), and
    `prey_captured_this_frame`. RunStore reproduces this exact shape from the
    DB (unit-tested for key equality).

    color_before: optional nested per-limb list of Color (the colors BEFORE
    this frame's camouflage step); when None, before == after.
    """
    limbs = []
    suckers = []
    for limb_ix, limb in enumerate(octo.limbs):
        limbs.append([{"x": float(pt.x), "y": float(pt.y)} for pt in limb.center_line])
        # Per-node motor state; a sucker inherits its centreline row's state
        # (row = sucker_ix % rows). The tip row shows "gripping" (4) when the arm
        # is crawl-anchored and not already doing something more urgent.
        node_state = getattr(limb, "last_node_state", None)
        rows = limb.rows
        gripping = bool(getattr(limb, "last_gripping", False))
        for sucker_ix, s in enumerate(limb.suckers):
            target = s.get_surf_color_at_this_sucker(surf)
            after_rgb = [float(s.c.r), float(s.c.g), float(s.c.b)]
            if color_before is not None:
                cb = color_before[limb_ix][sucker_ix]
                before_rgb = [float(cb.r), float(cb.g), float(cb.b)]
            else:
                before_rgb = after_rgb
            row = sucker_ix % rows
            st = int(node_state[row]) if node_state is not None else 0
            if gripping and row == rows - 1 and st in (0, 1):
                st = 4  # gripping tip (overrides idle/explore, not threat/prey)
            suckers.append(
                {
                    "x": float(s.x),
                    "y": float(s.y),
                    "color": after_rgb,
                    "color_before": before_rgb,
                    "target_color": [float(target.r), float(target.g), float(target.b)],
                    "state": st,
                    "limb": limb_ix,  # so the analyzer groups suckers by arm (the
                                      # live-frame path lacked this; playback has it)
                }
            )

    agents = []
    for i, agent in enumerate(ag.agents if ag is not None else []):
        agents.append(
            {
                "id": int(getattr(agent, "_rec_id", i)),
                "x": float(agent.x),
                "y": float(agent.y),
                "type": ("prey" if agent.agent_type == AgentType.PREY else "predator"),
                "vx": float(agent.vx),
                "vy": float(agent.vy),
                "angle": float(agent.t),
                "behavior": int(getattr(agent, "behavior", 0)),
            }
        )

    return {
        "octopus": {
            "head": {"x": float(octo.x), "y": float(octo.y),
                     "theta": float(getattr(octo, "theta", 0.0))},
            "limbs": limbs,
            "suckers": suckers,
            # Behavior policy at the limb and body levels (per-sucker is on each
            # sucker's "state"). Same code convention; for colour-coding.
            "limb_states": [int(getattr(limb, "last_limb_state", 0))
                            for limb in octo.limbs],
            "body_state": int(getattr(octo, "last_body_state", 0)),
        },
        "agents": agents,
        "metadata": {
            "iteration": int(iteration),
            "visibility_score": float(visibility_after),
            "visibility_score_before": float(
                visibility_after if visibility_before is None else visibility_before
            ),
            "prey_captured": int(prey_captured),
            "prey_captured_this_frame": int(prey_captured_this_frame),
        },
    }


class RunSummary(NamedTuple):
    status: str  # complete | cancelled | failed  (D17 vocabulary)
    frames_recorded: int  # num_frames + 1 when complete (frame 0 = initial)
    elapsed_s: float
    final_state: dict  # serialize_state of the last recorded frame.
    # CLI-only convenience - the server reads
    # simulate_complete.final_state via RunStore (D15).


class HeadlessRunner:
    """Builds the scenario and steps N frames on the calling thread.

    Cheap to construct anywhere (stores args only). `run()` does all the work
    and owns the DuckDB connection's whole lifetime on one thread.
    """

    def __init__(self, cfg, run_id=None, label="", db_path=None, setup=None):
        if cfg.run.num_iterations <= 0:
            raise ValueError(
                "num_iterations must be > 0 for a recorded run "
                f"(got {cfg.run.num_iterations}); the -1=infinite convention "
                "is rejected here"
            )
        self.cfg = cfg
        self.run_id = run_id or new_run_id()
        self.label = label
        self.db_path = db_path
        self.setup = setup or {}

    def run(self, progress_cb=None, should_stop=None) -> RunSummary:
        cfg = self.cfg
        num_frames = cfg.run.num_iterations

        # Seed FIRST, then build (D13) - reproducible regardless of the order
        # AgentGenerator's global-RNG seeding fires in.
        np.random.seed(cfg.run.rand_seed)
        surf = RandomSurface(cfg)
        octo = Octopus(cfg, start_xy=self.setup.get("octo_start"))
        ag = AgentGenerator(cfg)
        agent_positions = self.setup.get("agent_positions")
        if agent_positions:
            ag.place_agents(agent_positions)
        else:
            ag.generate(num_agents=cfg.agents.count)
            # Guarantee some food: `count`'s 50/50 flips can leave few/no prey.
            ag.generate(num_agents=3, fixed_agent_type=AgentType.PREY)
        inference_mode, model = _load_model(cfg)

        recorder = None
        status = "complete"
        t0 = time.perf_counter()
        frames_recorded = 0
        last_state = {}

        try:
            if cfg.output.record_run:
                recorder = SimRecorder(
                    cfg, run_id=self.run_id, db_path=self.db_path, run_label=self.label
                )
                recorder.record_surface(surf)

            def do_camouflage(frame, captured_this_frame, f_start):
                """The shared before/after color seam for one frame.

                f_start: perf_counter at the frame's start; wall_ms measured up
                to (and stored at) the recorder's end_frame.
                """
                nonlocal frames_recorded, last_state
                # Snapshot BEFORE colors (for the recorder's seam and the CLI
                # final_state); visibility_before from the same pre-step state.
                before = [
                    [Color(s.c.r, s.c.g, s.c.b) for s in limb.suckers]
                    for limb in octo.limbs
                ]
                vis_before = float(octo.visibility(surf))
                if recorder is not None:
                    with span("record.snapshot_state"):
                        recorder.begin_frame(frame)
                        recorder.snapshot_state(octo, ag, surf, captured_this_frame)
                with span("find_color"):
                    color_matrix = octo.find_color(surf, inference_mode, model)
                for limb, c_array in zip(octo.limbs, color_matrix, strict=True):
                    limb.force_color(c_array)
                if recorder is not None:
                    with span("record.snapshot_colors"):
                        recorder.snapshot_colors(color_matrix)
                with span("visibility"):
                    vis_after = float(octo.visibility(surf))
                wall_ms = (time.perf_counter() - f_start) * 1000.0
                if recorder is not None:
                    with span("record.end_frame"):
                        recorder.end_frame(wall_ms=wall_ms)
                frames_recorded += 1
                with span("serialize_state"):
                    last_state = serialize_state(
                        octo,
                        ag,
                        surf,
                        frame,
                        vis_after,
                        visibility_before=vis_before,
                        color_before=before,
                        prey_captured=(ag.prey_captured if ag else 0),
                        prey_captured_this_frame=captured_this_frame,
                    )
                return vis_after, wall_ms

            # Frame 0: initial post-setup state, same seam, no movement (D11).
            with span("frame"), span("camouflage"):
                vis_prev, _ = do_camouflage(0, 0, time.perf_counter())

            # Frames 1..N.
            for frame in range(1, num_frames + 1):
                if should_stop is not None and should_stop():
                    status = "cancelled"
                    break
                f_start = time.perf_counter()
                with span("frame"):
                    with span("agents.increment"):
                        ag.increment_all(octo, visibility=vis_prev)
                    with span("octo.move"):
                        octo.move(ag)
                    with span("capture"):
                        captured = ag.remove_captured_prey(octo)
                    with span("camouflage"):
                        vis_after, frame_ms = do_camouflage(frame, captured, f_start)
                vis_prev = vis_after
                if progress_cb is not None:
                    progress_cb(
                        {
                            "frame": frame,
                            "num_frames": num_frames,
                            "visibility_score": vis_after,
                            "prey_captured": int(ag.prey_captured if ag else 0),
                            "frame_ms": frame_ms,
                            "elapsed_s": time.perf_counter() - t0,
                            # Full per-frame geometry for a live preview: the same
                            # shape RunStore serves in playback, so the client
                            # renders it with the same drawWorld path. The server
                            # coalesces (latest-wins), so this is the newest frame
                            # only - no per-frame backlog.
                            "state": last_state,
                        }
                    )

            elapsed = time.perf_counter() - t0
            return RunSummary(
                status=status,
                frames_recorded=frames_recorded,
                elapsed_s=elapsed,
                final_state=last_state,
            )
        except Exception:
            status = "failed"
            raise
        finally:
            if recorder is not None:
                recorder.close(status)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Headless octopus simulation recorder (record & replay)."
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="number of frames to step (default: the RECORD profile's num_iterations)",
    )
    parser.add_argument(
        "--label", type=str, default="", help="human-readable run label"
    )
    parser.add_argument(
        "--no-ilqr-history",
        action="store_true",
        help="disable per-iteration iLQR history capture",
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        help="enable sucker exploration (off in the RECORD profile by default); "
        "records the visit-count map so the analyzer's exploration overlay lights up",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="time the sim and print a hierarchical breakdown of where the "
        "frame loop spends its time (per-phase / per-limb / iLQR solve)",
    )
    parser.add_argument(
        "--eager-backward",
        action="store_true",
        help="use the EAGER iLQR backward pass instead of the (default) "
        "graph-compiled one - the slower reference path, for A/B profiling",
    )
    args = parser.parse_args(argv)

    # Import profiles here so the module stays import-light for the server.
    from dataclasses import replace

    from octopus_ai.config import (  # noqa: F401 (profiles offered for swap)
        RECORD,
        VIZ_ILQR,
        print_config,
    )

    cfg = RECORD
    if args.frames is not None:
        cfg = replace(cfg, run=replace(cfg.run, num_iterations=args.frames))
    if args.no_ilqr_history:
        cfg = replace(cfg, output=replace(cfg.output, record_ilqr_history=False))
    if args.explore:
        o = cfg.octopus
        cfg = replace(cfg, octopus=replace(
            o, limb=replace(o.limb, ilqr=replace(
                o.limb.ilqr, explore_enabled=True))))
    if args.eager_backward:
        o = cfg.octopus
        cfg = replace(cfg, octopus=replace(
            o, limb=replace(o.limb, ilqr=replace(
                o.limb.ilqr, compiled_backward=False))))

    print_config(cfg, "headless_runner CONFIG")
    runner = HeadlessRunner(cfg, label=args.label)

    def progress(info):
        print(
            f"  frame {info['frame']}/{info['num_frames']}  "
            f"vis={info['visibility_score']:.4f}  "
            f"prey={info['prey_captured']}  "
            f"{info['frame_ms']:.0f} ms"
        )

    print(f"Recording run {runner.run_id} ...")
    if args.profile:
        with PROFILER.profile():
            summary = runner.run(progress_cb=progress)
    else:
        summary = runner.run(progress_cb=progress)
    db_path = runner.db_path or os.path.join("logs", "runs", f"{runner.run_id}.duckdb")
    print(
        f"\nRun {runner.run_id}: {summary.status}, "
        f"{summary.frames_recorded} frames recorded in "
        f"{summary.elapsed_s:.1f}s"
    )
    print(f"  -> {db_path}")
    if args.profile:
        print("\n" + PROFILER.render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
