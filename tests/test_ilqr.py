"""Tests for the single-limb iLQR motor-control tier (simulator/ilqr).

These exercise the compiled ArmController end to end: a reachable target should
be reached with the chain held near its rest spacing, one controller should
serve multiple targets (graph reuse, no retrace), and a target at the current
tip should produce almost no motion. Configs are kept small (few nodes, short
horizon) so the suite stays fast despite the one-time graph compile.
"""
import numpy as np
import tensorflow as tf

from simulator.ilqr.arm import (
    ArmController,
    _chain_positions,
    _segment_lengths,
)


def _tip_and_segments(res, base):
    base_t = tf.constant(base, tf.float32)
    chain = _chain_positions(res.x_traj[-1], base_t)
    tip = chain.numpy()[-1]
    seg = _segment_lengths(chain).numpy()
    return tip, seg


def test_reach_reachable_target_converges():
    """A target within reach is reached, with segments near rest length."""
    ctrl = ArmController(n_free=5, rest_length=1.0, horizon=8, max_iters=50)
    base = [0.0, 0.0]
    x0 = ctrl.straight_arm(base, init_angle=0.0)
    target = [2.5, 2.0]  # |target| ~= 3.2 < reach (5)

    res = ctrl.solve(base_xy=base, target=target, x0=x0)
    tip, seg = _tip_and_segments(res, base)

    assert res.converged, f"did not converge in {res.iterations} iters"
    tip_err = float(np.linalg.norm(tip - np.array(target, float)))
    assert tip_err < 0.15, f"tip {tip} missed target {target} by {tip_err}"
    # Springs keep the chain coherent: no segment far from rest.
    assert seg.min() > 0.7 and seg.max() < 1.4, f"segments drifted: {seg}"


def test_one_controller_serves_multiple_targets():
    """The compiled graph is reused across targets (no per-target retrace)."""
    ctrl = ArmController(n_free=5, rest_length=1.0, horizon=8, max_iters=50)
    base = [0.0, 0.0]
    x0 = ctrl.straight_arm(base, init_angle=0.0)

    for target in ([2.0, 2.0], [-2.0, 1.0], [0.0, -3.0]):
        res = ctrl.solve(base_xy=base, target=target, x0=x0)
        tip, _ = _tip_and_segments(res, base)
        tip_err = float(np.linalg.norm(tip - np.array(target, float)))
        assert tip_err < 0.25, f"target {target}: tip {tip} err {tip_err}"


def test_threat_repulsion_pushes_arm_away():
    """With a threat present the arm keeps farther from it than without one."""
    ctrl = ArmController(n_free=5, rest_length=1.0, horizon=8, max_iters=50,
                         repel_radius=2.5)
    base = [0.0, 0.0]
    x0 = ctrl.straight_arm(base, init_angle=0.0)  # arm along +x
    threat = [2.5, 0.6]                            # just off the arm's side
    target = [4.0, -1.0]                           # reach away from the threat

    def min_dist_to_threat(res):
        nodes = res.x_traj[-1].numpy().reshape(-1, 2)
        return float(np.linalg.norm(nodes - np.array(threat, float),
                                    axis=1).min())

    with_threat = ctrl.solve(base, target, x0, threat=threat)
    without_threat = ctrl.solve(base, target, x0, threat=None)

    assert (min_dist_to_threat(with_threat)
            > min_dist_to_threat(without_threat) + 0.2), \
        "repulsion did not push the arm away from the threat"


def test_target_at_tip_barely_moves():
    """With the target at the current tip, effort cost keeps the arm still."""
    ctrl = ArmController(n_free=5, rest_length=1.0, horizon=8, max_iters=30)
    base = [0.0, 0.0]
    x0 = ctrl.straight_arm(base, init_angle=0.0)
    tip0 = tf.reshape(x0, (-1, 2)).numpy()[-1]

    res = ctrl.solve(base_xy=base, target=tip0.tolist(), x0=x0)
    tip, _ = _tip_and_segments(res, base)

    assert float(np.linalg.norm(tip - tip0)) < 0.1, "arm moved despite goal=tip"
