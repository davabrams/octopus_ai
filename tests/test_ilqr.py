"""Tests for the single-limb iLQR motor-control tier (simulator/ilqr).

These exercise the compiled ArmController end to end: a tip that senses a target
reaches it with the chain held near rest spacing, one controller serves multiple
targets (graph reuse, no retrace), and an arm sensing nothing barely moves.
Attraction and repulsion are PER-NODE (node-autonomous sensing): each node has
its own target/threat + weight, so the tests build those arrays via `_solve`.
Configs are kept small so the suite stays fast despite the one-time graph compile.
"""
from itertools import pairwise

import numpy as np
import tensorflow as tf
from helpers import make_config

from simulator.ilqr.arm import ArmController
from simulator.ilqr.residuals import chain_positions, segment_lengths
from simulator.ilqr.solver import make_solver
from simulator.simutil import MovementMode


def _tip_and_segments(res, base):
    base_t = tf.constant(base, tf.float32)
    chain = chain_positions(res.x_traj[-1], base_t)
    tip = chain.numpy()[-1]
    seg = segment_lengths(chain).numpy()
    return tip, seg


def _min_node_err(res, target):
    """Distance from the target to the NEAREST free node."""
    nodes = res.x_traj[-1].numpy().reshape(-1, 2)
    return float(np.linalg.norm(nodes - np.array(target, float), axis=1).min())


def _solve(ctrl, base, x0, *, target=None, tip_only=False, repel_to=None, **kw):
    """Per-node solve helper: build the attract/repel arrays and call solve().

    target:   attract every node (or just the tip, tip_only=True) toward it.
    repel_to: flee = retract every node toward this point (the body centre).
    Reproduces simple reach/flee scenarios on the per-node interface.
    """
    n = ctrl.n_free
    at = np.zeros((n, 2), np.float32)
    asw = np.zeros(n, np.float32)
    if target is not None:
        at[:] = target
        if tip_only:
            asw[-1] = np.sqrt(ctrl.w_reach_terminal)
        else:
            asw[:] = np.sqrt(ctrl.w_reach_terminal)
    rt = np.zeros((n, 2), np.float32)
    rsw = np.zeros(n, np.float32)
    if repel_to is not None:
        rt[:] = repel_to
        rsw[:] = np.sqrt(ctrl.w_repel)
    return ctrl.solve(base, at, asw, rt, rsw, x0=x0, **kw)


def test_reach_reachable_target_converges():
    """A tip that senses a target reaches it, with the chain near rest spacing.

    Per-node attract: only the tip senses the target here (attract_sw nonzero at
    the tip only), so the arm extends to it and the springs keep the chain
    coherent (no balling up).
    """
    ctrl = ArmController(n_free=5, rest_length=1.0, horizon=8, max_iters=50)
    base = [0.0, 0.0]
    x0 = ctrl.straight_arm(base, init_angle=0.0)
    target = [2.5, 2.0]  # |target| ~= 3.2 < reach (5)

    res = _solve(ctrl, base, x0, target=target, tip_only=True)
    tip, seg = _tip_and_segments(res, base)

    assert res.converged, f"did not converge in {res.iterations} iters"
    assert float(np.linalg.norm(tip - np.array(target, float))) < 0.15
    # Springs keep the chain coherent: no segment far from rest.
    assert seg.min() > 0.7 and seg.max() < 1.4, f"segments drifted: {seg}"


def test_one_controller_serves_multiple_targets():
    """The compiled graph is reused across targets (no per-target retrace)."""
    ctrl = ArmController(n_free=5, rest_length=1.0, horizon=8, max_iters=50)
    base = [0.0, 0.0]
    x0 = ctrl.straight_arm(base, init_angle=0.0)

    for target in ([2.0, 2.0], [-2.0, 1.0], [0.0, -3.0]):
        res = _solve(ctrl, base, x0, target=target, tip_only=True)
        tip, _ = _tip_and_segments(res, base)
        err = float(np.linalg.norm(tip - np.array(target, float)))
        assert err < 0.25, f"target {target}: tip err {err}"


def test_spring_stiffen_is_superlinear_and_signed():
    """The stiffen residual is sw*dev*|dev| (= sw*dev^2, sign-preserved): signed
    so it corrects both stretch and compression, and super-linear (doubling dev
    quadruples the residual -> a cubic restoring FORCE)."""
    from simulator.ilqr.residuals import spring_stiffen_residual
    base = tf.constant([0.0, 0.0], tf.float32)
    # single free node; rest=1.0 so seg_len == the node's x.
    r_stretch = spring_stiffen_residual(tf.constant([2.0, 0.0], tf.float32),
                                        base, rest=1.0, sw=1.0).numpy()[0]
    r_compress = spring_stiffen_residual(tf.constant([0.4, 0.0], tf.float32),
                                         base, rest=1.0, sw=1.0).numpy()[0]
    r_more = spring_stiffen_residual(tf.constant([3.0, 0.0], tf.float32),
                                     base, rest=1.0, sw=1.0).numpy()[0]
    assert abs(r_stretch - 1.0) < 1e-5      # dev=+1  -> +1
    assert abs(r_compress - (-0.36)) < 1e-5  # dev=-0.6 -> -(0.6^2), signed
    assert abs(r_more - 4.0) < 1e-5          # dev=+2  -> +4 (dev^2: super-linear)


def test_spring_and_bend_deadbands_are_free_zones():
    """Within the deadbands the spring/bending residuals are zero (the node moves
    for free, paying only effort); only the overrun is penalized."""
    from simulator.ilqr.residuals import spring_residual, bending_residual
    base = tf.constant([0.0, 0.0], tf.float32)
    # Segment stretched 0.2 with slack 0.25 -> free.
    r_in = spring_residual(tf.constant([1.2, 0.0], tf.float32), base,
                           rest=1.0, sw_spring=2.0, slack=0.25).numpy()[0]
    assert abs(r_in) < 1e-6
    # Stretched 0.5 -> only the 0.25 excess is sprung.
    r_out = spring_residual(tf.constant([1.5, 0.0], tf.float32), base,
                            rest=1.0, sw_spring=2.0, slack=0.25).numpy()[0]
    assert abs(r_out - 2.0 * 0.25) < 1e-5           # sw * (0.5 - 0.25)
    # Bending: small curvatures (|curv| 0.05, 0.1) below a 0.3 deadzone are free.
    xb = tf.constant([1.0, 0.0, 2.0, 0.05, 3.0, 0.0], tf.float32)
    rb = bending_residual(xb, base, sw_bend=1.0, deadzone=0.3).numpy()
    assert np.allclose(rb, 0.0, atol=1e-5)


def test_effort_stiffen_is_superlinear_per_node():
    """The velocity-stiffen residual is sw*|u_i|*u_i per node (cost ~ |u_i|^4):
    a per-node speed penalty that rises super-linearly to forbid teleportation."""
    from simulator.ilqr.residuals import effort_stiffen_residual
    # two nodes: one moving (3,4) (speed 5), one still.
    u = tf.constant([3.0, 4.0, 0.0, 0.0], dtype=tf.float32)
    r = effort_stiffen_residual(u, sw=1.0).numpy()
    # node 0 residual = |u|*u = 5*(3,4) = (15,20); node 1 = 0.
    assert abs(r[0] - 15.0) < 1e-4 and abs(r[1] - 20.0) < 1e-4
    assert abs(r[2]) < 1e-4 and abs(r[3]) < 1e-4
    # cost (squared) for node 0 = 15^2+20^2 = 625 = |u|^4 = 5^4. Super-linear.
    assert abs((r[0] ** 2 + r[1] ** 2) - 625.0) < 1e-2


def test_repel_residual_grades_body_over_tip():
    """Flee pulls each node toward the body target; node_sw (with the body>tip
    grade) sets the strength. At EQUAL distance from the body the body-adjacent
    node (larger weight) pays more than the tip."""
    from simulator.ilqr.residuals import repel_residual
    # two free nodes, both 1 unit from the body target at the origin.
    x = tf.constant([1.0, 0.0, 0.0, 1.0], dtype=tf.float32)  # (1,0) and (0,1)
    targets = tf.constant([[0.0, 0.0], [0.0, 0.0]], dtype=tf.float32)  # body
    node_sw = tf.constant([1.0, np.sqrt(0.3)], dtype=tf.float32)  # body, tip
    r = repel_residual(x, targets, node_sw).numpy()  # (4,)
    c0 = r[0] ** 2 + r[1] ** 2   # body-adjacent node
    c1 = r[2] ** 2 + r[3] ** 2   # tip node
    assert c0 > c1 > 0.0
    assert abs(c1 / c0 - 0.3) < 1e-5  # tip pays 0.3x


def test_flee_retracts_arm_toward_body():
    """Fleeing pulls the arm IN toward the body ('scrunch up'), so the tip ends
    up closer to the base than when idle - NOT pushed away from the threat."""
    ctrl = ArmController(n_free=5, rest_length=1.0, horizon=8, max_iters=50)
    base = [0.0, 0.0]
    x0 = ctrl.straight_arm(base, init_angle=0.0)  # arm extended along +x

    def tip_dist(res):
        nodes = res.x_traj[-1].numpy().reshape(-1, 2)
        return float(np.linalg.norm(nodes[-1] - np.array(base, float)))

    fleeing = _solve(ctrl, base, x0, repel_to=base)  # retract toward the body
    idle = _solve(ctrl, base, x0)                    # nothing sensed
    assert tip_dist(fleeing) < tip_dist(idle) - 0.2, \
        "flee did not retract the arm toward the body"


def test_hold_barely_moves():
    """With nothing sensed (all attract/repel weights 0), effort/spring keep the
    arm still - no attractor pulls any node."""
    ctrl = ArmController(n_free=5, rest_length=1.0, horizon=8, max_iters=30)
    base = [0.0, 0.0]
    x0 = ctrl.straight_arm(base, init_angle=0.0)
    nodes0 = tf.reshape(x0, (-1, 2)).numpy()

    res = _solve(ctrl, base, x0)  # no target, no threat
    nodes = res.x_traj[-1].numpy().reshape(-1, 2)

    assert float(np.linalg.norm(nodes - nodes0, axis=1).max()) < 0.1, \
        "arm moved despite sensing nothing"


# --------------------------------------------------------------------------
# Per-iteration solve history (record & replay, Phase 1)
# --------------------------------------------------------------------------

def test_history_off_is_identical_and_none():
    """Recording off must not change the solve result, and history is None.

    Also asserts the compiled kernels are not re-traced by a recorded solve:
    record_history is an eager-Python flag that never reaches a @tf.function.
    """
    ctrl = ArmController(n_free=5, rest_length=1.0, horizon=8, max_iters=10)
    base = [0.0, 0.0]
    x0 = ctrl.straight_arm(base, init_angle=0.0)
    target = [2.5, 2.0]

    off = _solve(ctrl, base, x0, target=target, tip_only=True)
    assert off.history is None

    # Kernel tracing counts before/after a recorded solve are unchanged.
    kernels = ctrl._solve._kernels
    before = [k.experimental_get_tracing_count() for k in kernels]
    on = _solve(ctrl, base, x0, target=target, tip_only=True, record_history=True)
    after = [k.experimental_get_tracing_count() for k in kernels]
    assert before == after, "recorded solve retraced a compiled kernel"

    # Deterministic solver, no RNG: identical inputs -> identical outputs.
    assert np.allclose(off.x_traj.numpy(), on.x_traj.numpy())
    assert np.allclose(off.u_traj.numpy(), on.u_traj.numpy())
    assert float(off.cost) == float(on.cost)
    assert off.iterations == on.iterations
    assert off.converged == on.converged


def test_history_length_indices_and_shapes():
    """len(history) == iterations + 1; iter 0 is 'init'; shapes/indices hold."""
    ctrl = ArmController(n_free=5, rest_length=1.0, horizon=8, max_iters=10)
    base = [0.0, 0.0]
    x0 = ctrl.straight_arm(base, init_angle=0.0)
    res = _solve(ctrl, base, x0, target=[2.5, 2.0], tip_only=True,
                 record_history=True)

    h = res.history
    assert len(h) == res.iterations + 1
    assert h[0].phase == "init"
    assert [rec.iteration for rec in h] == list(range(len(h)))

    for rec in h:
        if rec.x_traj is not None:  # init + accepted carry a trajectory
            assert rec.x_traj.shape == (ctrl.horizon + 1, 2 * ctrl.n_free)
            assert rec.x_traj.dtype == np.float32
        if rec.phase == "accepted":
            assert rec.alpha in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01)


def test_history_accepted_costs_strictly_decrease():
    """Accepted iterations lower the cost; the last matches the result."""
    ctrl = ArmController(n_free=5, rest_length=1.0, horizon=8, max_iters=10)
    base = [0.0, 0.0]
    x0 = ctrl.straight_arm(base, init_angle=0.0)
    res = _solve(ctrl, base, x0, target=[2.5, 2.0], tip_only=True,
                 record_history=True)

    accepted = [rec for rec in res.history if rec.phase == "accepted"]
    costs = [rec.cost for rec in accepted]
    assert all(b < a for a, b in pairwise(costs)), \
        f"accepted costs not strictly decreasing: {costs}"
    assert accepted[-1].cost == float(res.cost)
    assert np.allclose(accepted[-1].x_traj, res.x_traj.numpy())


def test_rejected_iterations_carry_no_trajectory():
    """Cost already at its minimum => nothing improves => linesearch_fail.

    Residual = x with x0 = 0 sits exactly at the global min (cost 0), so the
    backward pass yields a zero step and the line search can never lower the
    cost. Uses make_solver directly with a tiny mu ceiling so escalation
    exhausts fast. Rejected entries must have x_traj/alpha/rel_improve None and
    an unchanged cost; mu is non-decreasing across them. (Residuals depend on x
    so the autodiff Jacobians stay defined, unlike identically-zero costs.)
    """
    n = 3

    def dynamics(x, u):
        return x + u

    def running_cost(x, u, params):
        # Depends on both x and u (so the autodiff Jacobians stay defined);
        # minimized at x = u = 0, exactly where the solve starts.
        return tf.concat([x, u], axis=0)

    def terminal_cost(x, params):
        return x

    solve = make_solver(dynamics, running_cost, terminal_cost, control_dim=n,
                        max_iters=10, mu_init=1.0, mu_max=4.0, mu_factor=2.0)
    x0 = np.zeros(n, dtype=np.float32)
    u_init = np.zeros((4, n), dtype=np.float32)
    res = solve(x0, np.zeros(1, dtype=np.float32), u_init,
                record_history=True)

    non_init = [rec for rec in res.history if rec.phase != "init"]
    assert non_init, "expected at least one solver iteration"
    for rec in non_init:
        assert rec.phase == "linesearch_fail"
        assert rec.x_traj is None
        assert rec.alpha is None
        assert rec.rel_improve is None
        assert rec.cost == res.history[0].cost  # cost never changed
    mus = [rec.mu for rec in res.history]
    assert all(b >= a for a, b in pairwise(mus)), f"mu decreased: {mus}"


def test_limb_integration_populates_history():
    """A Limb in ILQR mode drains solve history + metadata onto itself."""
    from simulator.octopus_generator import Limb

    cfg = make_config(record_ilqr_history=True,
                      octo_movement_mode=MovementMode.ILQR,
                      limb_movement_mode=MovementMode.ILQR,
                      octo_ilqr_horizon=4, octo_ilqr_max_iters=3, limb_rows=6)
    limb = Limb(5.0, 5.0, 0.0, cfg)

    # Frame 1: no warm start yet.
    limb.move(5.0, 5.0, agents=[])
    assert limb.last_ilqr_history is not None
    assert limb.last_ilqr_meta is not None
    assert not limb.last_ilqr_meta["attract_sw"].any()  # senses nothing
    assert not limb.last_ilqr_meta["repel_sw"].any()
    assert limb.last_ilqr_meta["u_init"] is None
    assert limb.last_ilqr_history[0].phase == "init"

    # Frame 2: warm start present with the expected shape.
    limb.move(5.0, 5.0, agents=[])
    u_init = limb.last_ilqr_meta["u_init"]
    assert u_init is not None
    assert u_init.shape == (cfg.octopus.limb.ilqr.horizon, 2 * (6 - 1))


def test_limb_integration_flag_off_stays_none():
    """With recording off, the limb never populates history/meta."""
    from simulator.octopus_generator import Limb

    cfg = make_config(octo_movement_mode=MovementMode.ILQR,
                      limb_movement_mode=MovementMode.ILQR,
                      octo_ilqr_horizon=4, octo_ilqr_max_iters=3, limb_rows=6)
    limb = Limb(5.0, 5.0, 0.0, cfg)
    limb.move(5.0, 5.0, agents=[])
    assert limb.last_ilqr_history is None
    assert limb.last_ilqr_meta is None
