"""Tests for the single-limb iLQR motor-control tier (simulator/ilqr).

These exercise the compiled ArmController end to end: a reachable target should
be reached with the chain held near its rest spacing, one controller should
serve multiple targets (graph reuse, no retrace), and a held arm (reach off)
should produce almost no motion. Attraction is WHOLE-ARM (every node pulls
toward the target, not just the tip), so "reached" means the nearest node lands
on the target. Configs are kept small (few nodes, short horizon) so the suite
stays fast despite the one-time graph compile.
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
    """Distance from the target to the NEAREST free node (whole-arm reach)."""
    nodes = res.x_traj[-1].numpy().reshape(-1, 2)
    return float(np.linalg.norm(nodes - np.array(target, float), axis=1).min())


def test_reach_reachable_target_converges():
    """A target within reach is reached by the arm, segments near rest length.

    Whole-arm attraction: "reached" means the NEAREST node lands on the target
    (any part of the arm can grab), not specifically the tip.
    """
    ctrl = ArmController(n_free=5, rest_length=1.0, horizon=8, max_iters=50)
    base = [0.0, 0.0]
    x0 = ctrl.straight_arm(base, init_angle=0.0)
    target = [2.5, 2.0]  # |target| ~= 3.2 < reach (5)

    res = ctrl.solve(base_xy=base, target=target, x0=x0)
    _, seg = _tip_and_segments(res, base)

    assert res.converged, f"did not converge in {res.iterations} iters"
    err = _min_node_err(res, target)
    # Whole-arm attraction is normalized (cost = w*mean_i|node-target|^2 via the
    # sqrt(n_free) residual scale), which reaches robustly at any node count. The
    # nearest node comes within ~0.3 rest-lengths of the target; in sim units
    # (rest=capture_radius=0.3) that is well inside capture range.
    assert err < 0.30, f"no node reached target {target}; min err {err}"
    # Whole-arm reach DRAPES the arm onto the target (outer segments bunch near
    # it, inner ones stretch to span the gap), so segments strain more than a
    # tip-only reach - but the chain stays connected (no collapse to a point).
    assert seg.min() > 0.4 and seg.max() < 1.7, f"segments drifted: {seg}"


def test_one_controller_serves_multiple_targets():
    """The compiled graph is reused across targets (no per-target retrace)."""
    ctrl = ArmController(n_free=5, rest_length=1.0, horizon=8, max_iters=50)
    base = [0.0, 0.0]
    x0 = ctrl.straight_arm(base, init_angle=0.0)

    for target in ([2.0, 2.0], [-2.0, 1.0], [0.0, -3.0]):
        res = ctrl.solve(base_xy=base, target=target, x0=x0)
        err = _min_node_err(res, target)
        assert err < 0.30, f"target {target}: nearest node err {err}"


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


def test_hold_reach_off_barely_moves():
    """Hold (reach_weight=0): with nothing to chase, effort/spring keep the arm
    still. With whole-arm attraction a pull toward the current tip would instead
    drag every other node inward, so reach must be gated OFF - this checks every
    node stays put, not just the tip.
    """
    ctrl = ArmController(n_free=5, rest_length=1.0, horizon=8, max_iters=30)
    base = [0.0, 0.0]
    x0 = ctrl.straight_arm(base, init_angle=0.0)
    nodes0 = tf.reshape(x0, (-1, 2)).numpy()

    res = ctrl.solve(base_xy=base, target=nodes0[-1].tolist(), x0=x0,
                     reach_weight=0.0)
    nodes = res.x_traj[-1].numpy().reshape(-1, 2)

    assert float(np.linalg.norm(nodes - nodes0, axis=1).max()) < 0.1, \
        "arm moved despite hold (reach off)"


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

    off = ctrl.solve(base_xy=base, target=target, x0=x0)
    assert off.history is None

    # Kernel tracing counts before/after a recorded solve are unchanged.
    kernels = ctrl._solve._kernels
    before = [k.experimental_get_tracing_count() for k in kernels]
    on = ctrl.solve(base_xy=base, target=target, x0=x0, record_history=True)
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
    res = ctrl.solve(base_xy=base, target=[2.5, 2.0], x0=x0,
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
    res = ctrl.solve(base_xy=base, target=[2.5, 2.0], x0=x0,
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
    assert limb.last_ilqr_meta["target_kind"] == "hold"  # no agents
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
