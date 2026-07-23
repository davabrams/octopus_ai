"""The graph-compiled backward solver (solver_parallel) must match the eager one.

solver_parallel.make_solver_parallel reuses make_solver's kernels and only moves
the backward Riccati recursion into one @tf.function. So an ArmController with
compiled_backward=True must produce the SAME solve as the default eager one -
same convergence, same iteration count, and the same trajectory to within float32
op-reorder noise. This is the whole safety net for opting the flag on.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.ilqr.arm import ArmController


def _solve(compiled, target, *, n=5, horizon=8, base=(0.0, 0.0), tip_only=True):
    c = ArmController(n_free=n, rest_length=1.0, horizon=horizon, max_iters=50,
                      compiled_backward=compiled)
    x0 = c.straight_arm(list(base), init_angle=0.0)
    at = np.zeros((n, 2), np.float32)
    asw = np.zeros(n, np.float32)
    at[:] = target
    if tip_only:
        asw[-1] = np.sqrt(c.w_reach_terminal)
    else:
        asw[:] = np.sqrt(c.w_reach_terminal)
    rt = np.zeros((n, 2), np.float32)
    rsw = np.zeros(n, np.float32)
    return c.solve(list(base), at, asw, rt, rsw, x0=x0)


def test_compiled_backward_matches_eager_reach():
    """A reach solve: same convergence/iters, trajectory equal to float32 noise."""
    for target in ([2.5, 2.0], [-2.0, 1.0], [0.0, -3.0], [3.5, 0.5]):
        eager = _solve(False, target)
        comp = _solve(True, target)
        assert comp.converged == eager.converged, target
        assert comp.iterations == eager.iterations, (
            f"iter mismatch {comp.iterations} vs {eager.iterations} @ {target}")
        dx = float(np.abs(eager.x_traj.numpy() - comp.x_traj.numpy()).max())
        assert dx < 5e-3, f"trajectory diverged by {dx} @ {target}"
        assert abs(float(eager.cost) - float(comp.cost)) < 1e-2, target


def test_compiled_backward_matches_eager_flee():
    """A flee solve (whole arm retracts to the body) also matches."""
    n = 6
    c_args = dict(n_free=n, rest_length=1.0, horizon=8, max_iters=50)
    outs = {}
    for compiled in (False, True):
        c = ArmController(compiled_backward=compiled, **c_args)
        base = [0.0, 0.0]
        x0 = c.straight_arm(base, init_angle=0.0)
        at = np.zeros((n, 2), np.float32)
        asw = np.zeros(n, np.float32)
        rt = np.zeros((n, 2), np.float32)
        rsw = np.full(n, np.sqrt(c.w_repel), np.float32)
        rt[:] = base  # retract toward the body
        outs[compiled] = c.solve(base, at, asw, rt, rsw, x0=x0)
    e, p = outs[False], outs[True]
    assert p.converged == e.converged
    assert p.iterations == e.iterations
    dx = float(np.abs(e.x_traj.numpy() - p.x_traj.numpy()).max())
    assert dx < 5e-3, f"flee trajectory diverged by {dx}"


def test_one_compiled_controller_serves_multiple_targets():
    """The compiled backward is traced once and reused (no per-target retrace)."""
    n = 5
    c = ArmController(n_free=n, rest_length=1.0, horizon=8, max_iters=50,
                      compiled_backward=True)
    base = [0.0, 0.0]
    x0 = c.straight_arm(base, init_angle=0.0)
    for target in ([2.0, 2.0], [-2.0, 1.0], [0.0, -3.0]):
        at = np.zeros((n, 2), np.float32)
        asw = np.zeros(n, np.float32)
        at[:] = target
        asw[-1] = np.sqrt(c.w_reach_terminal)
        rt = np.zeros((n, 2), np.float32)
        rsw = np.zeros(n, np.float32)
        res = c.solve(base, at, asw, rt, rsw, x0=x0)
        tip = res.x_traj[-1].numpy().reshape(-1, 2)[-1]
        assert np.linalg.norm(tip - np.array(target, float)) < 0.4, target


if __name__ == "__main__":
    import unittest
    unittest.main()
