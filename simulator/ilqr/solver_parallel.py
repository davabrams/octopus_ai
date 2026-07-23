"""iLQR solver variant with a GRAPH-COMPILED backward pass (one arm, unchanged).

The stock ``make_solver`` (solver.py) runs the backward Riccati recursion as an
*eager* Python loop over the horizon, so every timestep crosses the Python->TF
boundary for its Jacobians (``bwd.derivs``) and its Riccati step (``bwd.riccati``).
The profiler showed that dispatch overhead is ~73% of the whole sim's wall clock:
~15k tiny ops at ~1ms each, almost all launch overhead, not 6x6 arithmetic.

``make_solver_parallel`` keeps the arm **completely independent** (no batching
across arms - each ``ArmController`` still owns its own solver and could run in
its own process), and instead collapses that per-timestep dispatch by moving the
whole backward pass into a single ``@tf.function``:

- The horizon loop is UNROLLED at trace time, so the whole pass is ONE graph
  execution per solver-iteration instead of ~2*horizon eager dispatches. Inside
  the graph the per-timestep Jacobians (independent across t) are free for TF to
  schedule in parallel, while the Riccati value recursion stays sequential.
- The Cholesky early-break can't be a Python ``break`` inside a graph, so the
  pass computes every timestep and returns an ``ok`` flag (True iff no Cholesky
  produced a NaN). The eager caller checks it ONCE per iteration and, on failure,
  bumps ``mu`` and retries - identical outer-loop semantics to make_solver.

It REUSES make_solver's compiled kernels verbatim (rollout / total_cost /
dyn_jac / quad_run / quad_term / forward via ``solve._kernels``), so the math is
byte-for-byte the same as the eager solver on any solve that doesn't hit a
Cholesky failure - see tests/test_ilqr_parallel.py.
"""
import tensorflow as tf

from simulator.ilqr.solver import (
    ILQRIterationRecord,
    ILQRResult,
    make_solver,
)
from simulator.profiling import span


def make_solver_parallel(dynamics,
                         running_cost,
                         terminal_cost,
                         control_dim: int,
                         max_iters: int = 50,
                         tol: float = 1e-4,
                         mu_init: float = 1e-6,
                         mu_min: float = 1e-8,
                         mu_max: float = 1e3,
                         mu_factor: float = 2.0,
                         alphas=(1.0, 0.5, 0.25, 0.1, 0.05, 0.01)):
    """Same contract as ``make_solver`` (returns ``solve(x0, params, u_init)``),
    but with the backward pass graph-compiled. One solver per arm; nothing is
    shared across arms."""
    base_solve = make_solver(
        dynamics, running_cost, terminal_cost, control_dim,
        max_iters=max_iters, tol=tol, mu_init=mu_init, mu_min=mu_min,
        mu_max=mu_max, mu_factor=mu_factor, alphas=alphas)
    rollout, total_cost, dyn_jac, quad_run, quad_term, forward = base_solve._kernels
    eye_u = tf.eye(control_dim, dtype=tf.float32)

    @tf.function(reduce_retracing=True)
    def backward_pass(x_traj, u_traj, params, mu):
        """One graph execution of the whole Riccati recursion. Returns
        (k_stack, big_k_stack, ok). ``ok`` is False if any Cholesky hit a NaN
        (a non-PD regularized Q_uu) - the caller then raises mu and retries."""
        horizon = int(u_traj.shape[0])  # static at trace time -> the loop unrolls
        v_x, v_xx = quad_term(x_traj[-1], params)
        k_seq = [None] * horizon
        big_k_seq = [None] * horizon
        ok = tf.constant(True)
        for t in reversed(range(horizon)):
            xt, ut = x_traj[t], u_traj[t]
            f_x, f_u = dyn_jac(xt, ut)
            l_x, l_u, l_xx, l_uu, l_ux = quad_run(xt, ut, params)

            q_x = l_x + tf.linalg.matvec(f_x, v_x, transpose_a=True)
            q_u = l_u + tf.linalg.matvec(f_u, v_x, transpose_a=True)
            q_xx = l_xx + tf.matmul(f_x, tf.matmul(v_xx, f_x), transpose_a=True)
            q_uu = l_uu + tf.matmul(f_u, tf.matmul(v_xx, f_u), transpose_a=True)
            q_ux = l_ux + tf.matmul(f_u, tf.matmul(v_xx, f_x), transpose_a=True)

            q_uu_reg = q_uu + mu * eye_u
            chol = tf.linalg.cholesky(q_uu_reg)
            ok = tf.logical_and(
                ok, tf.logical_not(tf.reduce_any(tf.math.is_nan(chol))))

            neg_q_u = -tf.expand_dims(q_u, -1)
            k_t = tf.squeeze(tf.linalg.cholesky_solve(chol, neg_q_u), -1)
            big_k_t = tf.linalg.cholesky_solve(chol, -q_ux)
            k_seq[t] = k_t
            big_k_seq[t] = big_k_t

            v_x = (q_x
                   + tf.linalg.matvec(big_k_t,
                                      tf.linalg.matvec(q_uu, k_t),
                                      transpose_a=True)
                   + tf.linalg.matvec(big_k_t, q_u, transpose_a=True)
                   + tf.linalg.matvec(q_ux, k_t, transpose_a=True))
            v_xx = (q_xx
                    + tf.matmul(big_k_t, tf.matmul(q_uu, big_k_t),
                                transpose_a=True)
                    + tf.matmul(big_k_t, q_ux, transpose_a=True)
                    + tf.matmul(q_ux, big_k_t, transpose_a=True))
            v_xx = 0.5 * (v_xx + tf.transpose(v_xx))
        return tf.stack(k_seq, axis=0), tf.stack(big_k_seq, axis=0), ok

    def solve(x0, params, u_init, record_history: bool = False) -> ILQRResult:
        x0 = tf.convert_to_tensor(x0, dtype=tf.float32)
        params = tf.convert_to_tensor(params, dtype=tf.float32)
        u_traj = tf.convert_to_tensor(u_init, dtype=tf.float32)

        with span("ilqr.rollout"):
            x_traj = rollout(x0, u_traj)
            cost = total_cost(x_traj, u_traj, params)

        mu = mu_init
        converged = False
        iters_run = 0

        history = [] if record_history else None
        if record_history:
            history.append(ILQRIterationRecord(
                iteration=0, phase="init", cost=float(cost), alpha=None,
                mu=mu, rel_improve=None, x_traj=x_traj.numpy()))

        for iteration in range(max_iters):
            iters_run = iteration + 1

            # ---- Backward pass: ONE compiled graph call for the whole horizon.
            with span("ilqr.backward"):
                k_stack, big_k_stack, ok = backward_pass(
                    x_traj, u_traj, params, tf.constant(mu, tf.float32))
                backward_ok = bool(ok)

            if not backward_ok:
                if record_history:
                    history.append(ILQRIterationRecord(
                        iteration=iteration + 1, phase="cholesky_fail",
                        cost=float(cost), alpha=None, mu=mu,
                        rel_improve=None, x_traj=None))
                mu = min(mu * mu_factor, mu_max)
                if mu >= mu_max:
                    break
                continue

            # ---- Forward pass with line search over the step size alpha. ----
            improved = False
            rel_improve = tf.constant(0.0)
            with span("ilqr.forward"):
                for alpha in alphas:
                    x_cand, u_cand, cost_cand = forward(
                        x0, u_traj, k_stack, big_k_stack,
                        tf.constant(alpha, tf.float32), x_traj, params)
                    if bool(cost_cand < cost):
                        rel_improve = (cost - cost_cand) / (tf.abs(cost) + 1e-12)
                        x_traj, u_traj, cost = x_cand, u_cand, cost_cand
                        improved = True
                        break

            if not improved:
                if record_history:
                    history.append(ILQRIterationRecord(
                        iteration=iteration + 1, phase="linesearch_fail",
                        cost=float(cost), alpha=None, mu=mu,
                        rel_improve=None, x_traj=None))
                mu = min(mu * mu_factor, mu_max)
                if mu >= mu_max:
                    break
                continue

            if record_history:
                history.append(ILQRIterationRecord(
                    iteration=iteration + 1, phase="accepted",
                    cost=float(cost), alpha=float(alpha), mu=mu,
                    rel_improve=float(rel_improve), x_traj=x_traj.numpy()))

            mu = max(mu / mu_factor, mu_min)
            if float(rel_improve) < tol:
                converged = True
                break

        return ILQRResult(x_traj=x_traj, u_traj=u_traj, cost=cost,
                          iterations=iters_run, converged=converged,
                          history=history)

    solve._kernels = (*base_solve._kernels, backward_pass)
    return solve
