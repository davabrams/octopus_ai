"""Generic iterative LQR (iLQR) trajectory optimizer, in TensorFlow.

This is the motor-control tier of the compute hierarchy (ARCHITECTURE.md
§11): a small, sequential, CPU-favorable optimizer. It is deliberately
model-agnostic - it knows nothing about octopus arms. Callers supply three
differentiable TensorFlow callables and this module finds the control
sequence that minimizes the total trajectory cost:

    dynamics(x, u)      -> x_next          (state after one step)
    running_cost(x,u,p) -> residual vector  r,  contributes ||r||^2
    terminal_cost(x, p) -> residual vector rT, contributes ||rT||^2

``p`` is an opaque **params tensor**: the per-solve data that varies frame to
frame (e.g. the arm base position and the reach target). It is threaded
through as a tensor rather than captured in a Python closure precisely so the
compiled graph is *reused* across solves instead of retraced - the difference
between a millisecond solve and a multi-second one (ARCHITECTURE.md §11.6).

Costs are expressed as **residual vectors**, not scalars: the solver squares
and sums them. This lets us use the Gauss-Newton approximation of the cost
Hessian (H ~= 2 Jr^T Jr), symmetric positive-semidefinite by construction -
far more stable for iLQR than a raw (possibly indefinite) Hessian, and needing
only first derivatives of the residuals.

All Jacobians (of the dynamics and of the residuals) are obtained by autodiff,
so any differentiable model "just works"; single-integrator arm dynamics are a
linear special case the autodiff recovers exactly. The heavy per-step
Jacobian/cost kernels are wrapped in ``@tf.function`` and built once by
``make_solver``; call the returned solver repeatedly (one per limb) and the
graph is reused. Both iLQR passes are recurrences along the horizon (Python
loops over T), which is why this belongs on the CPU.
"""
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import tensorflow as tf

from simulator.profiling import span

DT: float = 1.0  # sample time, matches the simulator's dt


class ILQRIterationRecord(NamedTuple):
    """One iteration of a solve, captured only when record_history=True.

    Iteration 0 is the warm-start rollout (the plan the solve refines FROM);
    iterations 1..N are the solver iterations. Failed iterations get a
    status-only entry with x_traj=None so the analyzer scrubber does not show
    unexplained duplicate poses.
    """
    iteration: int            # 0 = initial warm-start rollout; 1..N = solver iters
    phase: str                # 'init' | 'accepted' | 'cholesky_fail' | 'linesearch_fail'
    cost: float               # total cost in effect AFTER this iteration
    alpha: "float | None"     # accepted line-search step; None for init/rejected
    mu: float                 # regularization used in this iteration's backward
                              # pass; the iter-0 'init' entry records mu_init
    rel_improve: "float | None"  # None for init/rejected
    x_traj: "np.ndarray | None"  # float32 (horizon+1, state_dim), UNCLAMPED solver
                              # space; None for rejected iterations


class ILQRResult(NamedTuple):
    """Outcome of a solve."""
    x_traj: tf.Tensor      # (T+1, state_dim) optimized state trajectory
    u_traj: tf.Tensor      # (T, control_dim) optimized control sequence
    cost: tf.Tensor        # scalar final total cost
    iterations: int        # iLQR iterations actually run
    converged: bool        # True if the relative cost improvement fell below tol
    history: "list | None" = None  # list[ILQRIterationRecord] when recording;
                                    # MUST stay the last field (see plan D-note)


def make_solver(dynamics: Callable,
                running_cost: Callable,
                terminal_cost: Callable,
                control_dim: int,
                max_iters: int = 50,
                tol: float = 1e-4,
                mu_init: float = 1e-6,
                mu_min: float = 1e-8,
                mu_max: float = 1e3,
                mu_factor: float = 2.0,
                alphas=(1.0, 0.5, 0.25, 0.1, 0.05, 0.01)) -> Callable:
    """Build a reusable iLQR solver bound to one model.

    Returns ``solve(x0, params, u_init) -> ILQRResult``. The expensive
    TensorFlow kernels are compiled once here and reused across every call, so
    constructing one solver per limb and calling it each frame (with fresh
    ``params``) stays fast. ``params`` is passed straight through to the cost
    callables; ``dynamics`` does not receive it (single-integrator dynamics
    need no per-frame data).
    """
    eye_u = tf.eye(control_dim, dtype=tf.float32)

    # ---- Compiled kernels (traced once, reused for every step/iter/solve) ----
    @tf.function(reduce_retracing=True)
    def rollout(x0, u_traj):
        xs = tf.TensorArray(tf.float32, size=u_traj.shape[0] + 1)
        xs = xs.write(0, x0)
        x = x0
        for t in tf.range(u_traj.shape[0]):
            x = dynamics(x, u_traj[t])
            xs = xs.write(t + 1, x)
        return xs.stack()

    @tf.function(reduce_retracing=True)
    def total_cost(x_traj, u_traj, params):
        total = tf.constant(0.0, dtype=tf.float32)
        for t in tf.range(u_traj.shape[0]):
            r = running_cost(x_traj[t], u_traj[t], params)
            total += tf.reduce_sum(tf.square(r))
        rT = terminal_cost(x_traj[-1], params)
        return total + tf.reduce_sum(tf.square(rT))

    @tf.function(reduce_retracing=True)
    def dyn_jac(x, u):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(u)
            x_next = dynamics(x, u)
        f_x = tape.jacobian(x_next, x)
        f_u = tape.jacobian(x_next, u)
        del tape
        return f_x, f_u

    @tf.function(reduce_retracing=True)
    def quad_run(x, u, params):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(u)
            r = running_cost(x, u, params)
        j_x = tape.jacobian(r, x)
        j_u = tape.jacobian(r, u)
        del tape
        g_x = 2.0 * tf.linalg.matvec(j_x, r, transpose_a=True)
        g_u = 2.0 * tf.linalg.matvec(j_u, r, transpose_a=True)
        h_xx = 2.0 * tf.matmul(j_x, j_x, transpose_a=True)
        h_uu = 2.0 * tf.matmul(j_u, j_u, transpose_a=True)
        h_ux = 2.0 * tf.matmul(j_u, j_x, transpose_a=True)
        return g_x, g_u, h_xx, h_uu, h_ux

    @tf.function(reduce_retracing=True)
    def quad_term(x, params):
        with tf.GradientTape() as tape:
            tape.watch(x)
            r = terminal_cost(x, params)
        j_x = tape.jacobian(r, x)
        g_x = 2.0 * tf.linalg.matvec(j_x, r, transpose_a=True)
        h_xx = 2.0 * tf.matmul(j_x, j_x, transpose_a=True)
        return g_x, h_xx

    @tf.function(reduce_retracing=True)
    def forward(x0, u_traj, k_stack, big_k_stack, alpha, x_ref, params):
        """Line-search rollout applying du = alpha*k + K*(x_new - x_ref)."""
        xs = tf.TensorArray(tf.float32, size=u_traj.shape[0] + 1)
        us = tf.TensorArray(tf.float32, size=u_traj.shape[0])
        xs = xs.write(0, x0)
        x_new = x0
        for t in tf.range(u_traj.shape[0]):
            du = alpha * k_stack[t] + tf.linalg.matvec(
                big_k_stack[t], x_new - x_ref[t])
            u_new = u_traj[t] + du
            us = us.write(t, u_new)
            x_new = dynamics(x_new, u_new)
            xs = xs.write(t + 1, x_new)
        x_cand = xs.stack()
        u_cand = us.stack()
        cost = total_cost(x_cand, u_cand, params)
        return x_cand, u_cand, cost

    def solve(x0, params, u_init, record_history: bool = False) -> ILQRResult:
        x0 = tf.convert_to_tensor(x0, dtype=tf.float32)
        params = tf.convert_to_tensor(params, dtype=tf.float32)
        u_traj = tf.convert_to_tensor(u_init, dtype=tf.float32)
        horizon = int(u_traj.shape[0])

        with span("ilqr.rollout"):
            x_traj = rollout(x0, u_traj)
            cost = total_cost(x_traj, u_traj, params)

        mu = mu_init
        converged = False
        iters_run = 0

        history = [] if record_history else None
        if record_history:
            # Iteration 0: the warm-start rollout the solve refines from.
            # mu records mu_init, the value the FIRST backward pass will use.
            history.append(ILQRIterationRecord(
                iteration=0, phase="init", cost=float(cost), alpha=None,
                mu=mu, rel_improve=None, x_traj=x_traj.numpy()))

        for iteration in range(max_iters):
            iters_run = iteration + 1

            # ---- Backward pass: feedforward k_t and feedback gains K_t ----
            with span("ilqr.backward"):
                v_x, v_xx = quad_term(x_traj[-1], params)
                k_seq = [None] * horizon
                big_k_seq = [None] * horizon
                backward_ok = True

                for t in reversed(range(horizon)):
                    xt, ut = x_traj[t], u_traj[t]
                    # Autodiff'd dynamics + running-cost derivatives (compiled).
                    with span("bwd.derivs"):
                        f_x, f_u = dyn_jac(xt, ut)
                        l_x, l_u, l_xx, l_uu, l_ux = quad_run(xt, ut, params)

                    # Riccati recursion: assemble Q, regularize, Cholesky-solve
                    # for the gains, propagate the value function.
                    with span("bwd.riccati"):
                        q_x = l_x + tf.linalg.matvec(f_x, v_x, transpose_a=True)
                        q_u = l_u + tf.linalg.matvec(f_u, v_x, transpose_a=True)
                        q_xx = l_xx + tf.matmul(f_x, tf.matmul(v_xx, f_x),
                                                transpose_a=True)
                        q_uu = l_uu + tf.matmul(f_u, tf.matmul(v_xx, f_u),
                                                transpose_a=True)
                        q_ux = l_ux + tf.matmul(f_u, tf.matmul(v_xx, f_x),
                                                transpose_a=True)

                        q_uu_reg = q_uu + mu * eye_u
                        chol = tf.linalg.cholesky(q_uu_reg)
                        if bool(tf.reduce_any(tf.math.is_nan(chol))):
                            backward_ok = False
                            break

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

            k_stack = tf.stack(k_seq, axis=0)
            big_k_stack = tf.stack(big_k_seq, axis=0)

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
                # Recorded BEFORE the mu relaxation below, so mu is the value
                # used in this iteration's backward pass. cost/x_traj already
                # hold the accepted candidate.
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

    # The compiled kernels are otherwise unreachable closure-locals; expose
    # them as a plain attribute (no @tf.function signature change) so tests can
    # assert that a recorded solve does not change their tracing counts.
    solve._kernels = (rollout, total_cost, dyn_jac, quad_run, quad_term,
                      forward)

    return solve
