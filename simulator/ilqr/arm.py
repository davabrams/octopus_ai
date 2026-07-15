"""Single-limb arm model for the iLQR motor-control tier (ARCHITECTURE.md
§11.4).

One limb is one independent controller: it owns its own optimizer and solves
its own control sequence, autonomously of the other arms. ``ArmController``
bundles the arm's differentiable dynamics + costs with a compiled iLQR solver
(built once) so it can be called every frame cheaply, with the per-frame data
(base position, reach target) passed as a **params tensor** so the compiled
graph is reused rather than retraced.

Model
-----
The arm is a chain of nodes. Node 0 is the **base**, pinned to the octopus
body, so it is a per-frame parameter, not a decision variable. The optimizer
moves the ``n_free`` nodes after it.

- **State** ``x`` = the free-node positions, flattened: shape ``(2 * n_free,)``.
- **Control** ``u`` = a velocity per free node: shape ``(2 * n_free,)``.
- **Dynamics** (single integrator): ``x' = x + u * dt``. Linear, so iLQR's
  dynamics linearization is exact and all the interesting curvature is in the
  cost.
- **params** ``p`` = ``[base_x, base_y, target_x, target_y]`` (shape ``(4,)``).

Costs (as squared residuals, for Gauss-Newton — see solver.py)
--------------------------------------------------------------
- **spring**: each adjacent pair (including base→node0) should sit at
  ``rest_length`` apart. Residual = ``sqrt(w_spring) * (dist - rest)``. Applied
  every step so the chain stays coherent throughout the trajectory.
- **effort**: penalize control magnitude, ``sqrt(w_effort) * u`` — smooth,
  economical motion (the "movement cost").
- **attractor**: pull the tip toward the target. A weak pull every step
  (``w_reach_run``) guides the approach; a strong pull at the terminal step
  (``w_reach_terminal``) sets the goal.
"""
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from simulator.ilqr.solver import DT, ILQRResult, make_solver

_EPS = 1e-9  # guards sqrt at zero distance (norm gradient is undefined there)


def _chain_positions(x: tf.Tensor, base_xy: tf.Tensor) -> tf.Tensor:
    """(2*n_free,) free-node vector -> (n_free+1, 2) chain incl. the base."""
    free = tf.reshape(x, (-1, 2))
    return tf.concat([tf.expand_dims(base_xy, 0), free], axis=0)


def _segment_lengths(chain: tf.Tensor) -> tf.Tensor:
    """Euclidean length of each consecutive segment in the chain."""
    deltas = chain[1:] - chain[:-1]
    return tf.sqrt(tf.reduce_sum(tf.square(deltas), axis=1) + _EPS)


@dataclass
class ArmController:
    """Persistent per-limb iLQR controller.

    Construct one per arm; the compiled solver is built in ``__post_init__``
    and reused for every ``solve`` call. Per-frame data (base, target) is
    passed as tensors, so calling ``solve`` each frame does not retrace.
    """
    n_free: int = 7
    rest_length: float = 1.0
    horizon: int = 15
    # Weights balance four wants. Reach dominates at the terminal step so the
    # tip actually gets to the prey (extending or curling as needed); the
    # spring keeps segments near rest but softly enough to yield when reaching a
    # close target; bending kills the high-frequency zigzag that a pure
    # distance-spring chain folds into, while still allowing a smooth curl.
    w_spring: float = 2.0
    w_bend: float = 1.0
    w_effort: float = 0.1
    w_reach_run: float = 0.1
    w_reach_terminal: float = 6.0
    w_repel: float = 8.0        # how strongly a nearby threat is avoided
    repel_radius: float = 2.5   # threat "keep-out" radius: every arm node pays
                                # for being closer than this to a threat
    max_iters: int = 50
    tol: float = 1e-4

    def __post_init__(self):
        n_free = self.n_free
        rest = self.rest_length
        r_safe = float(self.repel_radius)
        sw_spring = float(np.sqrt(self.w_spring))
        sw_bend = float(np.sqrt(self.w_bend))
        sw_effort = float(np.sqrt(self.w_effort))
        sw_reach_run = float(np.sqrt(self.w_reach_run))
        sw_reach_terminal = float(np.sqrt(self.w_reach_terminal))
        self._sw_repel = float(np.sqrt(self.w_repel))

        def dynamics(x, u):
            return x + u * DT

        def _spring_residual(x, base_xy):
            chain = _chain_positions(x, base_xy)
            return sw_spring * (_segment_lengths(chain) - rest)

        def _bending_residual(x, base_xy):
            # Discrete curvature at each interior node: the second difference
            # p_{i-1} - 2 p_i + p_{i+1}. Zero for a straight chain, so this
            # pulls the arm toward straightness (it can still curve toward a
            # target, just not fold back on itself). Includes the base so the
            # arm leaves the body smoothly.
            chain = _chain_positions(x, base_xy)  # (n_free+1, 2)
            curv = chain[2:] - 2.0 * chain[1:-1] + chain[:-2]  # (n_free-1, 2)
            return sw_bend * tf.reshape(curv, [-1])

        def _repel_residual(x, threat, threat_w):
            # One-sided barrier: every free node pays threat_w * (r_safe - dist)
            # while within r_safe of the threat, zero beyond it. threat_w is the
            # sqrt-weight, 0 when no threat is in range (so this term vanishes
            # without changing the residual's shape - no retrace). Pushes the
            # whole arm out of the keep-out zone.
            free = tf.reshape(x, (-1, 2))  # (n_free, 2)
            d = tf.sqrt(tf.reduce_sum(tf.square(free - threat), axis=1) + _EPS)
            return threat_w * tf.nn.relu(r_safe - d)  # (n_free,)

        def _tip(x):
            return tf.reshape(x, (-1, 2))[-1]

        def running_cost(x, u, params):
            base_xy = params[0:2]
            target = params[2:4]
            threat = params[4:6]
            threat_w = params[6]
            return tf.concat([
                sw_effort * u,
                _spring_residual(x, base_xy),
                _bending_residual(x, base_xy),
                _repel_residual(x, threat, threat_w),
                sw_reach_run * (_tip(x) - target),
            ], axis=0)

        def terminal_cost(x, params):
            base_xy = params[0:2]
            target = params[2:4]
            threat = params[4:6]
            threat_w = params[6]
            return tf.concat([
                sw_reach_terminal * (_tip(x) - target),
                _spring_residual(x, base_xy),
                _bending_residual(x, base_xy),
                _repel_residual(x, threat, threat_w),
            ], axis=0)

        self._dynamics = dynamics
        self._solve = make_solver(
            dynamics, running_cost, terminal_cost,
            control_dim=2 * n_free,
            max_iters=self.max_iters, tol=self.tol)

    def straight_arm(self, base_xy, init_angle: float) -> tf.Tensor:
        """Lay out a straight arm from the base along ``init_angle``.

        Returns the flattened free-node state ``x0`` (2*n_free,).
        """
        base_xy = np.asarray(base_xy, dtype=np.float32)
        idx = np.arange(1, self.n_free + 1, dtype=np.float32)
        px = base_xy[0] + idx * self.rest_length * np.cos(init_angle)
        py = base_xy[1] + idx * self.rest_length * np.sin(init_angle)
        return tf.constant(np.stack([px, py], axis=1).reshape(-1),
                           dtype=tf.float32)

    def solve(self,
              base_xy,
              target,
              x0,
              threat=None,
              u_init: tf.Tensor | None = None) -> ILQRResult:
        """Plan a trajectory for this arm.

        base_xy: (2,) current body-anchored base position.
        target:  (2,) tip goal (prey / attractor).
        x0:      (2*n_free,) current free-node positions (warm-start friendly).
        threat:  optional (2,) position of the nearest threat to avoid; None
                 disables the repulsion term for this solve.
        u_init:  optional (horizon, 2*n_free) initial controls; zeros if None
                 (pass last frame's shifted solution to warm-start).
        """
        base_xy = tf.convert_to_tensor(base_xy, dtype=tf.float32)
        target = tf.convert_to_tensor(target, dtype=tf.float32)
        if threat is None:
            # Park the (inert) threat on the base and zero its weight.
            threat_xy = base_xy
            threat_w = 0.0
        else:
            threat_xy = tf.convert_to_tensor(threat, dtype=tf.float32)
            threat_w = self._sw_repel
        params = tf.concat(
            [base_xy, target, threat_xy, [threat_w]], axis=0)  # (7,)
        if u_init is None:
            u_init = tf.zeros((self.horizon, 2 * self.n_free), dtype=tf.float32)
        return self._solve(x0, params, u_init)
