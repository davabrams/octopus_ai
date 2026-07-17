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

from simulator.ilqr import residuals as res
from simulator.ilqr.solver import DT, ILQRResult, make_solver


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
    repel_tip_fraction: float = 0.3  # tip's threat-avoidance strength relative
                                     # to the body-adjacent node's: the body
                                     # matters more than a limb tip, so the base
                                     # end recoils hardest and the tip least
    repel_range: float = 5.0    # per-node flee range = the sense window: a node
                                # flees a threat anywhere within this of it, the
                                # push growing as the threat closes
    max_iters: int = 50
    tol: float = 1e-4

    def __post_init__(self):
        n_free = self.n_free
        rest = self.rest_length
        sw_spring = float(np.sqrt(self.w_spring))
        sw_bend = float(np.sqrt(self.w_bend))
        sw_effort = float(np.sqrt(self.w_effort))
        # Per-node repel grade ramp (the body-adjacent free node avoids threats
        # hardest, the tip least). Stored on self so solve() can fold it into the
        # per-node repel sqrt-weight it packs (only nodes that sense a threat get
        # a nonzero weight; the ramp scales those that do).
        self._repel_grade = np.sqrt(
            np.linspace(1.0, self.repel_tip_fraction, n_free)).astype(np.float32)

        def dynamics(x, u):
            return x + u * DT

        # Costs are composed from the shared residual library (residuals.py).
        # Attraction and repulsion are now PER-NODE (node-autonomous sensing, not
        # a limb policy): params carries, for THIS arm this frame,
        #   [ base_xy(2),
        #     attract_tgt(2n), attract_sw(n),   # each node's sensed prey/explore
        #     repel_tgt(2n),   repel_sw(n),     # each node's sensed threat
        #     repel_range(1) ]                  # the sense window (flee range)
        # sw entries are 0 for nodes that sense nothing, so only nodes within the
        # sense window attract/flee. Slices below index into that fixed layout.
        a0 = 2
        a1 = a0 + 2 * n_free   # end of attract_tgt
        a2 = a1 + n_free       # end of attract_sw
        r1 = a2 + 2 * n_free   # end of repel_tgt
        r2 = r1 + n_free       # end of repel_sw

        def running_cost(x, u, params):
            base_xy = params[0:2]
            return tf.concat([
                res.effort_residual(u, sw_effort),
                res.spring_residual(x, base_xy, rest, sw_spring),
                res.bending_residual(x, base_xy, sw_bend),
                res.repel_residual(x, params[a2:r1], params[r1:r2],
                                   params[r2]),
                res.attract_residual(x, params[a0:a1], params[a1:a2]),
            ], axis=0)

        def terminal_cost(x, params):
            base_xy = params[0:2]
            return tf.concat([
                res.attract_residual(x, params[a0:a1], params[a1:a2]),
                res.spring_residual(x, base_xy, rest, sw_spring),
                res.bending_residual(x, base_xy, sw_bend),
                res.repel_residual(x, params[a2:r1], params[r1:r2],
                                   params[r2]),
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
              attract_tgt,
              attract_sw,
              repel_tgt,
              repel_sw,
              x0,
              u_init: tf.Tensor | None = None,
              record_history: bool = False) -> ILQRResult:
        """Plan a trajectory for this arm from PER-NODE sensing.

        base_xy:     (2,) current body-anchored base position.
        attract_tgt: (n_free, 2) the target each node is drawn to (prey or an
                     explore cell); arbitrary where the node senses nothing.
        attract_sw:  (n_free,) per-node attract sqrt-weight; 0 where the node
                     senses no target (so only sensing nodes attract).
        repel_tgt:   (n_free, 2) the threat each node senses; arbitrary where none.
        repel_sw:    (n_free,) per-node repel sqrt-weight (0 where no threat is in
                     the window). The controller multiplies in its body>tip grade.
        x0:          (2*n_free,) current free-node positions (warm-start friendly).
        u_init:      optional (horizon, 2*n_free) initial controls; zeros if None
                     (pass last frame's shifted solution to warm-start).
        record_history: capture per-iteration solve history (off = zero overhead).
        """
        base_xy = tf.convert_to_tensor(base_xy, dtype=tf.float32)
        attract_tgt = tf.reshape(
            tf.convert_to_tensor(attract_tgt, dtype=tf.float32), [-1])
        attract_sw = tf.convert_to_tensor(attract_sw, dtype=tf.float32)
        repel_tgt = tf.reshape(
            tf.convert_to_tensor(repel_tgt, dtype=tf.float32), [-1])
        # Fold the body>tip grade into the per-node repel weight.
        repel_sw = (tf.convert_to_tensor(repel_sw, dtype=tf.float32)
                    * self._repel_grade)
        params = tf.concat(
            [base_xy, attract_tgt, attract_sw, repel_tgt, repel_sw,
             [float(self.repel_range)]], axis=0)  # (3 + 6*n_free,)
        if u_init is None:
            u_init = tf.zeros((self.horizon, 2 * self.n_free), dtype=tf.float32)
        return self._solve(x0, params, u_init,
                           record_history=record_history)
