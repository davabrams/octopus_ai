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
        # Default terminal reach sqrt-weight; solve() can override per call
        # (a gentle weight for exploration vs the strong default for prey).
        self._sw_reach_terminal = sw_reach_terminal

        def dynamics(x, u):
            return x + u * DT

        # Costs are composed from the shared residual library (residuals.py);
        # each term returns a residual vector the solver squares and sums, with
        # its weight folded in as sqrt(w). params =
        # [base_xy, target, threat, threat_w, sw_reach_terminal] (see solve()) -
        # the terminal-reach sqrt-weight rides in params so one compiled
        # controller serves a strong prey reach and a gentle exploration reach
        # without a retrace.
        def running_cost(x, u, params):
            base_xy = params[0:2]
            target = params[2:4]
            threat = params[4:6]
            threat_w = params[6]
            return tf.concat([
                res.effort_residual(u, sw_effort),
                res.spring_residual(x, base_xy, rest, sw_spring),
                res.bending_residual(x, base_xy, sw_bend),
                res.repel_residual(x, threat, threat_w, r_safe),
                res.reach_residual(x, target, sw_reach_run),
            ], axis=0)

        def terminal_cost(x, params):
            base_xy = params[0:2]
            target = params[2:4]
            threat = params[4:6]
            threat_w = params[6]
            sw_reach = params[7]  # per-solve terminal reach sqrt-weight
            return tf.concat([
                res.reach_residual(x, target, sw_reach),
                res.spring_residual(x, base_xy, rest, sw_spring),
                res.bending_residual(x, base_xy, sw_bend),
                res.repel_residual(x, threat, threat_w, r_safe),
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
              u_init: tf.Tensor | None = None,
              record_history: bool = False,
              reach_weight: float | None = None) -> ILQRResult:
        """Plan a trajectory for this arm.

        base_xy: (2,) current body-anchored base position.
        target:  (2,) tip goal (prey / attractor).
        x0:      (2*n_free,) current free-node positions (warm-start friendly).
        threat:  optional (2,) position of the nearest threat to avoid; None
                 disables the repulsion term for this solve.
        u_init:  optional (horizon, 2*n_free) initial controls; zeros if None
                 (pass last frame's shifted solution to warm-start).
        record_history: capture per-iteration solve history on the returned
                 ILQRResult (off = zero overhead). A per-call flag, not a
                 controller field, so one compiled controller serves both modes.
        reach_weight: terminal tip-pull weight for THIS solve (not sqrt'd); None
                 uses the controller's w_reach_terminal. A gentle value drives a
                 soft exploration reach; the strong default drives prey.
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
        sw_reach = (self._sw_reach_terminal if reach_weight is None
                    else float(np.sqrt(max(reach_weight, 0.0))))
        params = tf.concat(
            [base_xy, target, threat_xy, [threat_w], [sw_reach]], axis=0)  # (8,)
        if u_init is None:
            u_init = tf.zeros((self.horizon, 2 * self.n_free), dtype=tf.float32)
        return self._solve(x0, params, u_init,
                           record_history=record_history)
