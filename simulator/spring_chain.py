"""Direct linear solver for the SPRING_CHAIN arm model.

Standalone (numpy-only) so the linear algebra can be unit-tested without
the full simulator. The Limb method _move_spring_chain calls solve_chain().

Model (per arm, one frame):
  Nodes 1..n are the free suckers; node 0 is the base, PINNED at the body.
  Forces on each free node, all LINEAR in position:
    - neighbor springs   k_spring * (x_{i-1} - x_i) + k_spring * (x_{i+1} - x_i)
    - agent pull (prey)  k_agent * w_i * (target - x_i)   [spring-to-target]
    - agent push (threat) constant force into f (frozen direction/frame)
    - movement cost      k_move * (x_prev_i - x_i)   [anchor to frame-start pos]
  w_i is a LINEAR tip ramp: w_i = i / (n) so the tip pulls hardest, base least.

Because every force is linear, equilibrium (sum of forces = 0 at every
node) is a single linear system K x = f, solved once per axis. K is
symmetric positive-definite (all stiffnesses positive; threats live in f),
so numpy.linalg.solve is stable. No iteration - that returns only when the
forces become nonlinear.

x and y decouple (isotropic springs share the same K), so we solve two
n-vectors against the same factorization.
"""
import numpy as np


def build_K(n, k_spring, k_move, agent_k_weighted):
    """Assemble the n x n stiffness matrix shared by both axes.

    n: number of free nodes (suckers 1..n; base excluded).
    k_spring: neighbor spring stiffness.
    k_move: uniform movement-cost (anchor) stiffness.
    agent_k_weighted: length-n array of per-node agent spring stiffness
        (k_agent * w_i), already tip-weighted; 0 where a node feels no prey.
    """
    K = np.zeros((n, n), dtype=float)
    for i in range(n):
        # diagonal: sum of every spring pulling on node i
        diag = k_move + agent_k_weighted[i]
        # neighbor to the inner side (node i-1; for i==0 that's the base,
        # which is pinned -> contributes to diag here and to f as a constant)
        diag += k_spring
        # neighbor to the outer side (node i+1) exists for all but the last
        if i < n - 1:
            diag += k_spring
            K[i, i + 1] -= k_spring
        if i > 0:
            K[i, i - 1] -= k_spring
        K[i, i] = diag
    return K


def solve_chain(base_xy, prev_xy, targets, threat_force,
                k_spring, k_agent, k_move):
    """Solve one arm's chain to equilibrium.

    base_xy: (2,) pinned base position (the body).
    prev_xy: (n, 2) each free node's position at the START of this frame.
    targets: (n, 2) prey spring-to-target point per node (only meaningful
        where has_prey; pass prev position where none so the term is inert
        once its stiffness is zeroed).
    threat_force: (n, 2) constant push per node (0 where none).
    k_spring, k_agent, k_move: scalars.
    Also needs, implicitly, which nodes have prey -> encoded by passing a
    per-node agent stiffness of 0 for nodes without prey. Here we take
    targets + a companion mask via NaN: a node with no prey has target NaN.

    Returns (n, 2) new node positions.
    """
    n = prev_xy.shape[0]

    # tip-weighted per-node agent stiffness; nodes without a prey target
    # (NaN) get zero stiffness and a target replaced by their prev pos.
    w = np.array([(i + 1) / n for i in range(n)], dtype=float)  # ramp -> tip
    has_prey = ~np.isnan(targets[:, 0])
    agent_kw = np.where(has_prey, k_agent * w, 0.0)
    safe_targets = np.where(has_prey[:, None], targets, prev_xy)

    K = build_K(n, k_spring, k_move, agent_kw)

    new_xy = np.empty((n, 2), dtype=float)
    for axis in range(2):
        f = np.zeros(n, dtype=float)
        # movement-cost anchor to frame-start position
        f += k_move * prev_xy[:, axis]
        # agent spring-to-target (prey), tip-weighted
        f += agent_kw * safe_targets[:, axis]
        # threat constant push
        f += threat_force[:, axis]
        # pinned base couples into node 0's equation as a constant
        f[0] += k_spring * base_xy[axis]
        new_xy[:, axis] = np.linalg.solve(K, f)

    return new_xy


def base_reaction(base_xy, node1_xy, k_spring):
    """Force the pinned base exerts back on the body: the spring between the
    base and the first free node. Directed base -> node1 when stretched.
    This is what the body feels from this arm (analogous to LUMPED_SPRING's
    tension_vector), keeping body dynamics consistent across modes.
    """
    return k_spring * (np.asarray(node1_xy) - np.asarray(base_xy))
