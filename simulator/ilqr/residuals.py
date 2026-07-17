"""Differentiable residual library for the iLQR arm cost (Gauss-Newton).

Home for every cost term that drives ``MovementMode.ILQR``. These were closures
inside ``ArmController.__post_init__``; hoisting them here gives the cost library
one obvious place to live, makes each term unit-testable in isolation, and is the
place new terms (e.g. an exploration reach) go.

CONTRACT for every residual here
--------------------------------
- It returns a **residual vector** (``tf.Tensor``). The solver (``solver.py``)
  forms the cost as ``||r||^2`` and gets curvature from the Gauss-Newton
  approximation ``2 Jr^T Jr`` via autodiff — so residuals must be
  **TF-differentiable** and **shape-stable**: no Python branching on tensor
  *values*, and one-sided barriers use shape-preserving ops (``tf.nn.relu``)
  rather than ``if``, so the compiled graph never retraces.
- Its **weight enters as ``sqrt(w)``**: ``||sqrt(w)*r||^2 == w*||r||^2``. Callers
  precompute the sqrt-weight once and pass it in.

This is the modern residual/Gauss-Newton paradigm — deliberately distinct from
the retired ``CostTemplate`` gradient-relaxation prototype.
"""
import tensorflow as tf

_EPS = 1e-9  # guards sqrt at zero distance (norm gradient is undefined there)


def chain_positions(x: tf.Tensor, base_xy: tf.Tensor) -> tf.Tensor:
    """(2*n_free,) free-node vector -> (n_free+1, 2) chain incl. the base."""
    free = tf.reshape(x, (-1, 2))
    return tf.concat([tf.expand_dims(base_xy, 0), free], axis=0)


def segment_lengths(chain: tf.Tensor) -> tf.Tensor:
    """Euclidean length of each consecutive segment in the chain."""
    deltas = chain[1:] - chain[:-1]
    return tf.sqrt(tf.reduce_sum(tf.square(deltas), axis=1) + _EPS)


def tip(x: tf.Tensor) -> tf.Tensor:
    """The last free node (the arm tip), (2,)."""
    return tf.reshape(x, (-1, 2))[-1]


def _deadband(v: tf.Tensor, slack: float) -> tf.Tensor:
    """Zero within ``[-slack, slack]``, the (sign-preserved) excess beyond it.

    ``relu(v-slack) - relu(-v-slack)``: a free zone where the term costs nothing
    (the node moves paying only effort) and only the overrun is penalized.
    """
    if slack <= 0.0:
        return v
    return tf.nn.relu(v - slack) - tf.nn.relu(-v - slack)


def spring_residual(x: tf.Tensor, base_xy: tf.Tensor, rest: float,
                    sw_spring: float, slack: float = 0.0) -> tf.Tensor:
    """Each adjacent pair (incl. base->node0) should sit ``rest`` apart.

    Linear residual -> quadratic cost -> LINEAR restoring force: good near rest
    (well-conditioned, corrects small errors) but a strong attractor can still
    stretch the chain far. Pair with ``spring_stiffen_residual`` to cap that.
    ``slack`` gives a deadband: the segment length may deviate +-slack from rest
    for free, and only the overrun is sprung.
    """
    chain = chain_positions(x, base_xy)
    return sw_spring * _deadband(segment_lengths(chain) - rest, slack)


def spring_stiffen_residual(x: tf.Tensor, base_xy: tf.Tensor, rest: float,
                            sw: float, slack: float = 0.0) -> tf.Tensor:
    """Super-linear spring: residual ``sw*dev*|dev|`` per segment (dev=len-rest),
    so cost ~ dev^4 and the restoring FORCE ~ dev^3 (cubic).

    The residual is ``dev^2`` (not ``dev``) SPECIFICALLY because the solver is
    Gauss-Newton least-squares: it forms the cost by squaring the residual
    (``||r||^2``), so a quartic cost has to be encoded as a quadratic residual.
    We can't keep ``r = sw*dev`` and ask for a ^4 cost - the solver's exponent is
    fixed at 2 (that's what gives it the cheap ``2*J^T J`` Hessian).

    Negligible near rest, but a fast-rising soft wall against large stretch OR
    compression - it clamps the extreme deformation a strong attractor/repel
    would otherwise cause (arm balling up at the tips or stretching near the
    body). Sign-preserving, so it pushes compressed segments out and pulls
    stretched ones in, like the linear spring. ``slack`` shares the linear
    spring's deadband, so the wall only starts beyond the free zone.
    """
    chain = chain_positions(x, base_xy)
    dev = _deadband(segment_lengths(chain) - rest, slack)
    return sw * dev * tf.abs(dev)


def bending_residual(x: tf.Tensor, base_xy: tf.Tensor,
                     sw_bend: float, deadzone: float = 0.0) -> tf.Tensor:
    """Discrete curvature at each interior node (second difference).

    Zero for a straight chain, so this pulls the arm toward straightness (it can
    still curve toward a target, just not fold back on itself). Includes the base
    so the arm leaves the body smoothly. ``deadzone`` is a free-bend tolerance on
    the curvature-vector MAGNITUDE (= 2*rest*sin(angle/2) for a bend `angle`): a
    node may bend that much for nothing, only the excess is penalized.
    """
    chain = chain_positions(x, base_xy)  # (n_free+1, 2)
    curv = chain[2:] - 2.0 * chain[1:-1] + chain[:-2]  # (n_free-1, 2)
    if deadzone > 0.0:
        # Deadband on the curvature magnitude: keep only the excess beyond the
        # free bend, in the same direction (so a bend within tolerance is free).
        mag = tf.sqrt(tf.reduce_sum(tf.square(curv), axis=1, keepdims=True)
                      + _EPS)
        curv = curv * (tf.nn.relu(mag - deadzone) / mag)
    return sw_bend * tf.reshape(curv, [-1])


def repel_residual(x: tf.Tensor, targets: tf.Tensor, node_sw) -> tf.Tensor:
    """Per-node flee: each free node RETRACTS toward the body ("scrunch up").

    A fleeing octopus arm does not extend away from the threat - it pulls IN
    toward the body. So the flee force points node -> body, NOT node -> away-
    from-threat. ``targets`` is ``(n_free, 2)`` = the body centre (same for every
    node); ``node_sw`` is the ``(n_free,)`` per-node sqrt-weight, 0 where the node
    senses no threat and larger the closer its sensed threat is (so the arm pulls
    in harder the nearer the danger). Same form as attract_residual - flee is
    just an attract whose target is the body and whose weight is threat
    proximity.
    """
    free = tf.reshape(x, (-1, 2))       # (n_free, 2)
    tgt = tf.reshape(targets, (-1, 2))  # (n_free, 2)
    return tf.reshape(node_sw[:, None] * (free - tgt), [-1])  # (2*n_free,)


def attract_residual(x: tf.Tensor, targets: tf.Tensor, node_sw) -> tf.Tensor:
    """Per-node attract: each free node is pulled toward ITS OWN sensed target.

    Node-autonomous, NOT a limb policy: ``targets`` is ``(n_free, 2)`` - the prey
    (strong) or explore cell (gentle) each node senses (arbitrary where it senses
    nothing) - and ``node_sw`` is the ``(n_free,)`` per-node sqrt-weight, 0 for
    nodes that sense no target (so only nodes that sense prey attract to it). Each
    node pulls to its own target at full per-node weight (no whole-arm
    normalization): nodes converging on one prey grab it; a node the prey never
    reached simply doesn't pull.
    """
    free = tf.reshape(x, (-1, 2))       # (n_free, 2)
    tgt = tf.reshape(targets, (-1, 2))  # (n_free, 2)
    return tf.reshape(node_sw[:, None] * (free - tgt), [-1])  # (2*n_free,)


def effort_residual(u: tf.Tensor, sw_effort: float) -> tf.Tensor:
    """Penalize control magnitude — smooth, economical motion (LINEAR force)."""
    return sw_effort * u


def effort_stiffen_residual(u: tf.Tensor, sw: float) -> tf.Tensor:
    """Super-linear control (per-node VELOCITY) penalty: residual ``sw*|u_i|*u_i``
    per node, so cost ~ ``sum_i |u_i|^4`` and the force ~ ``|u_i|^3``.

    Negligible for a normal per-step move, a fast-rising soft wall against a
    large one - forbids a node "teleporting" across the field in a single frame
    (the applied step is the first control) while leaving ordinary motion free.
    Same trick as spring_stiffen: the GN solver squares the residual, so a
    quartic cost is encoded as a quadratic (``|u_i|*u_i``) residual.
    """
    uu = tf.reshape(u, (-1, 2))  # (n_free, 2)
    mag = tf.sqrt(tf.reduce_sum(tf.square(uu), axis=1, keepdims=True) + _EPS)
    return sw * tf.reshape(uu * mag, [-1])  # (2*n_free,), each node x its speed
