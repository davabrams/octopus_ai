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


def spring_residual(x: tf.Tensor, base_xy: tf.Tensor, rest: float,
                    sw_spring: float) -> tf.Tensor:
    """Each adjacent pair (incl. base->node0) should sit ``rest`` apart."""
    chain = chain_positions(x, base_xy)
    return sw_spring * (segment_lengths(chain) - rest)


def bending_residual(x: tf.Tensor, base_xy: tf.Tensor,
                     sw_bend: float) -> tf.Tensor:
    """Discrete curvature at each interior node (second difference).

    Zero for a straight chain, so this pulls the arm toward straightness (it can
    still curve toward a target, just not fold back on itself). Includes the base
    so the arm leaves the body smoothly.
    """
    chain = chain_positions(x, base_xy)  # (n_free+1, 2)
    curv = chain[2:] - 2.0 * chain[1:-1] + chain[:-2]  # (n_free-1, 2)
    return sw_bend * tf.reshape(curv, [-1])


def repel_residual(x: tf.Tensor, threat: tf.Tensor, threat_w,
                   r_safe: float) -> tf.Tensor:
    """One-sided barrier: every free node pays ``threat_w*(r_safe - dist)``
    while within ``r_safe`` of the threat, zero beyond it.

    ``threat_w`` is the sqrt-weight, 0 when no threat is in range (so the term
    vanishes without changing the residual's shape — no retrace). Pushes the
    whole arm out of the keep-out zone.
    """
    free = tf.reshape(x, (-1, 2))  # (n_free, 2)
    d = tf.sqrt(tf.reduce_sum(tf.square(free - threat), axis=1) + _EPS)
    return threat_w * tf.nn.relu(r_safe - d)  # (n_free,)


def reach_residual(x: tf.Tensor, target: tf.Tensor, sw_reach) -> tf.Tensor:
    """Pull the tip toward ``target``. ``sw_reach`` is the sqrt-weight (may be a
    scalar tensor so the pull strength can vary per solve without a retrace)."""
    return sw_reach * (tip(x) - target)


def effort_residual(u: tf.Tensor, sw_effort: float) -> tf.Tensor:
    """Penalize control magnitude — smooth, economical motion."""
    return sw_effort * u
