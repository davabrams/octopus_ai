""" Utilities and objects specifically for octopus simulator """
from dataclasses import dataclass
from enum import Enum
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from abc import ABC
np.set_printoptions(precision=4)

 #1.0 represents the sample time
dt: float = 1.0

class MovementMode(Enum):
    """Enum stores agent, octopus, and limb movement types"""
    RANDOM: int = 0
    ATTRACT_REPEL: int = 1


class AgentType(Enum):
    """Enum stores the two types of agents"""
    PREY = 0
    THREAT = 1


class MLMode(Enum):
    """Enum, stores the possible ML modes including no ML at all (heuristic)"""
    NO_MODEL = 0
    SUCKER = 1
    LIMB = 2
    FULL = 3

class InferenceLocation(Enum):
    """Configuration for local v remote inference"""
    LOCAL = 0  # local inference
    REMOTE = 1  # remote server inference

class KinematicPrimitive(ABC):
    # A TensorFlow tensor of shape (6,) representing [x, y, θ, vx, vy, ω]
    # (annotation only: instances create their own tf.Variable in __init__,
    # and creating one at class definition time forces TF init on import)
    _dims: tf.Variable

class State(KinematicPrimitive):
    """ Contains the limb spline nodes' kinematic info
    Stores values as tensors, but things can be accessed as floats
    """

    def __init__(self,
                 x: float = 0.0,
                 y: float = 0.0,
                 t: float = 0.0,
                 vx: float = 0.0,
                 vy: float = 0.0,
                 w: float = 0.0
                 ) -> None:
        self._dims = tf.Variable([
            x,
            y,
            t,
            vx,
            vy,
            w
        ], dtype=tf.float32)

    @property
    def pos(self) -> tf.Variable:
        #x, y  position
        return self._dims[0:2]
    @property
    def vel(self) -> tf.Variable:
        #x, y  velocity
        return self._dims[3:5]

    @property
    def x(self) -> float:
        return float(self._dims[0])
    @property
    def y(self) -> float:
        return float(self._dims[1])
    @property
    def t(self) -> float:
        return float(self._dims[2])
    @property
    def vx(self) -> float:
        return float(self._dims[3])
    @property
    def vy(self) -> float:
        return float(self._dims[4])
    @property
    def w(self) -> float:
        return float(self._dims[5])

    @pos.setter
    def pos(self, value: tf.Variable) -> None:
        assert value.shape == (2,)
        self._dims[0].assign(value[0])
        self._dims[1].assign(value[1])

    @vel.setter
    def vel(self, value: tf.Variable) -> None:
        assert value.shape == (2,)
        self._dims[3].assign(value[0])
        self._dims[4].assign(value[1])

    @x.setter
    def x(self, value: float) -> None:
        self._dims[0].assign(value)
    @y.setter
    def y(self, value: float) -> None:
        self._dims[1].assign(value)
    @t.setter
    def t(self, value: float) -> None:
        self._dims[2].assign(value)
    @vx.setter
    def vx(self, value: float) -> None:
        self._dims[3].assign(value)
    @vy.setter
    def vy(self, value: float) -> None:
        self._dims[4].assign(value)
    @w.setter
    def w(self, value: float) -> None:
        self._dims[5].assign(value)

    def update_kinematics(self) -> None:
        # Update position and angle based on current velocities
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.t = (self.t + self.w * dt) % (2 * np.pi)
    
    def distance_to(self, other: "State") -> float:
        delta = tf.subtract(other.pos, self.pos)
        return tf.sqrt(tf.reduce_sum(tf.square(delta)))

    # def move_cartesian(self, delta_x: float, delta_y: float) -> None:
    #     # (1) move dx/dy points
    #     # (2) change the angle of motion to match the angle of dx, dy
    #     # (3) change the vx/vy to match dx/dy
    #     delta_xy = tf.constant([delta_x, delta_y], dtype=tf.float32)
    #     self.pos = tf.add(self.pos, delta_xy)
    #     self.t = tf.atan2(delta_y, delta_x)
    #     self.vel = tf.divide(delta_xy, dt)

    # def move_polar(self, distance: float, w: float = 0) -> None:
    #     # (1) apply angular momentum change
    #     # (2) move distance units at an angle of theta
    #     # (3) update the velocity to match this new motion
    #     self.t += w / dt
    #     dx = distance * tf.cos(self.t)
    #     dy = distance * tf.sin(self.t)
    #     self.x += dx
    #     self.y += dy
    #     self.vx = dx / dt
    #     self.vy = dy / dt

    def apply_grad(self, grad: tf.Tensor) -> None:
        self.vel = grad
        self.pos = tf.add(self.pos, grad)



class Agent(State):
    agent_type: AgentType = None
    """Data class to store agent properties: state vector and agent type"""
    def __init__(self, x=0.0, y=0.0, t=0.0, vx=0.0, vy=0.0, vel_t=0.0, agent_type=None):
        super().__init__(x, y, t, vx, vy, vel_t)
        self.agent_type = agent_type

    def __repr__(self):
        return f"<Agent\n\tType: {self.agent_type}, \n\tLoc: ({self.x}, {self.y}), \n\tVel: {self.vel}>\n"

class CenterPoint(State):
    """ Contains the limb spline nodes' kinematic info"""


@dataclass
class Color:
    """Data class to store color properties, r g b"""
    r: float = 0.5
    g: float = 0.5
    b: float = 0.5

    def to_rgb(self):
        """Getter function for vectorized values"""
        return np.array([self.r, self.g, self.b])


def setup_display():
    """Sets up the visualization figure"""
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Octopus Visualizer")
    fig.show()
    return fig, ax

def display_refresh(ax, octo, ag, surf, debug_mode = False, show_forces = False):
    """ Displays all of the octopus simulator components """
    from matplotlib.lines import Line2D

    ax.clear()

    # Print the patterned surface
    ax.imshow(surf.grid.astype(float), cmap="binary_r", vmin=0.0, vmax=1.0)

    #used to check for transposed behavior
    if debug_mode:
        for i_x, row in enumerate(surf.grid):
            for i_y, val in enumerate(row):
                ax.plot(i_x, i_y, marker='.', mfc=[float(val)] * 3, markeredgewidth = 0)

    # Print the octopus
    sucker_edge_width = 0
    if debug_mode:
        sucker_edge_width = 1
    for limb in octo.limbs:
        for sucker in limb.suckers:
            # TODO(davabrams) : we want this to use the update() method to save time
            ax.plot(sucker.x,
                    sucker.y,
                    marker = '.',
                    markersize = 10,
                    mfc = sucker.c.to_rgb(),
                    mec = [0.5, 0.5, 0.5],
                    markeredgewidth = sucker_edge_width)
        if debug_mode:
            for c_row in range(len(limb.center_line) - 1):
                pt_1 = limb.center_line[c_row]
                pt_2 = limb.center_line[c_row + 1]
                ax.plot([pt_1.x, pt_2.x], [pt_1.y, pt_2.y], color = 'brown')

    # Print the agents
    for agent in ag.agents:
        agent_range_ms = np.pi * np.power(ag.range_radius, 2)
        color: str = 'violet'
        if agent.agent_type == AgentType.PREY:
            color = 'lightgreen'
        if debug_mode:
            ax.plot(agent.x, agent.y, marker='o', color=color, ms=agent_range_ms, alpha=.5)
        ax.plot(agent.x, agent.y, marker='o', color=color)

    # Force vectors (opt-in via show_forces). Arrows are scaled up for
    # visibility since raw force magnitudes are small (< 1 grid unit).
    if show_forces:
        FORCE_SCALE = 3.0
        # Per-arm tension (what the arm feeds the body): cyan.
        for limb in octo.limbs:
            base = limb.center_line[0]
            ten = getattr(limb, 'last_tension', None)
            if ten is not None and (abs(ten[0]) > 1e-6 or abs(ten[1]) > 1e-6):
                ax.arrow(base.x, base.y,
                         ten[0] * FORCE_SCALE, ten[1] * FORCE_SCALE,
                         head_width=0.25, head_length=0.3, fc='cyan',
                         ec='cyan', length_includes_head=True, alpha=0.9,
                         zorder=5)
            # Tip attraction/repel: orange (prey pull) at the arm tip.
            tip = limb.center_line[-1]
            fa = getattr(limb, 'last_f_attract', None)
            if fa is not None and (abs(fa[0]) > 1e-6 or abs(fa[1]) > 1e-6):
                ax.arrow(tip.x, tip.y,
                         fa[0] * FORCE_SCALE, fa[1] * FORCE_SCALE,
                         head_width=0.2, head_length=0.25, fc='orange',
                         ec='orange', length_includes_head=True, alpha=0.8,
                         zorder=5)
        # Body drift (summed tension, capped): yellow, from the body center.
        bf = getattr(octo, 'last_body_force', None)
        if bf is not None and (abs(bf[0]) > 1e-6 or abs(bf[1]) > 1e-6):
            ax.arrow(octo.x, octo.y,
                     bf[0] * FORCE_SCALE, bf[1] * FORCE_SCALE,
                     head_width=0.35, head_length=0.4, fc='yellow',
                     ec='yellow', length_includes_head=True, zorder=6)

    # Legend explaining the on-screen elements.
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label='Prey (arms reach toward)',
               markerfacecolor='lightgreen', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Threat (arms recoil, body flees)',
               markerfacecolor='violet', markersize=8),
        Line2D([0], [0], marker='.', color='w', label='Suckers (shaded to camouflage)',
               markerfacecolor='gray', markeredgecolor=[0.5, 0.5, 0.5],
               markersize=12),
    ]
    if show_forces:
        legend_handles += [
            Line2D([0], [0], color='yellow', lw=2, label='Body drift force'),
            Line2D([0], [0], color='cyan', lw=2, label='Arm tension (to body)'),
            Line2D([0], [0], color='orange', lw=2, label='Tip attraction/repel'),
        ]
    ax.legend(handles=legend_handles, loc='upper left',
              fontsize='x-small', framealpha=0.8,
              bbox_to_anchor=(0.0, 1.0))


def agent_influence_vector(x: float,
                           y: float,
                           agents: list,
                           radius: float):
    """Sum of attract/repel influence from agents within `radius` of (x, y).

    Returns an (dx, dy) numpy vector pointing toward nearby PREY and away
    from nearby THREATs, with each agent weighted by how close it is
    (linear falloff: weight 1 at distance 0, weight 0 at the radius edge).
    Agents outside the radius contribute nothing. With no agents in range
    the result is the zero vector, which callers treat as "relax to the
    neutral pose".

    This is the shared primitive behind octopus body drift and limb
    reaching in MovementMode.ATTRACT_REPEL.
    """
    influence = np.zeros(2, dtype=float)
    if not agents or radius <= 0:
        return influence
    for agent in agents:
        dx = agent.x - x
        dy = agent.y - y
        dist = np.sqrt(dx * dx + dy * dy)
        if dist > radius or dist < 1e-9:
            continue
        weight = 1.0 - (dist / radius)  # linear falloff in [0, 1]
        ux, uy = dx / dist, dy / dist   # unit vector toward the agent
        if agent.agent_type == AgentType.PREY:
            influence[0] += weight * ux
            influence[1] += weight * uy
        else:  # THREAT
            influence[0] -= weight * ux
            influence[1] -= weight * uy
    return influence


def convert_adjacents_to_ragged_tensor(adjacents: list):
    """
    Converts a list of adjacent suckers to a ragged tensor for model ingestion
    """
    c_array = []
    dist_array = []
    for adj in adjacents:
        s = adj[0]
        c = s.c.r
        dist = adj[1]
        c_array.append(c)
        dist_array.append(dist)
    ragged = tf.ragged.constant([c_array, dist_array])
    return ragged
