""" Utilities and objects specifically for octopus simulator """
from dataclasses import dataclass
from enum import Enum
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=4)

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
    LOCAL = 0 #local inference
    REMOTE = 0 #remote server inference

@dataclass
class State:
    """ Contains the limb spline nodes' kinematic info
    Stores values as tensors, but things can be accessed as floats
    """
    #x, y  position, and theta
    pos: tf.Variable

    #since time ticks descretely, this is just the previous iteration's delta_x & delta_y
    vel: tf.Variable

    def __init__(self, x: float = 0.0, y: float = 0.0, t: float = 0.0) -> None:
        self.pos = tf.Variable([
            x,
            y,
            t
        ], dtype=tf.float32)
        self.vel = tf.Variable([
            0,
            0,
            0
        ], dtype=tf.float32)

    @property
    def x(self) -> float:
        return float(self.pos[0])
    @property
    def y(self) -> float:
        return float(self.pos[1])
    @property
    def t(self) -> float:
        return float(self.pos[2])
    
    @x.setter
    def x(self, value: float) -> None:
        self.pos[0].assign(value)
    @y.setter
    def y(self, value: float) -> None:
        self.pos[1].assign(value)
    @t.setter
    def t(self, value: float) -> None:
        self.pos[2].assign(value)
    
    def distance_to(self, other: "State") -> float:
        delta = tf.subtract(other.pos, self.pos)
        return np.sqrt(np.reduce_sum(np.square(delta)))

    def move_cartesian(self, delta_x: float, delta_y: float) -> None:
        delta = tf.constant([delta_x, delta_y], dtype=tf.float32)
        self.pos = tf.add(self.pos, delta)
        self.vel = tf.divide(delta, 1.0) #1.0 represents the time increment

        self.t = tf.atan2(delta_y, delta_x)

    def move_polar(self, distance: float, radians: float = 0) -> None:
        self.t += radians
        dx = distance * tf.cos(self.t)
        dy = distance * tf.sin(self.t)
        self.move_cartesian(dx, dy)

    def apply_grad(self, grad: tf.Tensor) -> None:
        self.vel = grad
        self.pos = tf.add(self.pos, grad)


@dataclass
class Agent(State):
    """Data class to store agent properties: state vector and agent type"""
    vel: float = 0
    agent_type: AgentType = None

    def __repr__(self):
        return f"<Agent\n\tType: {self.agent_type}, \n\tLoc: ({self.x}, {self.y}), \n\tVel: {self.vel}\>\n"

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

def display_refresh(ax, octo, ag, surf, debug_mode = False):
    """ Displays all of the octopus simulator components """

    ax.clear()

    # Print the patterned surface
    ax.imshow(surf.grid.astype(float), cmap="binary_r")

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
        if agent.Type == AgentType.PREY:
            color = 'lightgreen'
        if debug_mode:
            ax.plot(agent.x, agent.y, marker='o', color=color, ms=agent_range_ms, alpha=.5)
        ax.plot(agent.x, agent.y, marker='o', color=color)


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
