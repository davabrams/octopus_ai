""" Utilities and objects specifically for octopus simulator """
from dataclasses import dataclass
from enum import Enum
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


@dataclass
class Agent:
    """Data class to store agent properties: state vector and agent type"""
    x: float = 0
    y: float = 0
    vel: float = 0
    t: float = 0
    agent_type: AgentType = None
    
    def __repr__(self):
        return f"<Agent\n\tType: {self.agent_type}, \n\tLoc: ({self.x}, {self.y}), \n\tVel: {self.vel}, \n\tTheta = {self.t}>\n"


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
    fig.show()
    ax = fig.add_subplot(1,1,1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Octopus Visualizer")
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
