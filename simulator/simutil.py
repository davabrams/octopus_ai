import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from enum import Enum

np.set_printoptions(precision=4)

""" utilities for octopus simulator """

class MovementMode(Enum):
    RANDOM: int = 0
    ATTRACT_REPEL: int = 1


class AgentType(Enum):
    PREY = 0
    THREAT = 1

class MLMode(Enum):
    NO_MODEL = 0
    SUCKER = 1
    LIMB = 2
    FULL = 3

@dataclass
class Agent:
    x: float = 0
    y: float = 0
    vel: float = 0
    t: float = 0
    Type: AgentType = None
    
    def __repr__(self):
        return f"<Agent\n\tType: {self.Type}, \n\tLoc: ({self.x}, {self.y}), \n\tVel: {self.vel}, \n\tTheta = {self.t}>\n"


@dataclass
class Color:
    r: float = 0.5
    g: float = 0.5
    b: float = 0.5
    
    def to_rgb(self):
        return [self.r, self.g, self.b]
    

def print_setup():
    fig = plt.figure()
    fig.show()
    ax = fig.add_subplot(1,1,1)
    ax.set_xticks
    ax.set_xticks([]) 
    ax.set_xticks([]) 
    ax.set_title("Octopus AI Visualizer") 
    return fig, ax

def print_all(ax, octo, ag, surf, debug_mode = False):

    ax.clear()

    # Print the patterned surface
    ax.imshow(surf.grid.astype(float), cmap="binary_r") 

    #delete this.  used to check for transposed behavior
    if debug_mode == True:
        for i_x, row in enumerate(surf.grid):
            for i_y, val in enumerate(row):
                ax.plot(i_x, i_y, marker='.', mfc=[float(val)] * 3, markeredgewidth = 0)

    # Print the octopus
    sucker_edge_width = 0
    if debug_mode:
        sucker_edge_width = 1
    for limb in octo.limbs:
        for sucker in limb.suckers:
            ax.plot(sucker.x, sucker.y, marker='.', markersize = 10, mfc=sucker.c.to_rgb(), mec=[0.5, 0.5, 0.5], markeredgewidth=sucker_edge_width)
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
