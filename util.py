from dataclasses import dataclass
import matplotlib.pyplot as plt
from enum import Enum
import numpy as np
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

        Python

import numpy as np

def train_test_split(data, labels, test_size=0.2, random_state=None):
  """
  Splits data and labels into training and test sets.

  Args:
    data: NumPy array of data points.
    labels: NumPy array of corresponding labels.
    test_size: Proportion of data to be used for testing (between 0 and 1).
    random_state: Seed for random shuffling (optional).

  Returns:
    train_data, train_labels, test_data, test_labels: NumPy arrays of training and test data and labels.
  """
  if not isinstance(data, np.ndarray) or not isinstance(labels, np.ndarray):
    raise TypeError("Both data and labels must be NumPy arrays.")

  if test_size < 0 or test_size > 1:
    raise ValueError("test_size must be between 0 and 1.")

  num_samples = data.shape[0]
  shuffle_indices = np.arange(num_samples)
  if random_state is not None:
    # np.random.seed(random_state)
    np.random.shuffle(shuffle_indices)

  split_point = int(num_samples * test_size)

  train_data = data[shuffle_indices[:split_point]]
  train_labels = labels[shuffle_indices[:split_point]]
  test_data = data[shuffle_indices[split_point:]]
  test_labels = labels[shuffle_indices[split_point:]]

  return train_data, train_labels, test_data, test_labels
