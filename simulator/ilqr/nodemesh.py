"""
Defines the motion of an octopus and its arms

There are two stages:  
(1) define octopus and 

Thank you, Jonathan Hui
https://jonathan-hui.medium.com/rl-lqr-ilqr-linear-quadratic-regulator-a5de5104c750
"""
import math
import networkx as nx
import numpy as np
import tensorflow as tf
from simulator.simutil import State
from simulator.ilqr.costs import AllCosts
from typing import List, Tuple
import matplotlib.pyplot as plt

tf.config.run_functions_eagerly(True)


class Node:
    state: State
    neighbors: List
    is_tip: bool
    def __init__(self, state: State, neighbors: List = [],
                 is_tip=False) -> None:
        self.state = state
        self.neighbors = neighbors
        self.is_tip = is_tip

    def loc_as_tuple(self) -> tuple:
        return (self.state.x, self.state.y)

def generate_node_mesh() -> List[Node]:
    # Generate a node mesh, a graph of States
    return [
        Node(state=State(0, 0), neighbors=[1, 2]),
        Node(state=State(1, 0), neighbors=[3, 4]),
        Node(State(0, 1)),
        Node(State(1, 1)),
        Node(state=State(2, 0), neighbors=[5, 6]),
        Node(State(3, 0)),
        Node(State(3, 1))
    ]
    # 2 3   6
    # | |  /
    # 0-1-4-5

def generate_node_octopus() -> List[Node]:
    n_limbs = 4
    n_suckers = 4

    def rad_to_xy(radial_index: int,
                  distance: float) -> Tuple[float, float]:
        x = distance * math.cos(radial_index * 2 * np.pi / n_limbs)
        y = distance * math.sin(radial_index * 2 * np.pi / n_limbs)
        return x, y
    
    res = [Node(state=State(0, 0))]
    ix = 0

    for i in range(n_limbs):
        res[0].neighbors.append(ix + 1)
        for j in range(n_suckers):
            is_tip = False
            ix += 1
            if j == 0:
                neighbors = [0]
            else:
                neighbors = [ix - 1]

            if j < n_suckers - 1:
                neighbors += [ix + 1]
            else:
                is_tip = True
            x, y = rad_to_xy(i, j * 0.1)
            node = Node(
                state=State(x, y),
                neighbors=neighbors,
                is_tip=is_tip
                )
            res.append(node)
    return res

plt.ion()
node_array = generate_node_octopus()
for ix, n in enumerate(node_array):
    print(f"{ix}: {n.neighbors}")
G = nx.Graph()
node_pairs = [[(ix, n) for n in node.neighbors]
              for ix, node in enumerate(node_array)]
node_pairs = [item for sublist in node_pairs for item in sublist]
print(node_pairs)
G.add_edges_from(node_pairs)

x, y = None, None
for val in np.arange(0, 30.0, 0.1):

    plt.pause(0.1)
    plt.clf()

    def mouse_move(event):
        global x
        x = event.xdata
        global y
        y = event.ydata

    plt.connect('motion_notify_event', mouse_move)

    attractors = [
        State(1 * np.sin(val), 1 * np.cos(val))
    ]
    if x and y:
        attractors += [State(x, y)]

    tf.function()
    def gen_cost_array(attractor_array: List):
        return [
            AllCosts(
                origin_state=node.state,
                all_nodes=[n.state for n in node_array if n is not node],
                attractor_states=attractor_array if node.is_tip else None,
                neighbor_states=[node_array[n].state for n in node.neighbors],
                max_distance=0.3,
                min_distance=0.1,
                alphas=[0.2, 0.4, 0.6, 0.8, 1.0])
                for node in node_array
            ]
    cost_array = gen_cost_array(attractors)
    
    @tf.function()
    def compute_costs():
        list(map(lambda c: c.compute(), cost_array))
    compute_costs()

    def line_search():
        gradient_list = []
        alpha_list = []
        for c in cost_array:
            a, g, c = c.line_search()
            gradient_list.append(g)
            alpha_list.append(a)
        return gradient_list
    
    gradient_array_with_alpha = line_search()

    # Apply gradients with alpha selected
    for n, g in zip(node_array, gradient_array_with_alpha):
        n.state.apply_grad(g)


    # Apply gradients (a tensorflow operation)
    # for n, c in zip(node_array, cost_array):
    #     n.state.apply_grad(c.get_result().grad)

    plt_size = 2
    plt.axis([-plt_size, plt_size, -plt_size, plt_size])
    
    pos = {ix: n.loc_as_tuple() for ix, n in enumerate(node_array)}
    nx.draw_networkx(G, pos=pos, node_size=10, with_labels=False)
    for attractor in attractors:
        plt.scatter(x=attractor.pos[0], y=attractor.pos[1], c="Red")

    plt.draw()
