"""
Defines the motion of an octopus and its arms

There are two stages:  
(1) define octopus and 

Thank you, Jonathan Hui
https://jonathan-hui.medium.com/rl-lqr-ilqr-linear-quadratic-regulator-a5de5104c750
"""
import networkx as nx
import numpy as np
from simulator.simutil import State
from simulator.ilqr.costs import AllCosts
from typing import List
import matplotlib.pyplot as plt

class Node:
    state: State
    neighbors: List
    is_tip: bool
    def __init__(self, state: State, neighbors: List = [], is_tip = False) -> None:
        self.state = state
        self.neighbors = neighbors
        self.is_tip = is_tip

    def loc_as_tuple(self) -> tuple:
        return (self.state.x, self.state.y)

def generate_node_mesh() -> List[Node]:
    # Generate a node mesh, a graph of States
    return [
        Node(state=State(0,0), neighbors=[1, 2]),
        Node(state=State(1,0), neighbors=[3, 4]),
        Node(State(0,1)),
        Node(State(1,1)),
        Node(state=State(2,0), neighbors=[5, 6]),
        Node(State(3,0)),
        Node(State(3,1))
    ]
    # 2 3   6
    # | |  /
    # 0-1-4-5

def generate_node_octopus() -> List[Node]:
    res = [Node(state=State(0,0))]
    ix = 0

    i_max = 4
    j_max = 4

    for i in range(i_max):
        res[0].neighbors.append(ix + 1)
        for j in range(j_max):
            is_tip = False
            ix += 1
            if j == 0:
                neighbors = [0]
            else:
                neighbors = [ix - 1]

            if j < j_max - 1:
                neighbors += [ix + 1]
            else:
                is_tip = True
            res.append(Node(state=State(ix/20 + i / 8, j / 8), neighbors=neighbors, is_tip=is_tip))
    return res

plt.ion()
# node_array = generate_node_mesh()
node_array = generate_node_octopus()
for ix, n in enumerate(node_array):
    print(f"{ix}: {n.neighbors}")
G = nx.Graph()
node_pairs = [[(ix, n) for n in node.neighbors] for ix, node in enumerate(node_array)]
node_pairs = [item for sublist in node_pairs for item in sublist]
print(node_pairs)
G.add_edges_from(node_pairs)

x, y = None, None
for val in np.arange(0, 30, 0.1):

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

    cost_array = [
        AllCosts(
            origin_state=node.state,
            all_nodes=[n.state for n in node_array if n is not node],
            attractor_states=attractors if n.is_tip else None,
            neighbor_states=[node_array[n].state for n in node.neighbors],
            min_distance = 0.3)
            for node in node_array
        ]
    list(map(lambda c: c.compute(), cost_array))\

    for n, c in zip(node_array, cost_array):
        n.state.apply_grad(c.get_result().grad)

    plt.axis([-2, 2, -2, 2])
    
    pos = {ix: n.loc_as_tuple() for ix, n in enumerate(node_array)}
    nx.draw_networkx(G, pos=pos, node_size=10, with_labels=False)
    # for node in node_array:
    #     plt.scatter(x = node.state.pos[0], y = node.state.pos[1])
    for attractor in attractors:
        plt.scatter(x = attractor.pos[0], y = attractor.pos[1], c="Red")

    plt.draw()
