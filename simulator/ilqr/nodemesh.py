"""
Defines the motion of an octopus and its arms

There are two stages:  
(1) define octopus and 

Thank you, Jonathan Hui
https://jonathan-hui.medium.com/rl-lqr-ilqr-linear-quadratic-regulator-a5de5104c750
"""
from dataclasses import dataclass
import networkx as nx
import numpy as np
from simulator.simutil import State
from simulator.ilqr.costs import AllCosts
from typing import List
import matplotlib.pyplot as plt

class Node:
    state: State
    neighbors: List
    def __init__(self, state: State, neighbors: List = []) -> None:
        self.state = state
        self.neighbors = neighbors

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

plt.ion()
node_array = generate_node_mesh()
G = nx.Graph()
node_pairs = [[(ix, n) for n in node.neighbors] for ix, node in enumerate(node_array)]
node_pairs = [item for sublist in node_pairs for item in sublist]
G.add_edges_from(node_pairs)

for val in np.arange(0, 30, 0.1):
    attractors = [
        State(
            1 * np.sin(val),
            1 * np.cos(val)
            )
        ]
    
    cost_array = [
        AllCosts(
            origin_state=node.state,
            attractor_states=attractors,
            neighbor_states=[node_array[n].state for n in node.neighbors])
            for node in node_array
        ]
    list(map(lambda c: c.compute(), cost_array))

    for n, c in zip(node_array, cost_array):
        n.state.apply_grad(c.get_result().grad)

    plt.axis([-2, 2, -2, 2])
    
    pos = {ix: n.loc_as_tuple() for ix, n in enumerate(node_array)}
    nx.draw_networkx(G, pos=pos)
    # for node in node_array:
    #     plt.scatter(x = node.state.pos[0], y = node.state.pos[1])
    for attractor in attractors:
        plt.scatter(x = attractor.pos[0], y = attractor.pos[1], c="Red")
    plt.draw()
    plt.pause(0.1)
    plt.clf()
