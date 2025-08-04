import tensorflow as tf
from simulator.simutil import State
from abc import ABC
from copy import deepcopy
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt


@dataclass
class CostResult:
    cost = tf.Variable(initial_value=0., shape=(()), dtype=np.float32)
    grad = tf.Variable(initial_value=[0., 0.], shape=(2,), dtype=np.float32)


class CostTemplate(ABC):
    res: CostResult
    weight: tf.constant

    def __init__(self, **kwargs) -> None:
        if "weight" in kwargs:
            self.weight = kwargs.get("weight")
        else:
            self.weight = tf.constant(1.0, dtype=tf.float32)

        self.res = CostResult()

    def compute(self) -> None:
        self.res.grad = self._grad()  # 2-dimensional (x and y)
        self.res.cost = self._cost()  # 0-dimensional (scalar)
        if self.res.cost > 10:
            print(f"Exploding cost {self.__class__} : "
                  f"cost={self.res.cost} grad={self.res.grad}")

    def get_result(self) -> CostResult:
        return self.res
    
    def _cost(self) -> tf.Variable:
        # Returns the 0-dimensional cost
        # this is the square of the distance (gradients) multiplied by weight
        cost = tf.reduce_sum(tf.square(self.res.grad))
        return cost

    def _grad(self) -> tf.Variable:
        raise NotImplementedError


class ColocationRepeller(CostTemplate):
    """Keeps suckers from occupying the same exact space
    """
    min_distance_m: tf.constant

    def __init__(self, origin_state: State, destination_state: State,
                 min_dist=1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.origin: State = origin_state
        self.destination: State = destination_state
        self.min_distance_m = tf.constant(min_dist, dtype=tf.float32)

    def _grad(self) -> tf.constant:
        # return tf.subtract(self.origin.pos, self.destination.pos)
        delta = tf.subtract(self.destination.pos, self.origin.pos)
        dist = tf.norm(delta)
        if dist > self.min_distance_m:
            return tf.constant([0.0, 0.0], dtype=tf.float32)
        offset_components = tf.scalar_mul(
            self.min_distance_m, tf.math.l2_normalize(delta)
        )
        offset = tf.subtract(delta, offset_components)
        weighted_gradient = tf.scalar_mul(self.weight, offset)
        return weighted_gradient


class MaxDistanceRepeller(CostTemplate):
    """Keeps adjacent suckers on a limb from being too far away
    """
    def __init__(self,
                 origin_state: State,
                 destination_state: State,
                 max_distance: float = 3.0,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        self.origin = origin_state
        self.destination = destination_state
        self.max_distance_m = tf.constant(max_distance, dtype=tf.float32)

    def _grad(self) -> tf.constant:
        delta = tf.subtract(self.destination.pos, self.origin.pos)
        dist = tf.norm(delta)
        if dist < self.max_distance_m:
            return tf.constant([0.0, 0.0], dtype=tf.float32)
        offset_components = tf.scalar_mul(
            self.max_distance_m, tf.math.l2_normalize(delta)
        )
        offset = tf.subtract(delta, offset_components)
        weighted_gradient = tf.scalar_mul(self.weight, offset)
        return weighted_gradient


class PointAttractor(CostTemplate):
    """Simple parabolic attractor
    """
    def __init__(self,
                 origin_state: State,
                 attraction_point: State,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        self.origin = origin_state
        self.attraction_point = attraction_point

    def _grad(self) -> tf.constant:
        delta = tf.subtract(self.attraction_point.pos, self.origin.pos)
        dist = tf.norm(delta)
        if dist < 0.1:
            return tf.constant([0.0, 0.0], dtype=tf.float32)
        # nominal_gradient = tf.math.log(dist + 1)
        nominal_gradient = tf.math.exp(tf.negative(x=dist))
        offset_components = tf.scalar_mul(
            nominal_gradient, tf.math.l2_normalize(delta)
        )
        weighted_gradient = tf.scalar_mul(self.weight, offset_components)
        return weighted_gradient
    

class CollisionCost(CostTemplate):
    # Given an obstacle, this is an astronomical cost (+10000) with a zero
    # gradient
    # step 1 would be to generate edges between neighboring states
    # step 2 would be to identify intersections between node edges and
    # object edges
    # if there is an overlap, there is a collision, add the cost
    pass


class AllCosts(CostTemplate):
    # Contains all the costs such that they can be executed once
    # Also includes a line search to find the best alpha every iteration
    costs: List[CostTemplate]

    def __init__(self,
                 origin_state: State,
                 all_nodes: Optional[List[State]] = None,
                 neighbor_states: Optional[List[State]] = None,
                 attractor_states: Optional[List[State]] = None,
                 max_distance: float = 3.0,
                 min_distance: float = 3.0,
                 alphas: list[float] = [1.0],
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        self.origin = origin_state
        self.all = all_nodes
        self.neighbors = neighbor_states
        self.attractors = attractor_states
        self.alphas = alphas

        self.max_distance_m = tf.constant(max_distance, dtype=tf.float32)
        self.min_distance_m = tf.constant(min_distance, dtype=tf.float32)

        self.costs = []
        if self.all:
            for node in self.all:
                self.costs.append(ColocationRepeller(
                    self.origin, node, min_distance, weight=0.2
                ))
        if self.neighbors:
            for neighbor in self.neighbors:
                self.costs.append(MaxDistanceRepeller(
                    self.origin, neighbor, max_distance, weight=1.0
                ))
        if self.attractors:
            for attractor in self.attractors:
                self.costs.append(PointAttractor(
                    self.origin, attractor, weight=0.2
                ))

    def compute(self) -> None:
        for c in self.costs:
            c.compute()
            self.res.cost = tf.add(self.res.cost, c.res.cost)
            self.res.grad = tf.add(self.res.grad, c.res.grad)
    
    def line_search(self) -> Optional[Tuple[float]]:
        # compute each cost's gradient once
        # apply the gradient, once for each alpha value
        # compute the cost over all alphas
        # select the alpha associated with the smallest cost
        # return the alpha
        init_grad = tf.Variable(0.0)
        for c in self.costs:
            temp_grad = c._grad()
            init_grad = tf.add(init_grad, temp_grad)

        best_cost: tf.Tensor = tf.constant(np.inf)
        best_grad: tf.Tensor
        best_alpha: tf.Tensor
        for alpha in self.alphas:
            temp_origin = deepcopy(self.origin)
            temp_grad = tf.scalar_mul(alpha, init_grad)
            temp_origin.apply_grad(temp_grad)
            temp_cost = AllCosts(
                origin_state=temp_origin,
                all_nodes=self.all,
                neighbor_states=self.neighbors,
                attractor_states=self.attractors,
                max_distance=self.max_distance_m,
                min_distance=self.min_distance_m
                )
            temp_cost.compute()
            res = temp_cost.get_result()
            cost = res.cost
            # print(f"A:{alpha}: c: {cost}")
            if cost < best_cost:
                best_grad = temp_grad
                best_alpha = alpha
                best_cost = cost
        
        # print(f"selected {best_alpha}")

        return best_alpha, best_cost, best_grad



def cost_heatmap():
    neighbors = [State(0.0, 0.0)]
    attrs = [State(1.5, 1.5)]
    mat_output = []
    for i in np.arange(-3.5, 3.5, 0.1):
        line = []
        for j in np.arange(-3.5, 3.5, 0.1):
            origin = State(i, j)
            costs = AllCosts(origin_state=origin,
                             neighbor_states=neighbors,
                             attractor_states=attrs)
            costs.compute()
            results = costs.get_result()
            line.append(float(results.cost))
        mat_output.append(line)
    plt.matshow(mat_output)
    plt.show()


def plot_graph():
    origin = State(0.0, 0)
    attractor = State(0, np.pi/2.0)

    plt.ion()

    for val in np.arange(0, 30, 0.1):
        x_pos = 2 * np.sin(val)
        y_pos = 2 * np.cos(val / 3)
        attractor.x = x_pos
        attractor.y = y_pos
        attractors = [attractor]
        costs = AllCosts(
            origin_state=origin, attractor_states=attractors
        )
        costs.compute()
        res = costs.get_result()
        origin.apply_grad(res.grad)
        plt.axis([-3, 3, -3, 3])
        plt.scatter(x=origin.pos[0], y=origin.pos[1])
        plt.scatter(x=attractor.pos[0], y=attractor.pos[1])
        plt.draw()
        plt.pause(0.1)
        plt.clf()


if __name__ == "__main__":
    cost_heatmap()
    plot_graph()