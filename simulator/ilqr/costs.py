import tensorflow as tf
from simulator.simutil import State
from abc import ABC
import numpy as np
from dataclasses import dataclass
from typing import List
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
        self.res.grad = self._grad() # 2-dimensional (x and y)
        self.res.cost = self._cost() # 0-dimensional (scalar)

    def get_result(self) -> CostResult:
        return self.res
    
    def _cost(self) -> tf.Variable:
        raise NotImplementedError

    def _grad(self) -> tf.Variable:
        raise NotImplementedError

class ColocationRepeller(CostTemplate):
    """Keeps suckers from occupying the same exact space
    """
    min_distance_m: tf.constant

    def __init__(self, origin_state: State, destination_state: State, min_dist = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.origin: State = origin_state
        self.destination: State = destination_state
        self.min_distance_m = tf.constant(min_dist, dtype=tf.float32)

    def _cost(self) -> tf.constant:
        # Returns the 0-dimensional cost
        # this is the square of the distance (gradients) multiplied by weight
        raw_cost = tf.reduce_sum(tf.square(self.res.grad))
        weighted_cost = tf.multiply(raw_cost, self.weight)
        return weighted_cost

    def _grad(self) -> tf.constant:
        # return tf.subtract(self.origin.pos, self.destination.pos)
        delta = tf.subtract(self.destination.pos, self.origin.pos)
        dist = tf.norm(delta)
        if dist > self.min_distance_m:
            return tf.constant([0.0, 0.0], dtype=tf.float32)
        offset_components = tf.scalar_mul(self.min_distance_m, tf.math.l2_normalize(delta))
        offset = tf.subtract(delta, offset_components)
        return offset

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

    def _cost(self) -> tf.constant:
        # Returns the 0-dimensional cost
        raw_cost = tf.reduce_sum(tf.square(self.res.grad))
        weighted_cost = tf.multiply(raw_cost, self.weight)
        return weighted_cost

    def _grad(self) -> tf.constant:
        delta = tf.subtract(self.destination.pos, self.origin.pos)
        dist = tf.norm(delta)
        if dist < self.max_distance_m:
            return tf.constant([0.0, 0.0], dtype=tf.float32)
        offset_components = tf.scalar_mul(self.max_distance_m, tf.math.l2_normalize(delta))
        offset = tf.subtract(delta, offset_components)
        return offset

class AllCosts(CostTemplate):
    costs: List[CostTemplate]

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
        self.costs = [
            MaxDistanceRepeller(origin_state, destination_state, max_distance),
            ColocationRepeller(origin_state, destination_state)
            ]

    def compute(self) -> None:
        for c in self.costs:
            c.compute()
            self.res.cost = tf.add(self.res.cost, c.res.cost)
            self.res.grad = tf.add(self.res.grad, c.res.grad)

if __name__ == "__main__":
    state_1 = State(0.0, 0.0)
    print(state_1.pos)

    mat_output = []
    for i_ix, i in enumerate(np.arange(-3.5, 3.5, 0.1)):
        line = []
        for j_ix, j in enumerate(np.arange(-3.5, 3.5, 0.1)):
            state_2 = State(i, j)
            costs = AllCosts(state_1, state_2)
            costs.compute()
            res = costs.get_result()
            line.append(float(res.cost))
        mat_output.append(line)
    plt.matshow(mat_output)
    plt.show()