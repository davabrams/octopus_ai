"""
Regression tests for LIMB-model sucker adjacency.

Limb.find_color used to pass agent_range_radius (how far an ARM senses
AGENTS, 5.0) to find_adjacents, which builds the LIMB model's neighbour
list. datagen builds its training adjacents with adjacency_radius
(1.0). Two different parameters for two different concepts, so the model
was trained on one neighbourhood and served another - and at 5.0 the
filter is a no-op, returning the whole limb for every sucker.
"""
import os
import sys
import unittest
from unittest.mock import Mock

from tensorflow import keras

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulator.octopus_generator import Limb
from simulator.simutil import MLMode, MovementMode
from helpers import make_config


class TestLimbAdjacencyRadius(unittest.TestCase):
    ADJACENCY_RADIUS = 1.0   # what datagen trains the LIMB model with
    SENSING_RADIUS = 5.0     # how far an arm senses AGENTS - a different job

    def setUp(self):
        self.cfg = make_config(
            x_len=15, y_len=15,
            limb_rows=16, limb_cols=2,
            limb_movement_mode=MovementMode.RANDOM,
            adjacency_radius=self.ADJACENCY_RADIUS,
            agent_range_radius=self.SENSING_RADIUS,
        )
        self.limb = Limb(x_octo=7.5, y_octo=7.5, init_angle=0.0,
                         params=self.cfg)

    def test_limb_exposes_adjacency_radius(self):
        self.assertEqual(self.limb.adjacency_radius,
                         self.ADJACENCY_RADIUS)

    def test_adjacency_and_agent_sensing_are_distinct(self):
        """They are different concepts and must not be conflated."""
        self.assertNotEqual(self.limb.adjacency_radius,
                            self.limb.agent_range_radius)

    def test_find_color_uses_adjacency_radius_not_agent_range(self):
        """The LIMB path must ask for neighbours at adjacency_radius, the
        same radius datagen trains with."""
        seen = []
        real = self.limb.find_adjacents

        def spy(sucker, radius):
            seen.append(radius)
            return real(sucker, radius)

        self.limb.find_adjacents = spy
        model = Mock(spec=keras.Model)  # isinstance(model, keras.Model)
        model.predict.return_value = [[0.5]]
        surf = Mock()
        surf.get_val.return_value = [0.5, 0.5, 0.5]  # RGB triple

        self.limb.find_color(surf, MLMode.LIMB, model)

        self.assertTrue(seen, "find_adjacents was never called")
        self.assertTrue(
            all(r == self.ADJACENCY_RADIUS for r in seen),
            f"expected every call at adjacency_radius="
            f"{self.ADJACENCY_RADIUS}, got {set(seen)}")
        self.assertNotIn(self.SENSING_RADIUS, seen)

    def test_adjacency_radius_actually_filters(self):
        """At the old radius (5) every sucker was 'adjacent' to every other,
        making the filter meaningless for a limb spanning <= 4.5 units."""
        s = self.limb.suckers[0]
        n_adjacency = len(self.limb.find_adjacents(s, 1.0))
        n_agent_range = len(self.limb.find_adjacents(s, 5.0))
        total = len(self.limb.suckers)
        self.assertEqual(n_agent_range, total)      # no-op filter
        self.assertLess(n_adjacency, total)         # actually filters


if __name__ == '__main__':
    unittest.main()
