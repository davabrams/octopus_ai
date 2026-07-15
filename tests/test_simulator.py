import unittest
import numpy as np
import tensorflow as tf
from simulator.surface_generator import RandomSurface
from simulator.agent_generator import AgentGenerator
from simulator.simutil import MovementMode, AgentType, State


class TestSurfaceGenerator(unittest.TestCase):
    def test_random_surface_color(self) -> None:
        """surface_grayscale=False gives a full-colour surface: the grid is
        (y, x, 3) and each get_val is an [r, g, b] triple with independent
        channels."""
        def test_surf_with_size(max_dim: int) -> None:
            params = {
                "x_len": max_dim,
                "y_len": max_dim,
                "rand_seed": 0,
                "surface_grayscale": False,
            }
            surf = RandomSurface(params)
            self.assertEqual(surf.grid.shape, (max_dim, max_dim, 3))
            rgb = surf.get_val(1, 1)
            self.assertEqual(np.asarray(rgb).shape, (3,))
            self.assertTrue(all(0.0 <= float(c) <= 1.0 for c in rgb))

        for max_dim in range(2, 10):
            test_surf_with_size(max_dim)

    def test_random_surface_grayscale(self) -> None:
        params = {
            "x_len": 6,
            "y_len": 6,
            "rand_seed": 0,
            "surface_grayscale": True,
        }
        surf = RandomSurface(params)
        self.assertEqual(surf.grid.shape, (6, 6, 3))
        vals = [surf.get_val(i, j) for i in range(6) for j in range(6)]
        # Grayscale: every cell has r == g == b, in [0, 1).
        for rgb in vals:
            self.assertTrue(0.0 <= float(rgb[0]) < 1.0)
            self.assertAlmostEqual(float(rgb[0]), float(rgb[1]))
            self.assertAlmostEqual(float(rgb[0]), float(rgb[2]))
        # a 36-cell uniform random grid should not be binary
        self.assertGreater(len(set(round(float(v[0]), 3) for v in vals)), 2)

class TestAgentGenerator(unittest.TestCase):
    def test_agent_generator(self) -> None:
        max_dim = 5
        max_vel = 1
        max_theta = 1
        params = {
            "x_len": max_dim,
            "y_len": max_dim,
            "agent_max_velocity": max_vel,
            "agent_max_theta": max_theta,
            "agent_movement_mode": MovementMode.RANDOM,
            "agent_range_radius": 1,
            "agent_prey_capture_radius": 0.3,
            "agent_respawn_captured_prey": False,
            "rand_seed": 0
        }
        agent_gen = AgentGenerator(params)
        agent_gen.agents = []  # Clear class-level shared list
        agent_gen.generate(1, fixed_agent_type=AgentType.PREY)
        self.assertEqual(len(agent_gen.agents), 1)
        agent_gen.generate(1, fixed_agent_type=AgentType.THREAT)
        self.assertEqual(len(agent_gen.agents), 2)
        for agent in agent_gen.agents:
            self.assertLessEqual(agent.vx, max_vel)
            self.assertLessEqual(agent.vy, max_vel)

        old_agent_states = [(agent.x, agent.y, agent.t, agent.vx, agent.vy, agent.w) for agent in agent_gen.agents]

        agent_gen.increment_all()
        for agent in agent_gen.agents:
            self.assertLessEqual(agent.vx, max_vel)
            self.assertLessEqual(agent.vy, max_vel)

        new_agent_states = [(agent.x, agent.y, agent.t, agent.vx, agent.vy, agent.w) for agent in agent_gen.agents]

        self.assertNotEqual(old_agent_states, new_agent_states)

        for state_diff in zip(old_agent_states, new_agent_states):
            x_diff = state_diff[0][0] - state_diff[1][0]
            y_diff = state_diff[0][1] - state_diff[1][1]
            t_diff = (state_diff[0][2] - state_diff[1][2]) % (2 * np.pi)
            self.assertLessEqual(x_diff, max_vel)
            self.assertLessEqual(y_diff, max_vel)
            self.assertTrue(t_diff <= max_theta or t_diff > 2 * np.pi - max_theta)

class TestKinematicPrimitives(unittest.TestCase):
    def test_state_obect(self) -> None:

        # test initial values
        test_state = State()
        self.assertEqual(test_state.x, 0.0)
        self.assertEqual(test_state.y, 0.0)
        self.assertEqual(test_state.t, 0.0)
        self.assertEqual(test_state.vx, 0.0)
        self.assertEqual(test_state.vy, 0.0)
        self.assertEqual(test_state.w, 0.0)

        # test setters and getters
        test_state.x = 1.0
        test_state.y = 1.0
        test_state.t = 1.0
        test_state.vx = 1.0
        test_state.vy = 1.0
        test_state.w = 1.0
        self.assertEqual(test_state.x, 1.0)
        self.assertEqual(test_state.y, 1.0)
        self.assertEqual(test_state.t, 1.0)
        self.assertEqual(test_state.vx, 1.0)
        self.assertEqual(test_state.vy, 1.0)
        self.assertEqual(test_state.w, 1.0)

        distance = test_state.distance_to(State())
        self.assertAlmostEqual(distance, np.sqrt(2))

        self.assertTrue(tf.reduce_all(tf.equal(test_state.pos, tf.Variable([1, 1], dtype=tf.float32))))
        self.assertTrue(tf.reduce_all(tf.equal(test_state.vel, tf.Variable([1, 1], dtype=tf.float32))))


if __name__ == '__main__':
    unittest.main()
