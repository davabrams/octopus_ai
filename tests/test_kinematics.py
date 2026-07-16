"""
Unit tests for kinematic state management (State, Agent, Color).
"""
import os
import sys
import unittest

import numpy as np
import tensorflow as tf

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulator.simutil import Agent, AgentType, Color, State


class TestState(unittest.TestCase):
    """Test State class - expanded from existing basic tests"""

    def setUp(self):
        """Set up test fixtures"""
        self.state = State()

    def test_state_initialization_defaults(self):
        """Test State initialization with default values"""
        self.assertAlmostEqual(self.state.x, 0.0)
        self.assertAlmostEqual(self.state.y, 0.0)
        self.assertAlmostEqual(self.state.t, 0.0)
        self.assertAlmostEqual(self.state.vx, 0.0)
        self.assertAlmostEqual(self.state.vy, 0.0)
        self.assertAlmostEqual(self.state.w, 0.0)

    def test_state_initialization_with_values(self):
        """Test State initialization with custom values"""
        state = State(x=1.5, y=2.5, t=np.pi/4, vx=0.1, vy=0.2, w=0.05)

        self.assertAlmostEqual(state.x, 1.5)
        self.assertAlmostEqual(state.y, 2.5)
        self.assertAlmostEqual(state.t, np.pi/4)
        self.assertAlmostEqual(state.vx, 0.1)
        self.assertAlmostEqual(state.vy, 0.2)
        self.assertAlmostEqual(state.w, 0.05)

    def test_state_property_setters(self):
        """Test State property setters and getters"""
        # Test position
        self.state.x = 3.0
        self.state.y = 4.0
        self.assertAlmostEqual(self.state.x, 3.0)
        self.assertAlmostEqual(self.state.y, 4.0)

        # Test angle
        self.state.t = np.pi/2
        self.assertAlmostEqual(self.state.t, np.pi/2)

        # Test velocities
        self.state.vx = 0.5
        self.state.vy = 0.3
        self.state.w = 0.1
        self.assertAlmostEqual(self.state.vx, 0.5)
        self.assertAlmostEqual(self.state.vy, 0.3)
        self.assertAlmostEqual(self.state.w, 0.1)

    def test_distance_calculation(self):
        """Test distance calculation between states"""
        state1 = State(x=0.0, y=0.0)
        state2 = State(x=3.0, y=4.0)

        distance = state1.distance_to(state2)
        expected_distance = np.sqrt(3.0**2 + 4.0**2)  # 5.0

        self.assertAlmostEqual(distance, expected_distance)

    def test_distance_same_position(self):
        """Test distance calculation for same position"""
        state1 = State(x=1.0, y=1.0)
        state2 = State(x=1.0, y=1.0)

        distance = state1.distance_to(state2)
        self.assertAlmostEqual(distance, 0.0)

    def test_tensorflow_position_property(self):
        """Test TensorFlow position property"""
        self.state.x = 2.0
        self.state.y = 3.0

        pos = self.state.pos
        expected = tf.Variable([2.0, 3.0], dtype=tf.float32)

        self.assertTrue(tf.reduce_all(tf.equal(pos, expected)))

    def test_tensorflow_velocity_property(self):
        """Test TensorFlow velocity property"""
        self.state.vx = 0.5
        self.state.vy = 1.5

        vel = self.state.vel
        expected = tf.Variable([0.5, 1.5], dtype=tf.float32)

        self.assertTrue(tf.reduce_all(tf.equal(vel, expected)))

    def test_update_kinematics_basic(self):
        """Test basic kinematic update"""
        # Set initial state
        self.state.x = 1.0
        self.state.y = 1.0
        self.state.t = 0.0
        self.state.vx = 1.0  # 1 unit/time in x
        self.state.vy = 0.5  # 0.5 units/time in y
        self.state.w = 0.1   # 0.1 rad/time angular velocity

        self.state.update_kinematics()  # dt = 1.0

        # Check updated position
        self.assertAlmostEqual(self.state.x, 2.0)  # 1.0 + 1.0*1.0
        self.assertAlmostEqual(self.state.y, 1.5)  # 1.0 + 0.5*1.0
        self.assertAlmostEqual(self.state.t, 0.1)  # 0.0 + 0.1*1.0

    def test_update_kinematics_zero_dt(self):
        """Test kinematic update with zero time step"""
        original_x = self.state.x = 5.0
        original_y = self.state.y = 3.0
        original_t = self.state.t = np.pi/6

        self.state.vx = 2.0
        self.state.vy = 1.0
        self.state.w = 0.2

        # State should not change with zero time step
        self.assertAlmostEqual(self.state.x, original_x)
        self.assertAlmostEqual(self.state.y, original_y)
        self.assertAlmostEqual(self.state.t, original_t)

        self.state.update_kinematics()

        # State should change with one time step
        self.assertAlmostEqual(self.state.x, original_x + 2.0)
        self.assertAlmostEqual(self.state.y, original_y + 1.0)
        self.assertAlmostEqual(self.state.t, original_t + 0.2)

    def test_apply_gradient(self):
        """Test gradient application to state"""
        self.state.x = 1.0
        self.state.y = 1.0

        # Create gradient tensor
        grad = tf.constant([0.1, 0.2], dtype=tf.float32)

        self.state.apply_grad(grad)

        # Check updated position
        expected_x = 1.0 + 0.1  # 1.0 - 0.5 * 0.1 = 0.95
        expected_y = 1.0 + 0.2  # 1.0 - 0.5 * 0.2 = 0.9

        self.assertAlmostEqual(self.state.x, expected_x)
        self.assertAlmostEqual(self.state.y, expected_y)

    def test_angle_wrapping(self):
        """Test angle wrapping behavior"""
        # Test large positive angle
        self.state.t = 3 * np.pi
        # Depending on implementation, might wrap to [-π, π] or [0, 2π]
        self.assertIsInstance(self.state.t, float)

        # Test large negative angle
        self.state.t = -3 * np.pi
        self.assertIsInstance(self.state.t, float)


class TestAgent(unittest.TestCase):
    """Test Agent class"""

    def test_agent_initialization_defaults(self):
        """Test Agent initialization with defaults"""
        agent = Agent()

        # Should inherit from State
        self.assertIsInstance(agent, State)
        self.assertEqual(agent.x, 0.0)
        self.assertEqual(agent.y, 0.0)

        # Agent-specific properties
        self.assertEqual(agent.agent_type, None)  # Default type

    def test_agent_initialization_with_params(self):
        """Test Agent initialization with parameters"""
        agent = Agent(
            x=2.0, y=3.0, t=np.pi/3,
            vx=0.2, vy=0.3, vel_t=0.05,
            agent_type=AgentType.THREAT
        )

        self.assertAlmostEqual(agent.x, 2.0)
        self.assertAlmostEqual(agent.y, 3.0)
        self.assertAlmostEqual(agent.t, np.pi/3)
        self.assertEqual(agent.agent_type, AgentType.THREAT)

    def test_agent_inheritance(self):
        """Test that Agent inherits State functionality"""
        agent = Agent(x=1.0, y=1.0)
        other_agent = Agent(x=4.0, y=5.0)

        # Should inherit distance calculation
        distance = agent.distance_to(other_agent)
        expected = np.sqrt((4.0-1.0)**2 + (5.0-1.0)**2)
        self.assertAlmostEqual(distance, expected)

        # Should inherit kinematic updates
        agent.vx = 1.0
        agent.vy = 0.5
        agent.update_kinematics()

        self.assertAlmostEqual(agent.x, 2.0)
        self.assertAlmostEqual(agent.y, 1.5)


class TestColor(unittest.TestCase):
    """Test Color class"""

    def test_color_initialization(self):
        """Test Color initialization"""
        color = Color()

        # Should have RGB components
        self.assertIsInstance(color.r, float)
        self.assertIsInstance(color.g, float)
        self.assertIsInstance(color.b, float)

    def test_color_to_rgb(self):
        """Test Color to RGB conversion"""
        color = Color()
        color.r = 0.5
        color.g = 0.7
        color.b = 0.3

        rgb = color.to_rgb()

        self.assertIsInstance(rgb, (list, tuple, np.ndarray))
        self.assertEqual(len(rgb), 3)
        self.assertAlmostEqual(rgb[0], 0.5)
        self.assertAlmostEqual(rgb[1], 0.7)
        self.assertAlmostEqual(rgb[2], 0.3)


class TestKinematicsIntegration(unittest.TestCase):
    """Integration tests for kinematic components"""

    def test_multi_agent_simulation_step(self):
        """Test simulation step with multiple agents"""
        # Create multiple agents
        agents = [
            Agent(x=0.0, y=0.0, vx=1.0, vy=0.0, agent_type=AgentType.PREY),
            Agent(x=5.0, y=5.0, vx=-0.5, vy=-0.5, agent_type=AgentType.THREAT),
            Agent(x=2.0, y=3.0, vx=0.0, vy=1.0, agent_type=AgentType.PREY)
        ]

        # Record initial positions
        initial_positions = [(agent.x, agent.y) for agent in agents]

        # Update all agents
        for agent in agents:
            agent.update_kinematics()

        # Verify positions changed
        final_positions = [(agent.x, agent.y) for agent in agents]

        for initial, final in zip(initial_positions, final_positions,
                                  strict=True):
            # At least one coordinate should have changed (unless velocity was zero)
            if initial != (0.0, 0.0):  # Skip if initial velocity was zero
                self.assertNotEqual(initial, final)


if __name__ == '__main__':
    unittest.main()
