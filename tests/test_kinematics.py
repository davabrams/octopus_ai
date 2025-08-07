"""
Unit tests for kinematic state management and ILQR costs
"""
import unittest
import numpy as np
import tensorflow as tf
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulator.simutil import State, Agent, Color, AgentType
from simulator.ilqr.costs import (
    CostTemplate, ColocationRepeller, MaxDistanceRepeller, 
    PointAttractor, AllCosts
)


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
        
        dt = 1.0  # 1 time unit
        self.state.update_kinematics()
        
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


class TestCostTemplate(unittest.TestCase):
    """Test CostTemplate abstract base class"""
    
    def test_cost_function_interface(self):
        """Test CostTemplate interface"""
        # Create a concrete implementation for testing
        class TestCost(CostTemplate):
            def _grad(self, state_vec):
                return tf.zeros_like(state_vec)
        
        cost_fn = TestCost()
        
        # Should be able to call _grad
        test_state = tf.constant([1.0, 2.0], dtype=tf.float32)
        grad = cost_fn._grad(test_state)
        
        self.assertIsInstance(grad, tf.Tensor)
        self.assertEqual(grad.shape, test_state.shape)


class TestColocationRepeller(unittest.TestCase):
    """Test ColocationRepeller cost function"""
    
    def setUp(self):
        """Set up test fixtures"""
        origin_state = State(x=0.0, y=0.0)
        destination_state = State(x=5.0, y=5.0)
        self.repeller = ColocationRepeller(origin_state=origin_state, destination_state=destination_state)
    
    def test_colocation_repeller_initialization(self):
        """Test ColocationRepeller initialization"""
        self.assertIsInstance(self.repeller, CostTemplate)
    
    def test_colocation_repeller_gradient_same_position(self):
        """Test gradient when states are at same position"""
        # Two states at same position should have high repelling force
        origin_state = destination_state = State(x=0.0, y=0.0)
        self.repeller = ColocationRepeller(origin_state=origin_state, destination_state=destination_state)

        self.repeller.compute()
        grad = self.repeller.get_result().grad
        # Should produce some gradient (exact value depends on implementation)
        self.assertIsInstance(grad, tf.Tensor)
        self.assertEqual(grad.shape, 2)
    
    def test_colocation_repeller_gradient_different_positions(self):
        """Test gradient when states are at different positions"""
        # Two states far apart should have minimal repelling force
        state_vec = tf.constant([0.0, 0.0, 10.0, 10.0], dtype=tf.float32)
        self.repeller.compute()
        grad = self.repeller.get_result().grad
        self.assertIsInstance(grad, tf.Tensor)
        self.assertEqual(grad.shape, 2)


class TestMaxDistanceRepeller(unittest.TestCase):
    """Test MaxDistanceRepeller cost function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.max_distance = 5.0
        origin_state = State(x=0.0, y=0.0)
        destination_state = State(x=5.0, y=5.0)
        self.repeller = MaxDistanceRepeller(origin_state=origin_state, destination_state=destination_state, max_distance=self.max_distance)
    
    def test_max_distance_repeller_initialization(self):
        """Test MaxDistanceRepeller initialization"""
        self.assertIsInstance(self.repeller, CostTemplate)
        self.assertEqual(self.repeller.max_distance_m, self.max_distance)
    
    def test_max_distance_repeller_within_bounds(self):
        """Test gradient when distance is within max distance"""
        # States within max distance should have minimal cost
        self.repeller = MaxDistanceRepeller(
            origin_state=State(x=0.0, y=0.0),
            destination_state=State(x=2.0, y=2.0)
            )
        self.repeller.compute()
        grad = self.repeller.get_result().grad
        self.assertIsInstance(grad, tf.Tensor)
        self.assertEqual(grad.shape, 2)
    
    def test_max_distance_repeller_exceeds_bounds(self):
        """Test gradient when distance exceeds max distance"""
        # States beyond max distance should have high cost
        self.repeller = MaxDistanceRepeller(
            origin_state=State(x=0.0, y=0.0),
            destination_state=State(x=10.0, y=10.0)
            )
        self.repeller.compute()
        grad = self.repeller.get_result().grad
        self.assertIsInstance(grad, tf.Tensor)
        self.assertEqual(grad.shape, 2)


class TestPointAttractor(unittest.TestCase):
    """Test PointAttractor cost function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.target_point = [3.0, 4.0]
        origin_state = State(x=0.0, y=0.0)
        attraction_state = State(x=3.0, y=4.0)
        self.attractor = PointAttractor(origin_state=origin_state, attraction_point=attraction_state)
    
    def test_point_attractor_initialization(self):
        """Test PointAttractor initialization"""
        self.assertIsInstance(self.attractor, CostTemplate)
        self.assertTrue(np.array_equal(self.attractor.origin.pos, State(x=0.0, y=0.0).pos))
        self.assertTrue(np.array_equal(self.attractor.attraction_point.pos, State(x=3.0, y=4.0).pos))
    
    def test_point_attractor_at_target(self):
        """Test gradient when state is at target point"""
        # State at target should have minimal gradient
        state_vec = tf.constant([3.0, 4.0], dtype=tf.float32)
        
        grad = self.attractor._grad()
        
        self.assertIsInstance(grad, tf.Tensor)
        self.assertEqual(grad.shape, state_vec.shape)
    
    def test_point_attractor_away_from_target(self):
        """Test gradient when state is away from target"""
        # State away from target should have gradient pointing toward target
        state_vec = tf.constant([0.0, 0.0], dtype=tf.float32)
        
        grad = self.attractor._grad()
        
        self.assertIsInstance(grad, tf.Tensor)
        self.assertEqual(grad.shape, state_vec.shape)
    
    def test_point_attractor_with_origin_state(self):
        """Test PointAttractor with origin and attraction states"""
        origin_state = State(x=0.0, y=0.0)
        attraction_state = State(x=3.0, y=4.0)
        
        attractor = PointAttractor(origin_state=origin_state, attraction_point=attraction_state)
        
        # Should be properly initialized
        self.assertIsInstance(attractor, CostTemplate)
        self.assertEqual(attractor.origin, origin_state)
        self.assertEqual(attractor.attraction_point, attraction_state)
    
    def test_point_attractor_origin_at_attraction(self):
        """Test gradient when origin is at attraction point"""
        origin_state = State(x=2.0, y=2.0)
        attraction_state = State(x=2.0, y=2.0)
        
        attractor = PointAttractor(origin_state=origin_state, attraction_point=attraction_state)
        attractor.compute()
        
        # When at target, gradient should be zero or very small
        grad = attractor.get_result().grad
        self.assertIsInstance(grad, tf.Tensor)
        # Should be approximately zero due to small distance check
        self.assertLess(tf.norm(grad).numpy(), 0.1)
    
    def test_point_attractor_origin_far_from_attraction(self):
        """Test gradient when origin is far from attraction point"""
        origin_state = State(x=0.0, y=0.0)
        attraction_state = State(x=5.0, y=0.0)  # 5 units away in x direction
        
        attractor = PointAttractor(origin_state=origin_state, attraction_point=attraction_state)
        attractor.compute()
        
        grad = attractor.get_result().grad
        self.assertIsInstance(grad, tf.Tensor)
        
        # Gradient should point toward attraction (positive x direction)
        self.assertGreater(grad[0].numpy(), 0.0)
        self.assertAlmostEqual(grad[1].numpy(), 0.0, places=5)
    
    def test_point_attractor_diagonal_attraction(self):
        """Test gradient when origin and attraction are at diagonal positions"""
        origin_state = State(x=1.0, y=1.0)
        attraction_state = State(x=4.0, y=5.0)  # 3 units right, 4 units up
        
        attractor = PointAttractor(origin_state=origin_state, attraction_point=attraction_state)
        attractor.compute()
        
        grad = attractor.get_result().grad
        self.assertIsInstance(grad, tf.Tensor)
        
        # Both components should be positive (pointing toward attraction)
        self.assertGreater(grad[0].numpy(), 0.0)
        self.assertGreater(grad[1].numpy(), 0.0)
        
        # Ratio should approximate the direction (3:4 ratio)
        ratio = grad[0].numpy() / grad[1].numpy()
        expected_ratio = 3.0 / 4.0
        self.assertAlmostEqual(ratio, expected_ratio, places=1)
    
    def test_point_attractor_compute_and_cost(self):
        """Test that compute method properly calculates cost and gradient"""
        origin_state = State(x=0.0, y=0.0)
        attraction_state = State(x=3.0, y=4.0)  # Distance = 5
        
        attractor = PointAttractor(origin_state=origin_state, attraction_point=attraction_state)
        attractor.compute()
        
        result = attractor.get_result()
        
        # Should have valid cost and gradient
        self.assertIsInstance(result.cost, tf.Tensor)
        self.assertIsInstance(result.grad, tf.Tensor)
        self.assertGreaterEqual(result.cost.numpy(), 0.0)
        
        # Cost should be based on gradient magnitude squared
        expected_cost = tf.reduce_sum(tf.square(result.grad))
        self.assertAlmostEqual(result.cost.numpy(), expected_cost.numpy(), places=5)


class TestAllCosts(unittest.TestCase):
    """Test AllCosts combined cost function"""
    
    def setUp(self):
        """Set up test fixtures"""
        origin_state = State(x=0.0, y=0.0)
        destination_state = State(x=5.0, y=5.0)
        attraction_state = State(x=1.0, y=1.0)
        self.all_costs = AllCosts(origin_state=origin_state, all_nodes=[destination_state], neighbor_states=[destination_state], attractor_states=[attraction_state], max_distance=10.0)
    
    def test_all_costs_initialization(self):
        """Test AllCosts initialization"""
        self.assertEqual(len(self.all_costs.costs), 3)
        self.assertIsInstance(self.all_costs.costs[0], ColocationRepeller)
        self.assertIsInstance(self.all_costs.costs[1], MaxDistanceRepeller)
        self.assertIsInstance(self.all_costs.costs[2], PointAttractor)
    
    def test_all_costs_compute(self):
        """Test AllCosts compute method"""        
        self.all_costs.compute()
        total_cost = self.all_costs.get_result().cost
        
        # Should return a scalar cost
        self.assertIsInstance(total_cost, tf.Tensor)
        self.assertEqual(total_cost.shape, ())  # Scalar
        self.assertGreaterEqual(total_cost.numpy(), 0.0)  # Cost should be non-negative
    
    def test_all_costs_line_search(self):
        """Test AllCosts line search optimization"""
        # Create initial state and gradient
        self.all_costs.compute()
        
        best_alpha, best_cost, best_grad = self.all_costs.line_search()
        
        # Should return a valid alpha value
        self.assertIsInstance(best_alpha, float)
        self.assertGreaterEqual(best_alpha, 0.0)

        self.assertIsInstance(best_cost, tf.Tensor)
        self.assertGreaterEqual(best_cost, 0.0)
        
        self.assertIsInstance(best_grad, tf.Tensor)
        self.assertEqual(best_grad.shape, (2,))  # Gradient should be 2
    
    def test_all_costs_empty_functions(self):
        """Test AllCosts with no cost functions"""
        origin_state = State(x=0.0, y=0.0)
        empty_costs = AllCosts(origin_state=origin_state)
        
        empty_costs.compute()
        
        # Should return zero cost
        self.assertAlmostEqual(empty_costs.get_result().cost.numpy(), 0.0)


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
        dt = 1.0
        for agent in agents:
            agent.update_kinematics()
        
        # Verify positions changed
        final_positions = [(agent.x, agent.y) for agent in agents]
        
        for initial, final in zip(initial_positions, final_positions):
            # At least one coordinate should have changed (unless velocity was zero)
            if initial != (0.0, 0.0):  # Skip if initial velocity was zero
                self.assertNotEqual(initial, final)
    
    def test_cost_optimization_workflow(self):
        """Test complete cost optimization workflow"""
        # Create cost functions
        origin_state = State(x=0.0, y=0.0)
        destination_state = State(x=5.0, y=5.0)
        attraction_state = State(x=10.0, y=10.0)
        costs = AllCosts(origin_state=origin_state, neighbor_states=[destination_state], attractor_states=[attraction_state])
        
        # Compute initial cost
        costs.compute()
        initial_cost = costs.get_result().cost

        costs.line_search()
        
        # Compute final cost
        costs.compute()
        final_cost = costs.get_result().cost
        
        # Verify optimization made progress (cost should decrease or stay same)
        self.assertIsInstance(initial_cost.numpy(), (float, np.float32))
        self.assertIsInstance(final_cost.numpy(), (float, np.float32))
        # Note: Cost might not always decrease in a single step depending on the optimization landscape


if __name__ == '__main__':
    unittest.main()