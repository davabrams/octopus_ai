"""
Unit tests for octopus_generator module - core simulation classes
"""
import unittest
import numpy as np
import tensorflow as tf
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulator.octopus_generator import Sucker, Limb, Octopus
from simulator.simutil import State, Color, MLMode, InferenceLocation, MovementMode
from simulator.surface_generator import RandomSurface
from OctoConfig import GameParameters


class TestSucker(unittest.TestCase):
    """Test the Sucker class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.params = GameParameters.copy()
        self.params['x_len'] = 10
        self.params['y_len'] = 10
        self.params['rand_seed'] = 42
        
        # Create a mock surface
        self.mock_surface = Mock()
        self.mock_surface.get_val.return_value = 0.5
        
        # Create test sucker
        self.sucker = Sucker(
            x=5.0, y=5.0, 
            params=self.params
        )
    
    def test_sucker_initialization(self):
        """Test sucker is properly initialized"""
        self.assertAlmostEqual(self.sucker.x, 5.0)
        self.assertAlmostEqual(self.sucker.y, 5.0)
        self.assertIsInstance(self.sucker.c, Color)
    
    def test_distance_to(self):
        """Test distance calculation between suckers"""
        other_sucker = Sucker(x=8.0, y=9.0, params=self.params)
        expected_distance = np.sqrt((8.0 - 5.0)**2 + (9.0 - 5.0)**2)
        self.assertAlmostEqual(self.sucker.distance_to(other_sucker), expected_distance)
    
    def test_get_surf_color_at_this_sucker(self):
        """Test surface color retrieval"""
        # Test valid coordinates
        self.mock_surface.get_val.return_value = 0.7
        color = self.sucker.get_surf_color_at_this_sucker(self.mock_surface)
        self.assertEqual(color.r, 0.7)
        self.assertEqual(color.b, 0.7)
        self.assertEqual(color.g, 0.7)
        self.mock_surface.get_val.assert_called_with(5, 5)
    
    def test_find_color_change_heuristic(self):
        """Test heuristic color change logic"""
        self.sucker.c.r = 0.5
        surface_color = 0.8
        max_change = 0.1
        
        result = self.sucker._find_color_change(surface_color, max_change)
        
        # Should move towards surface color within max change limit
        self.assertLessEqual(abs(result - 0.5), max_change)
        self.assertGreater(result, 0.5)  # Should move towards higher surface color


class TestLimb(unittest.TestCase):
    """Test the Limb class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.params = GameParameters.copy()
        self.params['limb_rows'] = 4
        self.params['limb_cols'] = 2
        self.params['x_len'] = 10
        self.params['y_len'] = 10
        
        # Create mock surface
        self.mock_surface = Mock()
        self.mock_surface.get_val.return_value = 0.5
        
        # Create test limb
        root_state=State(x=5.0, y=5.0, t=0.0)
        initial_angle = 0
        self.limb = Limb(x_octo=root_state.x,y_octo=root_state.y,init_angle=initial_angle,params=self.params)
    
    def test_limb_initialization(self):
        """Test limb is properly initialized"""
        # Should have correct number of suckers
        expected_suckers = self.params['limb_rows'] * self.params['limb_cols']
        self.assertEqual(len(self.limb.suckers), expected_suckers)
        
        # All suckers should be initialized
        for sucker in self.limb.suckers:
            self.assertIsInstance(sucker, Sucker)
    
    def test_gen_centerline(self):
        """Test centerline generation"""
        self.limb._gen_centerline(0, 0, 0)
        
        # Should have correct length
        self.assertEqual(len(self.limb.center_line), self.params['limb_rows'])
        
        # Points should be ordered from root outward
        for i in range(len(self.limb.center_line) - 1):
            dist_from_root_i = np.sqrt(self.limb.center_line[i].x**2 + self.limb.center_line[i].y**2)
            dist_from_root_j = np.sqrt(self.limb.center_line[i+1].x**2 + self.limb.center_line[i+1].y**2)
            self.assertLessEqual(dist_from_root_i, dist_from_root_j)
    
    def test_refresh_sucker_locations(self):
        """Test sucker location refresh"""
        old_positions = [(s.x, s.y) for s in self.limb.suckers]
        
        # Change limb orientation and refresh
        self.limb.root_state.t = np.pi / 4
        self.limb._refresh_sucker_locations()
        
        new_positions = [(s.x, s.y) for s in self.limb.suckers]
        
        # Positions should have changed
        self.assertNotEqual(old_positions, new_positions)
    
    def test_find_adjacents(self):
        """Test adjacent sucker finding"""
        if self.limb.suckers:
            target_sucker = self.limb.suckers[0]
            adjacents_with_dist = self.limb.find_adjacents(target_sucker, radius=2.0)
            adjacents = [a for a, d in adjacents_with_dist]
            
            # Should return list of suckers
            self.assertIsInstance(adjacents, list)
            
            # All adjacents should be within radius
            for adj in adjacents:
                distance = target_sucker.distance_to(adj)
                self.assertLessEqual(distance, 2.0)
    
    
    def test_move_random_mode(self):
        """Test limb movement in random mode"""
        self.params['limb_movement_mode'] = MovementMode.RANDOM
        
        self.limb.move(1.0, 1.0)
        
        # Angle should have changed (with high probability)
        # Using a tolerance due to random nature
        self.assertIsInstance(self.limb.root_state.t, float)


class TestOctopus(unittest.TestCase):
    """Test the Octopus class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.params = GameParameters.copy()
        self.params['octo_num_arms'] = 4
        self.params['x_len'] = 10
        self.params['y_len'] = 10
        
        # Create mock surface
        self.mock_surface = Mock()
        self.mock_surface.get_val.return_value = 0.5
        
        # Create test octopus
        self.octopus = Octopus(
            body_state=State(x=5.0, y=5.0, t=0.0),
            params=self.params
        )
    
    def test_octopus_initialization(self):
        """Test octopus is properly initialized"""
        # Should have correct number of limbs
        self.assertEqual(len(self.octopus.limbs), self.params['octo_num_arms'])
        
        # All limbs should be properly initialized
        for limb in self.octopus.limbs:
            self.assertIsInstance(limb, Limb)
    
    def test_find_color_parallel(self):
        """Test parallel color finding across limbs"""
        with patch.object(self.octopus.limbs[0], 'find_color', return_value=[0.1, 0.2]):
            colors = self.octopus.find_color(self.mock_surface)
            
            # Should return nested list of colors
            self.assertIsInstance(colors, list)
            self.assertEqual(len(colors), len(self.octopus.limbs))
    
    def test_set_color(self):
        """Test color setting across all suckers"""
        # Create mock colors data        
        self.octopus.set_color(self.mock_surface)
        
        # Verify colors were set (basic smoke test)
        for limb in self.octopus.limbs:
            for sucker in limb.suckers:
                self.assertIsInstance(sucker.c.r, float)
    
    def test_visibility_calculation(self):
        """Test MSE visibility calculation"""
        # Set up some test colors
        for limb in self.octopus.limbs:
            for sucker in limb.suckers:
                sucker.c.r = 0.5
        
        visibility = self.octopus.visibility(self.mock_surface)
        
        # Should return a float MSE value
        self.assertIsInstance(visibility, float)
        self.assertGreaterEqual(visibility, 0.0)
    
    def test_move_body(self):
        """Test octopus body movement"""
        original_pos = (self.octopus.x, self.octopus.y)
        
        # Mock some agents for attract/repel logic
        mock_agents = [Mock()]
        mock_agents[0].x = 7.0
        mock_agents[0].y = 7.0
        mock_agents[0].agent_type = 1  # Threat
        
        self.octopus.move(ag=mock_agents)
        
        # Position might have changed based on attract/repel logic
        self.assertIsInstance(self.octopus.x, float)
        self.assertIsInstance(self.octopus.y, float)


class TestIntegration(unittest.TestCase):
    """Integration tests for octopus components"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.params = GameParameters.copy()
        self.params['x_len'] = 5
        self.params['y_len'] = 5
        self.params['octo_num_arms'] = 2
        self.params['limb_rows'] = 2
        self.params['limb_cols'] = 1
        
        # Use real surface for integration tests
        self.surface = RandomSurface(self.params)
        
        self.octopus = Octopus(
            body_state=State(x=2.5, y=2.5, t=0.0),
            params=self.params
        )
    
    def test_full_simulation_step(self):
        """Test a complete simulation step"""
        # Should be able to run without errors
        # try:
        # Move octopus
        self.octopus.move()
        
        # Find colors (heuristic mode for integration test)
        self.params['inference_location'] = InferenceLocation.LOCAL
        
        # Set colors
        self.octopus.set_color(surf=self.surface)
        
        # Calculate visibility
        visibility = self.octopus.visibility(surf=self.surface)
        
        self.assertIsInstance(visibility, float)
        
        # except Exception as e:
        #     self.fail(f"Full simulation step failed: {e}")


if __name__ == '__main__':
    unittest.main()