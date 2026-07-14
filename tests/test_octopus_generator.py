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
from simulator.simutil import (
    State, Color, MLMode, InferenceLocation, MovementMode,
    Agent, AgentType, agent_influence_vector,
)
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
        self.assertGreater(result, 0.5)


class TestLimb(unittest.TestCase):
    """Test the Limb class"""

    def setUp(self):
        """Set up test fixtures"""
        self.params = GameParameters.copy()
        self.params['limb_rows'] = 4
        self.params['limb_cols'] = 2
        self.params['x_len'] = 10
        self.params['y_len'] = 10

        # Create test limb
        self.limb = Limb(x_octo=5.0, y_octo=5.0, init_angle=0, params=self.params)

    def test_limb_initialization(self):
        """Test limb is properly initialized"""
        expected_suckers = self.params['limb_rows'] * self.params['limb_cols']
        self.assertEqual(len(self.limb.suckers), expected_suckers)

        for sucker in self.limb.suckers:
            self.assertIsInstance(sucker, Sucker)

    def test_gen_centerline(self):
        """Test centerline generation"""
        self.limb._gen_centerline(0, 0, 0)

        self.assertEqual(len(self.limb.center_line), self.params['limb_rows'])

        for i in range(len(self.limb.center_line) - 1):
            dist_from_root_i = np.sqrt(self.limb.center_line[i].x**2 + self.limb.center_line[i].y**2)
            dist_from_root_j = np.sqrt(self.limb.center_line[i+1].x**2 + self.limb.center_line[i+1].y**2)
            self.assertLessEqual(dist_from_root_i, dist_from_root_j)

    def test_refresh_sucker_locations(self):
        """Test sucker location refresh"""
        old_positions = [(s.x, s.y) for s in self.limb.suckers]

        # Change limb root angle and refresh
        self.limb.center_line[0].t = np.pi / 4
        self.limb._refresh_sucker_locations()

        new_positions = [(s.x, s.y) for s in self.limb.suckers]

        # Positions should have changed
        self.assertNotEqual(old_positions, new_positions)

    def test_find_adjacents(self):
        """Test adjacent sucker finding"""
        if self.limb.suckers:
            target_sucker = self.limb.suckers[0]
            adjacents_with_dist = self.limb.find_adjacents(target_sucker, radius=2.0)

            self.assertIsInstance(adjacents_with_dist, list)

            for adj, dist in adjacents_with_dist:
                distance = target_sucker.distance_to(adj)
                self.assertLessEqual(distance, 2.0)

    def test_move_random_mode(self):
        """Test limb movement in random mode"""
        self.params['limb_movement_mode'] = MovementMode.RANDOM

        # move takes (x_octo, y_octo)
        self.limb.move(1.0, 1.0)

        # center_line[0].t should be a float angle
        self.assertIsInstance(self.limb.center_line[0].t, float)


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

        # Octopus.__init__ takes (params) only
        self.octopus = Octopus(params=self.params)

    def test_octopus_initialization(self):
        """Test octopus is properly initialized"""
        self.assertEqual(len(self.octopus.limbs), self.params['octo_num_arms'])

        for limb in self.octopus.limbs:
            self.assertIsInstance(limb, Limb)

    def test_find_color_parallel(self):
        """Test parallel color finding across limbs"""
        colors = self.octopus.find_color(self.mock_surface)

        self.assertIsInstance(colors, list)
        self.assertEqual(len(colors), len(self.octopus.limbs))

    def test_set_color(self):
        """Test color setting across all suckers"""
        # set_color takes (surf, inference_mode, model)
        self.octopus.set_color(self.mock_surface)

        for limb in self.octopus.limbs:
            for sucker in limb.suckers:
                self.assertIsInstance(sucker.c.r, float)

    def test_visibility_calculation(self):
        """Test MSE visibility calculation"""
        for limb in self.octopus.limbs:
            for sucker in limb.suckers:
                sucker.c.r = 0.5

        visibility = self.octopus.visibility(self.mock_surface)

        self.assertIsInstance(visibility, float)
        self.assertGreaterEqual(visibility, 0.0)

    def test_move_body(self):
        """Test octopus body movement"""
        original_pos = (self.octopus.x, self.octopus.y)

        # move takes an Agent or None
        self.octopus.move()

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

        self.surface = RandomSurface(self.params)
        self.octopus = Octopus(params=self.params)

    def test_full_simulation_step(self):
        """Test a complete simulation step"""
        self.octopus.move()

        self.octopus.set_color(surf=self.surface)

        visibility = self.octopus.visibility(surf=self.surface)
        self.assertIsInstance(visibility, float)



class TestAttractRepelLocomotion(unittest.TestCase):
    """LUMPED_SPRING movement: reaching, anchoring, bend/length limits."""

    def _params(self):
        p = GameParameters.copy()
        p['x_len'] = 30
        p['y_len'] = 30
        p['limb_rows'] = 6
        p['limb_cols'] = 2
        p['octo_num_arms'] = 4
        p['limb_movement_mode'] = MovementMode.LUMPED_SPRING
        p['octo_movement_mode'] = MovementMode.LUMPED_SPRING
        p['agent_movement_mode'] = MovementMode.LUMPED_SPRING
        return p

    # ---- the shared influence primitive ----
    def test_influence_zero_when_no_agents_in_range(self):
        v = agent_influence_vector(5.0, 5.0, [], radius=5)
        self.assertAlmostEqual(v[0], 0.0)
        self.assertAlmostEqual(v[1], 0.0)

    def test_influence_points_toward_prey(self):
        prey = Agent(x=10.0, y=5.0, agent_type=AgentType.PREY)
        v = agent_influence_vector(5.0, 5.0, [prey], radius=10)
        self.assertGreater(v[0], 0.0)          # prey is to the +x side
        self.assertAlmostEqual(v[1], 0.0, places=6)

    def test_influence_points_away_from_threat(self):
        threat = Agent(x=10.0, y=5.0, agent_type=AgentType.THREAT)
        v = agent_influence_vector(5.0, 5.0, [threat], radius=10)
        self.assertLess(v[0], 0.0)             # pushed away from +x
        self.assertAlmostEqual(v[1], 0.0, places=6)

    def test_influence_ignores_out_of_range_agents(self):
        far_prey = Agent(x=100.0, y=100.0, agent_type=AgentType.PREY)
        v = agent_influence_vector(5.0, 5.0, [far_prey], radius=5)
        self.assertAlmostEqual(v[0], 0.0)
        self.assertAlmostEqual(v[1], 0.0)

    # ---- limb reaching ----
    def test_limb_base_anchored_to_body(self):
        p = self._params()
        limb = Limb(x_octo=15.0, y_octo=15.0, init_angle=0.0, params=p)
        prey = Agent(x=20.0, y=15.0, agent_type=AgentType.PREY)
        # body moves to a new spot; base must follow it exactly
        limb.move(16.0, 15.5, agents=[prey])
        self.assertAlmostEqual(limb.center_line[0].x, 16.0, places=6)
        self.assertAlmostEqual(limb.center_line[0].y, 15.5, places=6)

    def test_limb_tip_moves_toward_prey(self):
        p = self._params()
        limb = Limb(x_octo=15.0, y_octo=15.0, init_angle=0.0, params=p)
        # prey within the arm's sensing radius but off the arm's initial
        # (+x) heading, so a genuine turn toward it is required
        prey = Agent(x=17.0, y=18.0, agent_type=AgentType.PREY)
        # snapshot values (center_line[-1] is a live object that move mutates)
        tip = limb.center_line[-1]
        tip0_x, tip0_y = tip.x, tip.y
        d0 = np.hypot(prey.x - tip0_x, prey.y - tip0_y)
        for _ in range(8):
            limb.move(15.0, 15.0, agents=[prey])
        tip1_x, tip1_y = tip.x, tip.y
        d1 = np.hypot(prey.x - tip1_x, prey.y - tip1_y)
        # the tip should end up closer to the prey than it began
        self.assertLess(d1, d0)
        # and the arm tip should have risen toward the prey (+y)
        self.assertGreater(tip1_y, tip0_y)

    def test_limb_tip_moves_away_from_threat(self):
        p = self._params()
        # arm initially points +x (toward the threat); threat within range
        limb = Limb(x_octo=15.0, y_octo=15.0, init_angle=0.0, params=p)
        threat = Agent(x=18.0, y=15.0, agent_type=AgentType.THREAT)
        tip = limb.center_line[-1]
        t0_x, t0_y = tip.x, tip.y
        d0 = np.hypot(threat.x - t0_x, threat.y - t0_y)
        for _ in range(5):
            limb.move(15.0, 15.0, agents=[threat])
        d1 = np.hypot(threat.x - tip.x, threat.y - tip.y)
        self.assertGreater(d1, d0)

    def test_limb_segments_within_sucker_distance_bounds(self):
        p = self._params()
        limb = Limb(x_octo=15.0, y_octo=15.0, init_angle=0.0, params=p)
        prey = Agent(x=17.0, y=17.0, agent_type=AgentType.PREY)
        for _ in range(6):
            limb.move(15.0, 15.0, agents=[prey])
        cl = limb.center_line
        for i in range(1, len(cl)):
            d = np.hypot(cl[i].x - cl[i - 1].x, cl[i].y - cl[i - 1].y)
            # spring spacing is clamped to [min, max]; grid clamp can only
            # shorten a segment, so max is the hard upper bound
            self.assertLessEqual(d, limb.max_sucker_distance + 1e-6)

    def test_arm_extends_toward_prey_then_retracts(self):
        p = self._params()
        limb = Limb(x_octo=15.0, y_octo=15.0, init_angle=0.0, params=p)

        def arm_len(l):
            cl = l.center_line
            return sum(np.hypot(cl[i].x - cl[i-1].x, cl[i].y - cl[i-1].y)
                       for i in range(1, len(cl)))

        rest = limb._rest_length()
        # prey in range: arm should stretch beyond rest length
        prey = Agent(x=18.0, y=15.0, agent_type=AgentType.PREY)
        for _ in range(10):
            limb.move(15.0, 15.0, agents=[prey])
        stretched = arm_len(limb)
        self.assertGreater(stretched, rest + 1e-6)

        # prey gone: spring reels the arm back toward rest length
        for _ in range(20):
            limb.move(15.0, 15.0, agents=[])
        retracted = arm_len(limb)
        self.assertLess(retracted, stretched)

    def test_tension_vector_zero_at_rest_nonzero_when_stretched(self):
        p = self._params()
        limb = Limb(x_octo=15.0, y_octo=15.0, init_angle=0.0, params=p)
        # fresh arm is near rest -> little/no tension
        # (build it to exactly rest by relaxing with no agents)
        for _ in range(20):
            limb.move(15.0, 15.0, agents=[])
        t_rest = np.hypot(*limb.tension_vector())
        self.assertLess(t_rest, 1e-3)
        # stretch it toward prey -> tension should appear, pointing at tip
        prey = Agent(x=18.0, y=15.0, agent_type=AgentType.PREY)
        for _ in range(10):
            limb.move(15.0, 15.0, agents=[prey])
        tv = limb.tension_vector()
        self.assertGreater(np.hypot(*tv), 1e-3)
        self.assertGreater(tv[0], 0.0)  # tip is +x of base, so is tension

    def test_predator_compresses_arm_below_rest(self):
        p = self._params()
        limb = Limb(x_octo=15.0, y_octo=15.0, init_angle=np.pi, params=p)
        for _ in range(25):
            limb.move(15.0, 15.0, agents=[])

        def arm_len(l):
            cl = l.center_line
            return sum(np.hypot(cl[i].x - cl[i-1].x, cl[i].y - cl[i-1].y)
                       for i in range(1, len(cl)))
        rest = limb._rest_length()
        self.assertAlmostEqual(arm_len(limb), rest, delta=0.15)
        tip = limb.center_line[-1]
        foe = Agent(x=tip.x - 1.5, y=15.0, agent_type=AgentType.THREAT)
        for _ in range(10):
            limb.move(15.0, 15.0, agents=[foe])
        self.assertLess(arm_len(limb), rest)

    def test_predator_tension_flees_body_away(self):
        p = self._params()
        limb = Limb(x_octo=15.0, y_octo=15.0, init_angle=np.pi, params=p)
        for _ in range(25):
            limb.move(15.0, 15.0, agents=[])
        tip = limb.center_line[-1]
        foe = Agent(x=tip.x - 1.5, y=15.0, agent_type=AgentType.THREAT)
        for _ in range(10):
            limb.move(15.0, 15.0, agents=[foe])
        tv = limb.tension_vector()
        self.assertGreater(tv[0], 0.0)

    def test_octopus_body_flees_predator(self):
        p = self._params()
        octo = Octopus(params=p)

        class _AG:
            pass
        ag = _AG()
        ag.agents = [Agent(x=octo.x + 2.0, y=octo.y,
                           agent_type=AgentType.THREAT)]
        x0 = octo.x
        for _ in range(12):
            octo.move(ag)
        self.assertLess(octo.x, x0)

    def test_limb_bend_within_reach_theta(self):
        p = self._params()
        limb = Limb(x_octo=15.0, y_octo=15.0, init_angle=0.0, params=p)
        prey = Agent(x=15.0, y=28.0, agent_type=AgentType.PREY)  # 90deg away
        cap = p['octo_max_arm_reach_theta']
        for _ in range(6):
            limb.move(15.0, 15.0, agents=[prey])
        cl = limb.center_line
        for i in range(1, len(cl)):
            dtheta = (cl[i].t - cl[i - 1].t + np.pi) % (2 * np.pi) - np.pi
            self.assertLessEqual(abs(dtheta), cap + 1e-6)

    # ---- octopus body ----
    def test_body_drifts_toward_prey(self):
        p = self._params()
        octo = Octopus(params=p)

        class _AG:
            pass
        ag = _AG()
        # prey to the +x/+y side, within arm sensing range
        ag.agents = [Agent(x=octo.x + 2.5, y=octo.y + 2.5,
                           agent_type=AgentType.PREY)]
        x0, y0 = octo.x, octo.y
        # pure-tension body: arms strain first, body follows over steps
        for _ in range(12):
            octo.move(ag)
        moved = np.hypot(octo.x - x0, octo.y - y0)
        self.assertGreater(moved, 0.0)
        # net drift should be toward the prey quadrant
        self.assertGreater(octo.x, x0)
        self.assertGreater(octo.y, y0)

    def test_body_static_without_agents_in_range(self):
        p = self._params()
        octo = Octopus(params=p)

        class _AG:
            pass
        ag = _AG()
        ag.agents = [Agent(x=octo.x + 100, y=octo.y + 100,
                           agent_type=AgentType.PREY)]  # far away
        x0, y0 = octo.x, octo.y
        octo.move(ag)
        self.assertAlmostEqual(octo.x, x0, places=6)
        self.assertAlmostEqual(octo.y, y0, places=6)

if __name__ == '__main__':
    unittest.main()
