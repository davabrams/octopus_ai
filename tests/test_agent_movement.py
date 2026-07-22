"""
Unit tests for agent movement: random walk, and reactive flee/hunt.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulator.octopus_generator import Octopus
from simulator.agent_generator import AgentGenerator
from simulator.simutil import MovementMode, Agent, AgentType
from helpers import make_config


def nearest_sucker_dist(octo, agent):
    xy = np.array([[s.x, s.y] for limb in octo.limbs for s in limb.suckers])
    return float(np.hypot(xy[:, 0] - agent.x, xy[:, 1] - agent.y).min())


class TestRandomAgentMovement(unittest.TestCase):
    def _params(self, **over):
        base = dict(
            x_len=24, y_len=24, octo_num_arms=4,
            limb_rows=6, limb_cols=2, rand_seed=1,
            agent_movement_mode=MovementMode.RANDOM,
        )
        base.update(over)  # callers may replace any of the above
        return make_config(**base)

    def test_random_walk_is_not_direction_biased(self):
        """Velocities must be symmetric about zero.

        They were uniform(0, max_velocity) - never negative - so every agent
        drifted monotonically +x/+y into the top-right corner rather than
        wandering. Regression guard for that.
        """
        p = self._params()
        ag = AgentGenerator(p)
        ag.agents = [Agent(x=12.0, y=12.0, agent_type=AgentType.PREY)
                     for _ in range(8)]
        for _ in range(60):
            ag.increment_all()
        xs = np.array([a.x for a in ag.agents])
        ys = np.array([a.y for a in ag.agents])
        # agents must not have all marched into the +x/+y corner
        self.assertFalse(np.all(xs > 20.0))
        self.assertFalse(np.all(ys > 20.0))
        # and they should straddle the start point rather than all exceed it
        self.assertTrue(np.any(xs < 12.0) or np.any(ys < 12.0))

    def test_random_agents_stay_on_grid(self):
        p = self._params()
        ag = AgentGenerator(p)
        ag.agents = [Agent(x=23.0, y=23.0, agent_type=AgentType.PREY),
                     Agent(x=0.0, y=0.0, agent_type=AgentType.THREAT)]
        for _ in range(50):
            ag.increment_all()
        for a in ag.agents:
            self.assertGreaterEqual(a.x, 0.0)
            self.assertGreaterEqual(a.y, 0.0)
            self.assertLessEqual(a.x, p.world.x_len - 1.0)
            self.assertLessEqual(a.y, p.world.y_len - 1.0)


class TestReactiveAgentMovement(unittest.TestCase):
    def _params(self, **over):
        base = dict(
            x_len=24, y_len=24, octo_num_arms=8,
            limb_rows=16, limb_cols=2, rand_seed=3,
            octo_movement_mode=MovementMode.SPRING_CHAIN,
            limb_movement_mode=MovementMode.SPRING_CHAIN,
            agent_movement_mode=MovementMode.SPRING_CHAIN,
        )
        base.update(over)  # callers may replace any of the above
        return make_config(**base)

    def test_prey_flees_the_octopus(self):
        p = self._params()
        octo = Octopus(p)
        ag = AgentGenerator(p)
        prey = Agent(x=octo.x + 3.0, y=octo.y, agent_type=AgentType.PREY)
        ag.agents = [prey]
        d0 = nearest_sucker_dist(octo, prey)
        for _ in range(5):
            # visibility=1.0: octopus fully visible so prey reacts
            ag.increment_all(octo, visibility=1.0)
        self.assertGreater(nearest_sucker_dist(octo, prey), d0)

    def test_threat_hunts_the_octopus(self):
        p = self._params()
        octo = Octopus(p)
        ag = AgentGenerator(p)
        threat = Agent(x=octo.x + 3.0, y=octo.y,
                       agent_type=AgentType.THREAT)
        ag.agents = [threat]
        d0 = nearest_sucker_dist(octo, threat)
        for _ in range(5):
            ag.increment_all(octo, visibility=1.0)
        self.assertLess(nearest_sucker_dist(octo, threat), d0)

    def test_both_spring_modes_are_reactive(self):
        """SPRING_CHAIN used to fall through increment_all's if/elif and
        silently leave agents motionless."""
        for mode in (MovementMode.LUMPED_SPRING, MovementMode.SPRING_CHAIN):
            with self.subTest(mode=mode):
                p = self._params(agent_movement_mode=mode,
                                 octo_movement_mode=mode,
                                 limb_movement_mode=mode)
                octo = Octopus(p)
                ag = AgentGenerator(p)
                threat = Agent(x=octo.x + 3.0, y=octo.y,
                               agent_type=AgentType.THREAT)
                ag.agents = [threat]
                d0 = nearest_sucker_dist(octo, threat)
                for _ in range(5):
                    ag.increment_all(octo, visibility=1.0)
                self.assertLess(nearest_sucker_dist(octo, threat), d0)

    def test_reactive_mode_requires_octopus(self):
        p = self._params()
        ag = AgentGenerator(p)
        ag.agents = [Agent(x=1.0, y=1.0, agent_type=AgentType.PREY)]
        with self.assertRaises(AssertionError):
            ag.increment_all()  # no octopus passed

    def test_out_of_range_agent_wanders_instead_of_reacting(self):
        p = self._params(agent_range_radius=2)
        octo = Octopus(p)
        ag = AgentGenerator(p)
        # far outside sensing range: should random-walk, not flee
        far = Agent(x=1.0, y=1.0, vx=0.1, vy=0.1,
                    agent_type=AgentType.PREY)
        ag.agents = [far]
        before = (far.x, far.y)
        ag.increment_all(octo)
        self.assertNotEqual((far.x, far.y), before)

    def test_reactive_agents_stay_on_grid(self):
        p = self._params()
        octo = Octopus(p)
        ag = AgentGenerator(p)
        ag.agents = [Agent(x=octo.x + 0.5, y=octo.y,
                           agent_type=AgentType.PREY)]
        for _ in range(80):
            ag.increment_all(octo, visibility=1.0)
        a = ag.agents[0]
        self.assertGreaterEqual(a.x, 0.0)
        self.assertGreaterEqual(a.y, 0.0)
        self.assertLessEqual(a.x, p.world.x_len - 1.0)
        self.assertLessEqual(a.y, p.world.y_len - 1.0)

    def test_octopus_can_still_catch_fleeing_prey(self):
        """The chase must remain winnable: body velocity (0.25) exceeds
        agent velocity (0.1), so the octopus gains on prey."""
        p = self._params()
        octo = Octopus(p)
        ag = AgentGenerator(p)
        ag.agents = [Agent(x=octo.x + 2.0, y=octo.y,
                           agent_type=AgentType.PREY)]
        caught = False
        for _ in range(150):
            ag.increment_all(octo, visibility=1.0)
            octo.move(ag)
            if ag.remove_captured_prey(octo):
                caught = True
                break
        self.assertTrue(caught)

    def test_invisible_octopus_makes_agents_wander(self):
        """At visibility=0 agents should random-walk even in reactive mode."""
        p = self._params()
        octo = Octopus(p)
        ag = AgentGenerator(p)
        threat = Agent(x=octo.x + 3.0, y=octo.y,
                       agent_type=AgentType.THREAT)
        ag.agents = [threat]
        # With visibility=0, threat should NOT reliably close distance.
        np.random.seed(42)
        distances = []
        for _ in range(20):
            ag.increment_all(octo, visibility=0.0)
            distances.append(nearest_sucker_dist(octo, threat))
        # Not all steps should decrease distance (random walk)
        decreases = sum(1 for i in range(1, len(distances))
                        if distances[i] < distances[i - 1])
        self.assertLess(decreases, len(distances) - 1)


class TestPursuitFleeAgentMovement(unittest.TestCase):
    """PURSUIT_FLEE: threats pursue and prey flee at full (deterministic)
    strength inside the sense window - but ONLY once the octopus is visible
    enough (visibility > visibility_threshold). A well-camouflaged octopus is
    left alone: camouflage is protective."""

    def _params(self, **over):
        base = dict(
            x_len=24, y_len=24, octo_num_arms=8,
            limb_rows=16, limb_cols=2, rand_seed=3,
            # Octopus mode is irrelevant here (agents only read sucker
            # positions); RANDOM keeps the test off the iLQR compile path.
            octo_movement_mode=MovementMode.RANDOM,
            limb_movement_mode=MovementMode.RANDOM,
            agent_movement_mode=MovementMode.PURSUIT_FLEE,
            agent_visibility_threshold=0.05,
        )
        base.update(over)
        return make_config(**base)

    def test_threat_pursues_a_visible_octopus(self):
        """Above the threshold the threat commits to pursuit and closes in."""
        p = self._params()
        octo = Octopus(p)
        ag = AgentGenerator(p)
        threat = Agent(x=octo.x + 3.0, y=octo.y, agent_type=AgentType.THREAT)
        ag.agents = [threat]
        d0 = nearest_sucker_dist(octo, threat)
        for _ in range(5):
            ag.increment_all(octo, visibility=1.0)  # clearly visible
        self.assertLess(nearest_sucker_dist(octo, threat), d0)

    def test_camouflaged_octopus_is_not_pursued(self):
        """Below the threshold (well hidden) threats wander, not pursue."""
        p = self._params()
        octo = Octopus(p)
        ag = AgentGenerator(p)
        threat = Agent(x=octo.x + 3.0, y=octo.y, agent_type=AgentType.THREAT)
        ag.agents = [threat]
        np.random.seed(7)
        dists = [nearest_sucker_dist(octo, threat)]
        for _ in range(20):
            ag.increment_all(octo, visibility=0.0)  # perfectly hidden
            dists.append(nearest_sucker_dist(octo, threat))
        decreases = sum(1 for i in range(1, len(dists))
                        if dists[i] < dists[i - 1])
        self.assertLess(decreases, len(dists) - 1)  # NOT steady pursuit

    def test_prey_flees_a_visible_octopus(self):
        p = self._params()
        octo = Octopus(p)
        ag = AgentGenerator(p)
        prey = Agent(x=octo.x + 3.0, y=octo.y, agent_type=AgentType.PREY)
        ag.agents = [prey]
        d0 = nearest_sucker_dist(octo, prey)
        for _ in range(5):
            ag.increment_all(octo, visibility=1.0)
        self.assertGreater(nearest_sucker_dist(octo, prey), d0)

    def test_behavior_codes(self):
        """Per-agent behavior policy is stamped each step: 1 pursuing (threat),
        2 fleeing (prey), 0 idle/wandering (out of range, or camouflaged)."""
        p = self._params()
        octo = Octopus(p)
        ag = AgentGenerator(p)
        threat = Agent(x=octo.x + 3.0, y=octo.y, agent_type=AgentType.THREAT)
        prey = Agent(x=octo.x, y=octo.y + 3.0, agent_type=AgentType.PREY)
        far = Agent(x=1.0, y=1.0, agent_type=AgentType.THREAT)  # a far corner
        ag.agents = [threat, prey, far]
        ag.increment_all(octo, visibility=1.0)   # visible
        self.assertEqual(threat.behavior, 1)     # pursuing
        self.assertEqual(prey.behavior, 2)       # fleeing
        self.assertEqual(far.behavior, 0)        # out of range -> wandering
        # Below the notice threshold (camouflaged): the in-range threat wanders.
        ag.increment_all(octo, visibility=0.0)
        self.assertEqual(threat.behavior, 0)

    def test_threshold_is_the_notice_boundary(self):
        """Visibility just above the threshold pursues (the exact gate)."""
        p = self._params(agent_visibility_threshold=0.1)
        octo = Octopus(p)
        ag = AgentGenerator(p)
        threat = Agent(x=octo.x + 3.0, y=octo.y, agent_type=AgentType.THREAT)
        ag.agents = [threat]
        d0 = nearest_sucker_dist(octo, threat)
        for _ in range(5):
            ag.increment_all(octo, visibility=0.2)  # above 0.1 -> pursues
        self.assertLess(nearest_sucker_dist(octo, threat), d0)

    def test_only_acts_in_sense_window(self):
        """A VISIBLE octopus beyond the sense window still isn't pursued."""
        p = self._params(agent_range_radius=2)
        octo = Octopus(p)
        ag = AgentGenerator(p)
        far = Agent(x=octo.x + 10.0, y=octo.y, agent_type=AgentType.THREAT)
        ag.agents = [far]
        np.random.seed(11)
        dists = [nearest_sucker_dist(octo, far)]
        for _ in range(20):
            ag.increment_all(octo, visibility=1.0)  # visible, but out of range
            dists.append(nearest_sucker_dist(octo, far))
        decreases = sum(1 for i in range(1, len(dists))
                        if dists[i] < dists[i - 1])
        self.assertLess(decreases, len(dists) - 1)  # not steady pursuit

    def test_pursuit_flee_stays_on_grid(self):
        p = self._params()
        octo = Octopus(p)
        ag = AgentGenerator(p)
        # Prey pinned in a corner, fleeing a VISIBLE octopus - must clamp to the
        # grid, not walk off.
        ag.agents = [Agent(x=1.0, y=1.0, agent_type=AgentType.PREY)]
        for _ in range(80):
            ag.increment_all(octo, visibility=1.0)
        a = ag.agents[0]
        self.assertGreaterEqual(a.x, 0.0)
        self.assertGreaterEqual(a.y, 0.0)
        self.assertLessEqual(a.x, p.world.x_len - 1.0)
        self.assertLessEqual(a.y, p.world.y_len - 1.0)


if __name__ == '__main__':
    unittest.main()
