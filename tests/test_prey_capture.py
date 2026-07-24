"""
Unit tests for prey capture (prey touched by a sucker disappears).
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


class TestPreyCapture(unittest.TestCase):
    def _params(self, **over):
        base = dict(
            x_len=24, y_len=24, octo_num_arms=4,
            limb_rows=8, limb_cols=2, rand_seed=3,
            octo_movement_mode=MovementMode.RANDOM,
            limb_movement_mode=MovementMode.RANDOM,
            agent_movement_mode=MovementMode.RANDOM,
        )
        base.update(over)  # callers may replace any of the above
        return make_config(**base)

    def _octo_and_gen(self, **over):
        p = self._params(**over)
        return Octopus(p), AgentGenerator(p)

    def test_prey_on_a_sucker_is_captured(self):
        # respawn off so this isolates CAPTURE (the caught prey leaves the list);
        # the respawn/replacement behaviour is covered separately below.
        octo, ag = self._octo_and_gen(agent_respawn_captured_prey=False)
        s = octo.limbs[0].suckers[0]
        ag.agents = [Agent(x=s.x, y=s.y, agent_type=AgentType.PREY)]
        n = ag.remove_captured_prey(octo)
        self.assertEqual(n, 1)
        self.assertEqual(len(ag.agents), 0)
        self.assertEqual(ag.prey_captured, 1)

    def test_distant_prey_is_not_captured(self):
        octo, ag = self._octo_and_gen()
        far = Agent(x=0.5, y=0.5, agent_type=AgentType.PREY)
        ag.agents = [far]
        n = ag.remove_captured_prey(octo)
        self.assertEqual(n, 0)
        self.assertEqual(len(ag.agents), 1)
        self.assertEqual(ag.prey_captured, 0)

    def test_threat_is_never_captured(self):
        octo, ag = self._octo_and_gen()
        s = octo.limbs[0].suckers[0]
        # sitting exactly on a sucker, yet must survive
        ag.agents = [Agent(x=s.x, y=s.y, agent_type=AgentType.THREAT)]
        n = ag.remove_captured_prey(octo)
        self.assertEqual(n, 0)
        self.assertEqual(len(ag.agents), 1)

    def test_only_touching_prey_removed_others_kept(self):
        octo, ag = self._octo_and_gen()
        s = octo.limbs[0].suckers[0]
        touching = Agent(x=s.x, y=s.y, agent_type=AgentType.PREY)
        far_prey = Agent(x=0.5, y=0.5, agent_type=AgentType.PREY)
        threat = Agent(x=s.x, y=s.y, agent_type=AgentType.THREAT)
        ag.agents = [touching, far_prey, threat]
        n = ag.remove_captured_prey(octo)
        self.assertEqual(n, 1)
        self.assertIn(far_prey, ag.agents)
        self.assertIn(threat, ag.agents)
        self.assertNotIn(touching, ag.agents)

    def test_capture_radius_boundary(self):
        """Capture uses the MIN distance over every sucker, so the boundary
        must be probed beyond the octopus's outermost sucker - placing prey
        0.45 from one sucker says nothing, since a neighbour 0.1 away may
        still be within range."""
        octo, ag = self._octo_and_gen(agent_prey_capture_radius=0.3)
        sucker_xy = np.array(
            [[s.x, s.y] for limb in octo.limbs for s in limb.suckers])
        # the +x-most sucker: any point beyond it in +x is at least that
        # x-gap away from EVERY sucker (all others have x <= this one)
        ix = int(np.argmax(sucker_xy[:, 0]))
        far_x, far_y = sucker_xy[ix]

        outside = Agent(x=far_x + 0.45, y=far_y, agent_type=AgentType.PREY)
        ag.agents = [outside]
        self.assertEqual(ag.remove_captured_prey(octo), 0)

        inside = Agent(x=far_x + 0.2, y=far_y, agent_type=AgentType.PREY)
        ag.agents = [inside]
        self.assertEqual(ag.remove_captured_prey(octo), 1)

    def test_zero_radius_disables_capture(self):
        octo, ag = self._octo_and_gen(agent_prey_capture_radius=0.0)
        s = octo.limbs[0].suckers[0]
        ag.agents = [Agent(x=s.x, y=s.y, agent_type=AgentType.PREY)]
        self.assertEqual(ag.remove_captured_prey(octo), 0)
        self.assertEqual(len(ag.agents), 1)

    def test_respawn_keeps_population(self):
        octo, ag = self._octo_and_gen(agent_respawn_captured_prey=True)
        s = octo.limbs[0].suckers[0]
        ag.agents = [Agent(x=s.x, y=s.y, agent_type=AgentType.PREY)]
        n = ag.remove_captured_prey(octo)
        self.assertEqual(n, 1)
        # one fresh prey generated to replace the captured one
        self.assertEqual(len(ag.agents), 1)
        self.assertEqual(ag.agents[0].agent_type, AgentType.PREY)

    def test_respawn_is_on_by_default(self):
        """Respawn is the default: a caught prey is replaced (population constant)."""
        octo, ag = self._octo_and_gen()
        s = octo.limbs[0].suckers[0]
        ag.agents = [Agent(x=s.x, y=s.y, agent_type=AgentType.PREY)]
        self.assertEqual(ag.remove_captured_prey(octo), 1)
        self.assertEqual(len(ag.agents), 1)          # replaced, not removed
        self.assertEqual(ag.agents[0].agent_type, AgentType.PREY)

    def test_no_respawn_when_disabled(self):
        octo, ag = self._octo_and_gen(agent_respawn_captured_prey=False)
        s = octo.limbs[0].suckers[0]
        ag.agents = [Agent(x=s.x, y=s.y, agent_type=AgentType.PREY)]
        ag.remove_captured_prey(octo)
        self.assertEqual(len(ag.agents), 0)          # removed, not replaced

    def test_empty_agents_is_safe(self):
        octo, ag = self._octo_and_gen()
        ag.agents = []
        self.assertEqual(ag.remove_captured_prey(octo), 0)

    def test_tally_accumulates_across_calls(self):
        octo, ag = self._octo_and_gen()
        s = octo.limbs[0].suckers[0]
        for _ in range(3):
            ag.agents = [Agent(x=s.x, y=s.y, agent_type=AgentType.PREY)]
            ag.remove_captured_prey(octo)
        self.assertEqual(ag.prey_captured, 3)


if __name__ == '__main__':
    unittest.main()
