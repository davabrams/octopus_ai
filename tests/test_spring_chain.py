"""
Unit tests for the SPRING_CHAIN movement mode and its linear solver.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulator.octopus_generator import Octopus, Limb
from simulator.simutil import MovementMode, Agent, AgentType
from simulator.spring_chain import build_K, solve_chain, base_reaction
from OctoConfig import GameParameters


class TestSpringChainSolver(unittest.TestCase):
    """The standalone linear-algebra layer."""

    def test_K_is_symmetric_positive_definite(self):
        n = 8
        K = build_K(n, k_spring=1.0, k_move=2.0,
                    agent_k_weighted=np.zeros(n))
        self.assertTrue(np.allclose(K, K.T))
        self.assertGreater(np.linalg.eigvalsh(K).min(), 0.0)

    def test_solve_pulls_tip_toward_prey(self):
        n = 6
        base = np.array([0.0, 0.0])
        prev = np.array([[i + 1.0, 0.0] for i in range(n)])
        targets = np.full((n, 2), np.nan)
        prey = np.array([9.0, 3.0])
        targets[-1] = prey
        threat = np.zeros((n, 2))
        out = solve_chain(base, prev, targets, threat,
                          k_spring=1.0, k_agent=2.0, k_move=1.0)
        d_before = np.hypot(*(prey - prev[-1]))
        d_after = np.hypot(*(prey - out[-1]))
        self.assertLess(d_after, d_before)

    def test_higher_move_cost_reduces_movement(self):
        n = 6
        base = np.array([0.0, 0.0])
        prev = np.array([[i + 1.0, 0.0] for i in range(n)])
        targets = np.full((n, 2), np.nan)
        targets[-1] = np.array([9.0, 3.0])
        threat = np.zeros((n, 2))
        lo = solve_chain(base, prev, targets, threat, 1.0, 2.0, k_move=0.5)
        hi = solve_chain(base, prev, targets, threat, 1.0, 2.0, k_move=10.0)
        self.assertLess(np.linalg.norm(hi - prev),
                        np.linalg.norm(lo - prev))

    def test_tip_moves_more_than_base_node(self):
        n = 6
        base = np.array([0.0, 0.0])
        prev = np.array([[i + 1.0, 0.0] for i in range(n)])
        targets = np.full((n, 2), np.nan)
        targets[-1] = np.array([9.0, 3.0])
        threat = np.zeros((n, 2))
        out = solve_chain(base, prev, targets, threat, 1.0, 2.0, 1.0)
        move = np.linalg.norm(out - prev, axis=1)
        self.assertGreater(move[-1], move[0])

    def test_base_reaction_points_toward_node1(self):
        br = base_reaction(np.array([0.0, 0.0]), np.array([1.0, 0.5]),
                           k_spring=2.0)
        # spring base->node1: node1 is +x/+y of base, so reaction is +,+
        self.assertGreater(br[0], 0.0)
        self.assertGreater(br[1], 0.0)


class TestSpringChainLimb(unittest.TestCase):
    """The mode wired into Limb / Octopus."""

    def _params(self):
        p = GameParameters.copy()
        p.update({
            'x_len': 24, 'y_len': 24, 'limb_rows': 10, 'limb_cols': 2,
            'octo_num_arms': 4, 'rand_seed': 3,
            'octo_movement_mode': MovementMode.SPRING_CHAIN,
            'limb_movement_mode': MovementMode.SPRING_CHAIN,
            'agent_movement_mode': MovementMode.RANDOM,
        })
        return p

    def test_base_stays_pinned_to_body(self):
        p = self._params()
        limb = Limb(x_octo=12.0, y_octo=12.0, init_angle=0.0, params=p)
        prey = Agent(x=16.0, y=14.0, agent_type=AgentType.PREY)
        limb.move(13.0, 11.5, agents=[prey])
        self.assertAlmostEqual(limb.center_line[0].x, 13.0, places=6)
        self.assertAlmostEqual(limb.center_line[0].y, 11.5, places=6)

    def test_tip_reaches_toward_prey(self):
        p = self._params()
        limb = Limb(x_octo=12.0, y_octo=12.0, init_angle=0.0, params=p)
        prey = Agent(x=15.0, y=14.0, agent_type=AgentType.PREY)
        tip = limb.center_line[-1]
        d0 = np.hypot(prey.x - tip.x, prey.y - tip.y)
        for _ in range(20):
            limb.move(12.0, 12.0, agents=[prey])
        d1 = np.hypot(prey.x - tip.x, prey.y - tip.y)
        self.assertLess(d1, d0)

    def test_no_nan_over_long_run(self):
        p = self._params()
        limb = Limb(x_octo=12.0, y_octo=12.0, init_angle=0.0, params=p)
        prey = Agent(x=15.0, y=15.0, agent_type=AgentType.PREY)
        for _ in range(50):
            limb.move(12.0, 12.0, agents=[prey])
        for pt in limb.center_line:
            self.assertTrue(np.isfinite(pt.x) and np.isfinite(pt.y))

    def test_higher_move_k_less_movement_in_sim(self):
        p_lo = self._params(); p_lo['octo_chain_move_k'] = 0.5
        p_hi = self._params(); p_hi['octo_chain_move_k'] = 20.0
        prey = Agent(x=16.0, y=15.0, agent_type=AgentType.PREY)

        def total_move(params):
            limb = Limb(x_octo=12.0, y_octo=12.0, init_angle=0.0,
                        params=params)
            before = np.array([[p.x, p.y] for p in limb.center_line])
            limb.move(12.0, 12.0, agents=[prey])
            after = np.array([[p.x, p.y] for p in limb.center_line])
            return np.linalg.norm(after - before)

        self.assertLess(total_move(p_hi), total_move(p_lo))

    def test_octopus_body_drifts_toward_prey(self):
        p = self._params()
        octo = Octopus(params=p)

        class _AG:
            pass
        ag = _AG()
        ag.agents = [Agent(x=octo.x + 4, y=octo.y + 2,
                           agent_type=AgentType.PREY)]
        x0, y0 = octo.x, octo.y
        for _ in range(20):
            octo.move(ag)
        self.assertGreater(octo.x, x0)
        self.assertGreater(octo.y, y0)


if __name__ == '__main__':
    unittest.main()
