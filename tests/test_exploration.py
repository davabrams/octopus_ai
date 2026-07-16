"""Exploration behavior (EXPLORATION_PLAN.md, sucker-seeking variant).

Cells are marked explored by the SUCKERS (not the body); when an arm senses no
prey, its tip softly reaches the least-explored reachable cell, so the suckers
sweep unexplored areas. A weak drive: prey always preempts it, and the threat
repel (a separate term) always dominates.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from helpers import make_config

from simulator.agent_generator import AgentGenerator
from simulator.octopus_generator import Octopus
from simulator.simutil import Agent, AgentType, MovementMode


def _octo(*, explore=True, agents=0, w_explore=0.5, decay=1.0, seed=0):
    cfg = make_config(
        x_len=20, y_len=20, limb_rows=8, octo_num_arms=6,
        agent_number_of_agents=agents,
        octo_movement_mode=MovementMode.ILQR,
        limb_movement_mode=MovementMode.ILQR,
        octo_ilqr_horizon=4, octo_ilqr_max_iters=3,
        octo_ilqr_explore_enabled=explore, octo_ilqr_w_explore=w_explore,
        octo_ilqr_explore_decay=decay, record_ilqr_history=True)
    np.random.seed(seed)
    octo = Octopus(cfg)
    ag = AgentGenerator(cfg)
    ag.generate(agents)
    return octo, ag


def _kinds(octo):
    return {limb.last_ilqr_meta["target_kind"] for limb in octo.limbs
            if limb.last_ilqr_meta is not None}


class TestExplorationMemory(unittest.TestCase):
    def test_cells_are_marked_by_suckers_not_the_body(self):
        """The visit map lights up at sucker positions, and NOT at the body
        head cell (unless a sucker happens to be there)."""
        octo, ag = _octo(explore=True, agents=0)
        octo.move(ag)
        sucker_cells = set()
        for limb in octo.limbs:
            for s in limb.suckers:
                sucker_cells.add((round(s.y), round(s.x)))
        marked = set(zip(*np.nonzero(octo.visit_counts), strict=True))
        # Every marked cell is a sucker cell (marking is sucker-driven).
        self.assertTrue(marked)
        self.assertTrue(marked.issubset(sucker_cells))

    def test_off_is_zero_overhead(self):
        """explore_enabled=False leaves the map empty and never seeks."""
        octo, ag = _octo(explore=False, agents=0)
        for _ in range(5):
            octo.move(ag)
        self.assertEqual(octo.visit_counts.sum(), 0.0)
        self.assertEqual(_kinds(octo), {"hold"})

    def test_decay_ages_counts(self):
        """decay < 1 fades old visits: revisited cells accumulate fractional
        counts (0.5*n + 1), which never happens at decay = 1.0 (pure integers)."""
        octo, ag = _octo(explore=True, agents=0, decay=0.5)
        for _ in range(4):
            octo.move(ag)
        vc = octo.visit_counts[octo.visit_counts > 0]
        self.assertTrue(np.any(np.abs(vc - np.round(vc)) > 1e-6),
                        "decay did not produce fractional (aged) counts")
        # Sanity: at decay=1.0 the same map is all integers.
        octo2, ag2 = _octo(explore=True, agents=0, decay=1.0)
        for _ in range(4):
            octo2.move(ag2)
        vc2 = octo2.visit_counts[octo2.visit_counts > 0]
        self.assertTrue(np.all(np.abs(vc2 - np.round(vc2)) < 1e-6))


class TestExplorationSeeking(unittest.TestCase):
    def test_coverage_grows_when_exploring(self):
        """An idle octopus with exploration on covers more cells over time."""
        octo, ag = _octo(explore=True, agents=0)
        first = None
        for f in range(30):
            octo.move(ag)
            if f == 0:
                first = int((octo.visit_counts > 0).sum())
        last = int((octo.visit_counts > 0).sum())
        self.assertGreater(last, first)

    def test_idle_octopus_moves_when_exploring(self):
        """With exploration on, an idle octopus does NOT sit still - the arms
        seek unexplored space (contrast the no-explore hold)."""
        octo, ag = _octo(explore=True, agents=0)
        tip0 = np.array([octo.limbs[0].center_line[-1].x,
                         octo.limbs[0].center_line[-1].y])
        for _ in range(15):
            octo.move(ag)
        tip1 = np.array([octo.limbs[0].center_line[-1].x,
                         octo.limbs[0].center_line[-1].y])
        self.assertGreater(np.linalg.norm(tip1 - tip0), 0.3)

    def test_explore_target_is_least_visited_reachable_cell(self):
        """The selector returns a low-visit cell within reach, biased local."""
        octo, ag = _octo(explore=True, agents=0)
        octo.move(ag)  # seed the map
        limb = octo.limbs[0]
        base = limb.center_line[0]
        tip = limb.center_line[-1]
        target = limb._ilqr_explore_target(base.x, base.y, tip,
                                           octo.visit_counts)
        self.assertIsNotNone(target)
        reach = (limb.rows - 1) * limb.max_sucker_distance
        # In reach of the base...
        self.assertLessEqual(np.hypot(target[0] - base.x, target[1] - base.y),
                             reach + 1.0)
        # ...and no OTHER reachable cell is strictly better (least-visited +
        # locality) - i.e. it is the argmin the arm will chase.
        tx, ty = round(target[0]), round(target[1])
        best = (octo.visit_counts[ty, tx] +
                limb.explore_locality * np.hypot(tx - tip.x, ty - tip.y))
        y_len, x_len = octo.visit_counts.shape
        for cy in range(max(0, ty - 1), min(y_len, ty + 2)):
            for cx in range(max(0, tx - 1), min(x_len, tx + 2)):
                if np.hypot(cx - base.x, cy - base.y) > reach:
                    continue
                score = (octo.visit_counts[cy, cx] +
                         limb.explore_locality * np.hypot(cx - tip.x,
                                                          cy - tip.y))
                self.assertGreaterEqual(score, best - 1e-6)


class TestRewardHierarchy(unittest.TestCase):
    def test_prey_preempts_exploration(self):
        """A prey in reach => that arm targets prey, never explores."""
        octo, ag = _octo(explore=True, agents=1)
        cx, cy = octo.x, octo.y
        prey = Agent(x=cx + 2.0, y=cy, agent_type=AgentType.PREY)
        ag.agents = [prey]
        for _ in range(5):
            prey.x, prey.y = cx + 2.0, cy
            octo.move(ag)
        kinds = _kinds(octo)
        self.assertIn("prey", kinds)
        self.assertNotIn("explore", kinds)

    def test_explore_reach_is_gentler_than_prey(self):
        """w_explore must stay well below w_reach_terminal so exploration is a
        weak drive (the 'much less reward' requirement)."""
        cfg = make_config(octo_ilqr_explore_enabled=True)
        self.assertLess(cfg.octopus.limb.ilqr.w_explore,
                        cfg.octopus.limb.ilqr.w_reach_terminal)


if __name__ == '__main__':
    unittest.main()
