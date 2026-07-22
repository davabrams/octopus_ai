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


def _attract_sw(octo):
    """All free nodes' attract sqrt-weights across the octopus (per-node)."""
    return np.concatenate([limb.last_ilqr_meta["attract_sw"]
                           for limb in octo.limbs
                           if limb.last_ilqr_meta is not None])


def _repel_sw(octo):
    return np.concatenate([limb.last_ilqr_meta["repel_sw"]
                           for limb in octo.limbs
                           if limb.last_ilqr_meta is not None])


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
        marked = set(zip(*np.nonzero(octo.visit_recency), strict=True))
        # Every marked cell is a sucker cell (marking is sucker-driven).
        self.assertTrue(marked)
        self.assertTrue(marked.issubset(sucker_cells))

    def test_off_is_zero_overhead(self):
        """explore_enabled=False leaves the map empty and never seeks."""
        octo, ag = _octo(explore=False, agents=0)
        for _ in range(5):
            octo.move(ag)
        self.assertEqual(octo.visit_recency.sum(), 0.0)
        # No agents and explore off => no node senses anything => no attraction.
        self.assertFalse(_attract_sw(octo).any())

    def test_recency_sets_not_accumulates(self):
        """A cell under a sucker is SET to 1.0, never incremented, so dwell can't
        inflate it - the map never exceeds 1.0 no matter how long it runs."""
        octo, ag = _octo(explore=True, agents=0, decay=0.5)
        for _ in range(8):
            octo.move(ag)
        self.assertLessEqual(float(octo.visit_recency.max()), 1.0 + 1e-9)
        # Cells the suckers are on right now read exactly 1.0 (freshest).
        cells = {(min(max(round(s.y), 0), 19), min(max(round(s.x), 0), 19))
                 for limb in octo.limbs for s in limb.suckers}
        for (cy, cx) in cells:
            self.assertAlmostEqual(octo.visit_recency[cy, cx], 1.0, places=6)

    def test_decay_ages_untouched_cells(self):
        """decay < 1 fades a cell by decay**frames_since_last_visit. A far corner
        no sucker occupies decays one step per _mark_explored; at decay = 1.0 it
        never fades (touched-once == touched-forever, pure recency)."""
        octo, ag = _octo(explore=True, agents=0, decay=0.5)
        octo.move(ag)
        octo.visit_recency[0, 0] = 1.0     # a far corner: no sucker is here
        octo._mark_explored()              # decays all, re-marks sucker cells
        self.assertAlmostEqual(octo.visit_recency[0, 0], 0.5, places=6)
        octo._mark_explored()
        self.assertAlmostEqual(octo.visit_recency[0, 0], 0.25, places=6)
        # decay = 1.0: an untouched cell never fades.
        octo2, ag2 = _octo(explore=True, agents=0, decay=1.0)
        octo2.move(ag2)
        octo2.visit_recency[0, 0] = 1.0
        octo2._mark_explored()
        self.assertAlmostEqual(octo2.visit_recency[0, 0], 1.0, places=6)


class TestExplorationSeeking(unittest.TestCase):
    def test_coverage_grows_when_exploring(self):
        """An idle octopus with exploration on covers more cells over time."""
        octo, ag = _octo(explore=True, agents=0)
        first = None
        for f in range(30):
            octo.move(ag)
            if f == 0:
                first = int((octo.visit_recency > 0).sum())
        last = int((octo.visit_recency > 0).sum())
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

    def test_explore_target_is_nearest_least_recently_visited_cell(self):
        """LEXICOGRAPHIC selection: the chosen cell has the MINIMUM recency in
        reach, and is the CLOSEST of that least-recently-visited set (recency
        wins over distance - never prefer a near recent cell to a far stale one).
        """
        octo, ag = _octo(explore=True, agents=0)
        octo.move(ag)  # seed the recency map
        limb = octo.limbs[0]
        node = limb.center_line[1]  # a free node
        _, cell = limb._node_explore_target(node.x, node.y,
                                            octo.visit_recency, None)
        self.assertIsNotNone(cell)
        cx, cy = int(round(cell[0])), int(round(cell[1]))
        reach = limb.explore_node_radius
        vr = octo.visit_recency
        y_len, x_len = vr.shape
        reachable = [(float(vr[gy, gx]), np.hypot(gx - node.x, gy - node.y),
                      gx, gy)
                     for gy in range(y_len) for gx in range(x_len)
                     if np.hypot(gx - node.x, gy - node.y) <= reach]
        min_r = min(r for r, _, _, _ in reachable)
        # Chosen cell is at the minimum recency...
        self.assertAlmostEqual(float(vr[cy, cx]), min_r, places=6)
        # ...and is the closest among the min-recency set.
        chosen_d = np.hypot(cx - node.x, cy - node.y)
        for r, d, _gx, _gy in reachable:
            if abs(r - min_r) <= 1e-6:
                self.assertLessEqual(chosen_d, d + 1e-9)

    def test_explore_target_avoids_a_threat(self):
        """A sensed threat pushes the explore goal off the cell it would
        otherwise pick - the fix for the stall behind a threat."""
        octo, ag = _octo(explore=True, agents=0)
        octo.move(ag)  # seed geometry + map
        limb = octo.limbs[0]
        node = limb.center_line[1]
        # Where it goes with no threat...
        _, free_cell = limb._node_explore_target(node.x, node.y,
                                                 octo.visit_recency, None)
        self.assertIsNotNone(free_cell)
        # ...drop a threat right on that cell; the goal must move off it.
        threat = [free_cell[0], free_cell[1]]
        _, avoided = limb._node_explore_target(node.x, node.y,
                                               octo.visit_recency, threat)
        self.assertIsNotNone(avoided)
        d_free = np.hypot(free_cell[0] - threat[0], free_cell[1] - threat[1])
        d_avoid = np.hypot(avoided[0] - threat[0], avoided[1] - threat[1])
        self.assertGreater(d_avoid, d_free)


class TestRewardHierarchy(unittest.TestCase):
    def test_prey_preempts_exploration(self):
        """A node that senses prey attracts to the PREY (strong weight), not to
        the gentle explore target - prey outranks exploration, per node."""
        octo, ag = _octo(explore=True, agents=1)
        cx, cy = octo.x, octo.y
        prey = Agent(x=cx + 2.0, y=cy, agent_type=AgentType.PREY)
        ag.agents = [prey]
        for _ in range(5):
            prey.x, prey.y = cx + 2.0, cy
            octo.move(ag)
        sw = _attract_sw(octo)
        strong = np.sqrt(octo.limbs[0].ilqr_cfg.w_reach_terminal)  # prey
        gentle = np.sqrt(octo.limbs[0].w_explore)                  # explore
        self.assertTrue(np.isclose(sw.max(), strong))  # a node senses prey
        self.assertGreater(strong, gentle)             # prey > explore drive

    def test_explore_reach_is_gentler_than_prey(self):
        """w_explore must stay well below w_reach_terminal so exploration is a
        weak drive (the 'much less reward' requirement)."""
        cfg = make_config(octo_ilqr_explore_enabled=True)
        self.assertLess(cfg.octopus.limb.ilqr.w_explore,
                        cfg.octopus.limb.ilqr.w_reach_terminal)

    def test_prey_and_threat_act_per_node(self):
        """Node-autonomous (NOT a limb policy): with a prey AND a threat both in
        range, some nodes attract (prey) while others flee (threat) at the same
        time - neither preempts the other at the limb level."""
        octo, ag = _octo(explore=True, agents=0)
        cx, cy = octo.x, octo.y
        prey = Agent(x=cx + 2.0, y=cy, agent_type=AgentType.PREY)
        threat = Agent(x=cx - 2.0, y=cy, agent_type=AgentType.THREAT)
        ag.agents = [prey, threat]
        for _ in range(5):
            prey.x, prey.y = cx + 2.0, cy
            threat.x, threat.y = cx - 2.0, cy
            octo.move(ag)
        self.assertTrue(_attract_sw(octo).any())  # some node attracts (prey)
        self.assertTrue(_repel_sw(octo).any())    # some node flees (threat)


if __name__ == '__main__':
    unittest.main()
