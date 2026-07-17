"""Base ring + body rotation (BODY_ROTATION_PLAN.md).

The bug being fixed: in MovementMode.ILQR every limb's base was pinned to the
same body-center point, so the whole fan could collapse into one arm. The base
ring roots each limb at its own point on a ring around the body (keeping them
separated without coupling the independent solvers), and the body integrates the
arms' net torque into an orientation so the fan can rotate.
"""
import math
import os
import sys
import unittest
from itertools import combinations, pairwise

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from helpers import make_config

from simulator.agent_generator import AgentGenerator
from simulator.octopus_generator import Octopus
from simulator.simutil import MovementMode


def _ilqr_octo(*, ring_radius=1.0, num_arms=8, agents=3, seed=1):
    cfg = make_config(
        x_len=14, y_len=14, limb_rows=5, octo_num_arms=num_arms,
        agent_number_of_agents=agents,
        octo_movement_mode=MovementMode.ILQR,
        limb_movement_mode=MovementMode.ILQR,
        agent_movement_mode=MovementMode.LUMPED_SPRING,  # reactive => strain
        octo_ilqr_horizon=4, octo_ilqr_max_iters=3,
        octo_ring_radius=ring_radius)
    np.random.seed(seed)
    octo = Octopus(cfg)
    ag = AgentGenerator(cfg)
    ag.generate(agents)
    return octo, ag


def _base_points(octo):
    return np.array([[limb.center_line[0].x, limb.center_line[0].y]
                     for limb in octo.limbs])


def _min_pairwise(points):
    return min(float(np.linalg.norm(points[i] - points[j]))
               for i, j in combinations(range(len(points)), 2))


class TestBaseRing(unittest.TestCase):
    def test_bases_stay_equally_spaced_on_the_ring(self):
        """The N base points sit on a circle of radius R about the body center
        at equal angular spacing, EVERY frame (the no-collapse guarantee)."""
        N, R = 8, 1.0
        octo, ag = _ilqr_octo(ring_radius=R, num_arms=N)
        expected_chord = 2.0 * R * math.sin(math.pi / N)  # adjacent-base gap
        for _ in range(15):
            octo.move(ag)
            ag.increment_all(octo)
            bases = _base_points(octo)
            center = np.array([octo.x, octo.y])
            radii = np.linalg.norm(bases - center, axis=1)
            self.assertTrue(np.allclose(radii, R, atol=1e-4),
                            f"bases off the ring: {radii}")
            # Bases never collide: the closest pair is the equal-spacing chord.
            self.assertAlmostEqual(_min_pairwise(bases), expected_chord,
                                   places=4)

    def test_ring_radius_zero_reproduces_legacy_collapse(self):
        """R = 0 => all bases at the body center (min pairwise distance 0) —
        the legacy single-point base, as an escape hatch."""
        octo, ag = _ilqr_octo(ring_radius=0.0)
        for _ in range(5):
            octo.move(ag)
            ag.increment_all(octo)
        self.assertAlmostEqual(_min_pairwise(_base_points(octo)), 0.0,
                               places=6)


class TestBodyRotation(unittest.TestCase):
    def test_rotates_under_asymmetric_strain(self):
        """Reactive prey => asymmetric arm strain => the body orientation
        changes over time, and each step's rotation respects the cap."""
        octo, ag = _ilqr_octo(seed=3)
        thetas = [octo.theta]
        for _ in range(25):
            octo.move(ag)
            ag.increment_all(octo)
            thetas.append(octo.theta)
            self.assertLessEqual(abs(octo.last_body_dtheta),
                                 octo.max_body_angular_velocity + 1e-9)
        total_rotation = sum(abs(b - a) for a, b in pairwise(thetas))
        self.assertGreater(total_rotation, 1e-3,
                           "body never rotated under asymmetric strain")

    def test_idle_rotation_stays_capped(self):
        """The arm is NOT carried rigidly through the body's rotation (turning
        costs effort). Fully idle - no agents, nothing to reach - the body may
        slowly drift or even wind up to a spin, which is accepted. The invariant
        is only that each step still respects the angular-velocity cap (the
        clamp holds; no super-cap runaway)."""
        octo, ag = _ilqr_octo(agents=0)
        for _ in range(30):
            octo.move(ag)
            self.assertLessEqual(abs(octo.last_body_dtheta),
                                 octo.max_body_angular_velocity + 1e-9)

    def test_rotation_settles_with_reachable_stimulus(self):
        """A REACHABLE off-axis prey makes the body rotate to face it, then
        settle: the arm reaches the prey, its base strain turns radial, torque
        resolves. The arm is not carried rigidly through the rotation, so turning
        costs effort and this solve smooths the arm back toward the moved base;
        the spring DEADBAND absorbs the small per-frame rotation-lag strain, so it
        makes no feedback torque and the rotation is stable (settles) once the arm
        has something to reach.
        """
        from simulator.simutil import Agent, AgentType
        octo, ag = _ilqr_octo(agents=1)
        cx, cy = octo.x, octo.y
        # Reachable, off-axis (reach ~2.2 from centre; prey at 1.8).
        prey = Agent(x=cx, y=cy + 1.8, agent_type=AgentType.PREY)
        ag.agents = [prey]
        steps = []
        for _ in range(120):
            prey.x, prey.y = cx, cy + 1.8  # freeze the stimulus
            octo.move(ag)
            steps.append(abs(octo.last_body_dtheta))
        cap = octo.max_body_angular_velocity
        late = sum(steps[-15:]) / 15.0
        self.assertLess(late, 0.2 * cap,
                        f"rotation did not settle (late avg {late} vs cap {cap})")

    def test_rotation_cap_is_enforced(self):
        """A large torque gain still can't rotate faster than the cap."""
        octo, ag = _ilqr_octo(seed=5)
        octo.body_torque_gain = 1e6  # absurd gain
        octo.max_body_angular_velocity = 0.05
        for _ in range(10):
            octo.move(ag)
            ag.increment_all(octo)
            self.assertLessEqual(abs(octo.last_body_dtheta), 0.05 + 1e-9)


class TestThreatResponse(unittest.TestCase):
    """The body-drift response to prey/threats (two-sided base spring gated on
    a sensed threat, + whole-arm threat sensing)."""

    def _fixed_agent(self, kind, dx, dy):
        """Run with one FROZEN agent offset (dx, dy) from the start; return the
        body's net displacement vector."""
        from simulator.simutil import Agent, AgentType
        octo, ag = _ilqr_octo(agents=1)
        cx, cy = octo.x, octo.y
        at = AgentType.PREY if kind == "prey" else AgentType.THREAT
        agent = Agent(x=cx + dx, y=cy + dy, agent_type=at)
        ag.agents = [agent]
        x0, y0 = octo.x, octo.y
        for _ in range(20):
            agent.x, agent.y = cx + dx, cy + dy  # freeze the stimulus
            octo.move(ag)
        return octo.x - x0, octo.y - y0

    def test_idle_body_does_not_wander(self):
        """No agents => rope-like base tension => the body stays put (no idle
        jitter from the two-sided spring)."""
        octo, ag = _ilqr_octo(agents=0)
        x0, y0 = octo.x, octo.y
        for _ in range(20):
            octo.move(ag)
        self.assertLess(math.hypot(octo.x - x0, octo.y - y0), 0.05)

    def test_body_pursues_prey(self):
        """A fixed prey to the east pulls the body toward it (+x)."""
        ddx, _ = self._fixed_agent("prey", 5.0, 0.0)
        self.assertGreater(ddx, 0.02)

    def test_body_flees_threat(self):
        """A fixed threat to the east pushes the body away (-x) — the two-sided
        base spring pushing when the arm recoils."""
        ddx, _ = self._fixed_agent("threat", 1.5, 0.0)
        self.assertLess(ddx, -0.1)

    def test_threat_sensed_per_node(self):
        """Per-node sensing (node-autonomous): a threat inside a node's window
        makes THAT node flee (repel_sw > 0); a threat outside every node's window
        makes none flee."""
        from simulator.simutil import Agent, AgentType
        cfg = make_config(
            x_len=14, y_len=14, limb_rows=5, octo_num_arms=8,
            octo_movement_mode=MovementMode.ILQR,
            limb_movement_mode=MovementMode.ILQR,
            octo_ilqr_horizon=4, octo_ilqr_max_iters=3,
            record_ilqr_history=True)
        np.random.seed(1)
        octo = Octopus(cfg)
        ag = AgentGenerator(cfg)
        ag.generate(0)

        def any_flee():
            return any(limb.last_ilqr_meta["repel_sw"].any()
                       for limb in octo.limbs if limb.last_ilqr_meta)

        # A threat on the body: some node is inside its window -> it flees.
        ag.agents = [Agent(x=octo.x, y=octo.y, agent_type=AgentType.THREAT)]
        octo.move(ag)
        self.assertTrue(any_flee())

        # A threat far outside every node's window: nobody flees.
        ag.agents = [Agent(x=octo.x + 100.0, y=octo.y + 100.0,
                           agent_type=AgentType.THREAT)]
        octo.move(ag)
        self.assertFalse(any_flee())


class TestLimbsRemainIndependent(unittest.TestCase):
    def test_each_limb_has_its_own_controller_and_base_angle(self):
        """No cross-limb coupling: distinct controllers, distinct fixed angular
        slots evenly spaced around the body."""
        octo, ag = _ilqr_octo(num_arms=8)
        octo.move(ag)
        controllers = {id(limb._ilqr_controller) for limb in octo.limbs}
        self.assertEqual(len(controllers), len(octo.limbs))  # all distinct
        angles = sorted(limb.base_angle for limb in octo.limbs)
        expected = sorted(2 * math.pi * i / len(octo.limbs)
                          for i in range(len(octo.limbs)))
        self.assertTrue(np.allclose(angles, expected))


if __name__ == '__main__':
    unittest.main()
