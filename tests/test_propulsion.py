"""External-reaction propulsion (PropulsionMode.REACTION, Octopus._propel_body).

The legacy body (PropulsionMode.INTERNAL, _drift_body_by_tension) translates
along the summed INTERNAL arm tension - non-physical, since internal forces
cannot move a free body's centre of mass. REACTION replaces that with forces the
environment reacts against:

  CRAWL - a PLANTED (world-stationary tip = gripping), PULLING arm hauls the body
    toward its grip point, capped per arm by a friction/adhesion budget; a
    swinging/reaching arm (fast-moving tip) is unanchored and moves nothing.
  JET   - a near threat fires a siphon burst straight away from it, decaying as
    the mantle refills.

These exercise _propel_body directly with rigged limb state, so they are fast and
independent of the iLQR solve.
"""
import os
import sys
import unittest
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from octopus_ai.config import TEST
from simulator.octopus_generator import Octopus
from simulator.simutil import AgentType, MovementMode, PropulsionMode


class _Threat:
    """Minimal duck-typed agent: _nearest_threat_escape reads only these."""
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.agent_type = AgentType.THREAT


def _reaction_octo(**octo_overrides):
    cfg = replace(TEST, octopus=replace(
        TEST.octopus,
        movement_mode=MovementMode.ILQR,          # so limbs build a center_line
        propulsion_mode=PropulsionMode.REACTION,
        **octo_overrides))
    octo = Octopus(cfg)
    octo.x, octo.y = 7.5, 7.5
    return octo


def _quiesce(octo):
    """No arm pulls, and every tip is marked as it currently sits (planted)."""
    for lb in octo.limbs:
        lb.last_tension = np.zeros(2, dtype=float)
        lb._propel_prev_tip = np.array([lb.center_line[-1].x,
                                        lb.center_line[-1].y], dtype=float)


class PropulsionDefaults(unittest.TestCase):
    def test_internal_is_the_default(self):
        # A fresh octopus must keep the legacy propulsion so existing runs and
        # every other test are untouched by this feature.
        self.assertEqual(TEST.octopus.propulsion_mode, PropulsionMode.INTERNAL)


class CrawlReaction(unittest.TestCase):
    def test_planted_pulling_arm_hauls_body_toward_grip_and_is_friction_capped(self):
        octo = _reaction_octo(crawl_grip_limit=0.15, crawl_plant_speed=0.08)
        _quiesce(octo)
        # Rig limb 0: tip planted at (10, 7.5) (+x of the body), pulling hard.
        lb = octo.limbs[0]
        lb.center_line[-1].x, lb.center_line[-1].y = 10.0, 7.5
        lb.last_tension = np.array([1.0, 0.0])          # pull >> grip budget
        lb._propel_prev_tip = np.array([10.0, 7.5])     # tip did not move: planted

        x0, y0 = octo.x, octo.y
        octo._propel_body(agents=[])                     # no threat -> pure crawl

        self.assertGreater(octo.x - x0, 0.0)             # moved toward the grip
        self.assertAlmostEqual(octo.y - y0, 0.0, places=6)
        # Thrust is clamped to the friction budget even though the pull is 1.0.
        self.assertAlmostEqual(float(np.hypot(*octo.last_crawl_thrust)),
                               0.15, places=6)

    def test_swinging_arm_produces_no_translation(self):
        # Same pull, but the tip moved far since last frame -> not anchored ->
        # an unanchored reach must not glide the body (the physical fix).
        octo = _reaction_octo(crawl_plant_speed=0.08)
        _quiesce(octo)
        lb = octo.limbs[0]
        lb.center_line[-1].x, lb.center_line[-1].y = 10.0, 7.5
        lb.last_tension = np.array([1.0, 0.0])
        lb._propel_prev_tip = np.array([2.0, 7.5])       # tip swung 8 units

        x0, y0 = octo.x, octo.y
        octo._propel_body(agents=[])
        self.assertAlmostEqual(octo.x - x0, 0.0, places=6)
        self.assertAlmostEqual(octo.y - y0, 0.0, places=6)
        self.assertAlmostEqual(float(np.hypot(*octo.last_crawl_thrust)),
                               0.0, places=6)

    def test_relaxed_arm_produces_no_translation(self):
        # Planted but slack (no tension) -> nothing to react against.
        octo = _reaction_octo()
        _quiesce(octo)
        x0, y0 = octo.x, octo.y
        octo._propel_body(agents=[])
        self.assertAlmostEqual(octo.x - x0, 0.0, places=6)
        self.assertAlmostEqual(octo.y - y0, 0.0, places=6)


class JetEscape(unittest.TestCase):
    def test_jet_pushes_straight_away_from_a_near_threat(self):
        octo = _reaction_octo(jet_enabled=True, jet_impulse=0.9,
                              max_jet_velocity=1.2, jet_trigger_radius=8.0)
        _quiesce(octo)
        # Threat directly +x of the body -> body must jet in -x.
        threat = _Threat(octo.x + 2.0, octo.y)
        x0, y0 = octo.x, octo.y
        octo._propel_body(agents=[threat])
        self.assertLess(octo.x - x0, 0.0)                # fled in -x
        self.assertAlmostEqual(octo.y - y0, 0.0, places=6)
        self.assertGreater(float(np.hypot(*octo.last_jet_v)), 0.0)

    def test_jet_is_speed_capped(self):
        octo = _reaction_octo(jet_enabled=True, jet_impulse=5.0,
                              max_jet_velocity=1.2, jet_decay=0.6,
                              jet_trigger_radius=8.0)
        _quiesce(octo)
        threat = _Threat(octo.x + 1.0, octo.y)
        octo._propel_body(agents=[threat])               # huge impulse...
        self.assertLessEqual(float(np.hypot(*octo.last_jet_v)),
                             1.2 + 1e-6)                  # ...clamped to the cap

    def test_threat_beyond_trigger_radius_does_not_fire(self):
        # A threat just outside jet_trigger_radius must not scramble the jet.
        octo = _reaction_octo(jet_enabled=True, jet_trigger_radius=6.0)
        _quiesce(octo)
        x0 = octo.x
        octo._propel_body(agents=[_Threat(octo.x + 7.0, octo.y)])
        self.assertAlmostEqual(octo.x - x0, 0.0, places=6)
        self.assertAlmostEqual(float(np.hypot(*octo.last_jet_v)), 0.0, places=6)

    def test_jet_decays_to_rest_once_the_threat_is_gone(self):
        octo = _reaction_octo(jet_enabled=True, jet_impulse=0.9,
                              max_jet_velocity=1.2, jet_decay=0.6,
                              jet_trigger_radius=8.0)
        _quiesce(octo)
        octo._propel_body(agents=[_Threat(octo.x + 2.0, octo.y)])
        fired = float(np.hypot(*octo.last_jet_v))
        self.assertGreater(fired, 0.0)
        # No threat now: the jet must decay, not persist.
        for _ in range(20):
            _quiesce(octo)                               # keep crawl out of it
            octo._propel_body(agents=[])
        self.assertLess(float(np.hypot(*octo.last_jet_v)), fired)
        self.assertAlmostEqual(float(np.hypot(*octo.last_jet_v)), 0.0, places=4)

    def test_disabled_jet_never_fires(self):
        octo = _reaction_octo(jet_enabled=False, jet_trigger_radius=8.0)
        _quiesce(octo)
        x0 = octo.x
        octo._propel_body(agents=[_Threat(octo.x + 1.0, octo.y)])
        self.assertAlmostEqual(octo.x - x0, 0.0, places=6)
        self.assertAlmostEqual(float(np.hypot(*octo.last_jet_v)), 0.0, places=6)


if __name__ == '__main__':
    unittest.main()
