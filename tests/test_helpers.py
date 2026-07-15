"""
Tests for the make_config() helper.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))  # so `helpers` is importable

from octopus_ai.config_schema import Config
from helpers import UnknownConfigKey, make_config, make_flat
from simulator.simutil import MLMode, MovementMode


class TestTestConfigHelper(unittest.TestCase):
    def test_returns_a_typed_config(self):
        self.assertIsInstance(make_config(), Config)

    def test_baseline_is_the_test_profile(self):
        cfg = make_config()
        self.assertFalse(cfg.output.log_forces)
        self.assertFalse(cfg.output.save_images)
        self.assertEqual(cfg.octopus.movement_mode, MovementMode.RANDOM)
        self.assertEqual(cfg.inference.mode, MLMode.NO_MODEL)

    def test_flat_overrides_reach_the_right_nested_field(self):
        cfg = make_config(x_len=30, limb_rows=6, octo_num_arms=3)
        self.assertEqual(cfg.world.x_len, 30)
        self.assertEqual(cfg.octopus.limb.rows, 6)
        self.assertEqual(cfg.octopus.num_arms, 3)

    def test_mode_specific_overrides(self):
        cfg = make_config(octo_chain_spring_k=4.2, octo_arm_stiffness=1.5,
                          octo_max_arm_theta=0.9)
        self.assertEqual(cfg.octopus.limb.chain.spring_k, 4.2)
        self.assertEqual(cfg.octopus.limb.lumped.arm_stiffness, 1.5)
        self.assertEqual(cfg.octopus.limb.random.max_arm_theta, 0.9)

    def test_enum_overrides(self):
        cfg = make_config(octo_movement_mode=MovementMode.SPRING_CHAIN,
                          limb_movement_mode=MovementMode.SPRING_CHAIN)
        self.assertEqual(cfg.octopus.movement_mode,
                         MovementMode.SPRING_CHAIN)
        self.assertEqual(cfg.octopus.limb.movement_mode,
                         MovementMode.SPRING_CHAIN)

    def test_unknown_key_raises(self):
        """A typo must fail loudly. p['x_lenn'] = 30 never did."""
        with self.assertRaises(UnknownConfigKey):
            make_config(x_lenn=30)

    def test_unknown_key_message_names_the_key(self):
        with self.assertRaises(UnknownConfigKey) as ctx:
            make_config(octo_arm_stifness=1.0)  # missing an f
        self.assertIn("octo_arm_stifness", str(ctx.exception))

    def test_overrides_do_not_leak_between_calls(self):
        make_config(x_len=99)
        self.assertNotEqual(make_config().world.x_len, 99)

    def test_make_flat_returns_a_dict(self):
        flat = make_flat(x_len=30)
        self.assertIsInstance(flat, dict)
        self.assertEqual(flat['x_len'], 30)

    def test_agent_range_radius_sets_both_sensing_radii(self):
        """One flat key, two fields - documented behaviour of the flat
        boundary, preserved from before the split."""
        cfg = make_config(agent_range_radius=7.0)
        self.assertEqual(cfg.agents.sensing_radius, 7.0)
        self.assertEqual(cfg.octopus.sensing_radius, 7.0)


if __name__ == '__main__':
    unittest.main()
