"""
Tests for the typed config schema, its profiles, and the legacy flat views.
"""
import os
import sys
import unittest
from dataclasses import FrozenInstanceError, replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config_schema import Config, OutputConfig
from OctoConfig import (
    DATAGEN,
    DEBUG,
    DEFAULT,
    TEST,
    VIZ,
    default_models,
    to_game_parameters,
    to_training_parameters,
)
from simulator.simutil import MLMode, MovementMode


class TestConfigIsImmutable(unittest.TestCase):
    def test_config_cannot_be_mutated_in_place(self):
        """Frozen configs delete the bug class where a test or the
        websocket server half-mutates shared global state."""
        with self.assertRaises(FrozenInstanceError):
            DEFAULT.run.num_iterations = 999

    def test_nested_config_cannot_be_mutated(self):
        with self.assertRaises(FrozenInstanceError):
            DEFAULT.octopus.limb.chain.spring_k = 999

    def test_replace_derives_without_touching_the_original(self):
        before = DEFAULT.output.show_forces
        derived = replace(DEFAULT,
                          output=replace(DEFAULT.output, show_forces=True))
        self.assertTrue(derived.output.show_forces)
        self.assertEqual(DEFAULT.output.show_forces, before)


class TestProfiles(unittest.TestCase):
    def test_default_writes_nothing(self):
        """A fresh checkout must not write DBs or videos unprompted."""
        self.assertFalse(DEFAULT.output.log_forces)
        self.assertFalse(DEFAULT.output.save_images)
        self.assertFalse(DEFAULT.output.show_forces)
        self.assertFalse(DEFAULT.output.debug_mode)

    def test_test_profile_is_side_effect_free(self):
        self.assertFalse(TEST.output.log_forces)
        self.assertFalse(TEST.output.save_images)
        self.assertFalse(TEST.output.show_forces)
        self.assertFalse(TEST.output.debug_mode)

    def test_test_profile_needs_no_model_on_disk(self):
        self.assertEqual(TEST.inference.mode, MLMode.NO_MODEL)
        self.assertIsNone(TEST.inference_model_path)

    def test_test_profile_uses_random_movement(self):
        """RANDOM is the only mode where move() works without an agent."""
        self.assertEqual(TEST.octopus.movement_mode, MovementMode.RANDOM)
        self.assertEqual(TEST.octopus.limb.movement_mode,
                         MovementMode.RANDOM)
        self.assertEqual(TEST.agents.movement_mode, MovementMode.RANDOM)

    def test_viz_profile_shows_forces_but_stays_quiet_on_disk(self):
        self.assertTrue(VIZ.output.show_forces)
        self.assertFalse(VIZ.output.log_forces)
        self.assertFalse(VIZ.output.save_images)

    def test_debug_profile_turns_everything_on(self):
        self.assertTrue(DEBUG.output.debug_mode)
        self.assertTrue(DEBUG.output.show_forces)
        self.assertTrue(DEBUG.output.log_forces)
        self.assertTrue(DEBUG.output.save_images)

    def test_datagen_profile_is_quiet(self):
        self.assertFalse(DATAGEN.output.save_images)
        self.assertTrue(DATAGEN.datagen.save_to_disk)


class TestDuplicatedHyperparamsHaveOneSource(unittest.TestCase):
    """epochs / batch_size / test_size / constraint_loss_weight used to live
    in BOTH flat dicts with nothing keeping them in sync."""

    def test_both_legacy_views_read_the_same_training_source(self):
        cfg = replace(DEFAULT,
                      training=replace(DEFAULT.training, epochs=7,
                                       batch_size=3))
        game = to_game_parameters(cfg)
        train = to_training_parameters(cfg)
        self.assertEqual(game['epochs'], 7)
        self.assertEqual(train['epochs'], 7)
        self.assertEqual(game['batch_size'], 3)
        self.assertEqual(train['batch_size'], 3)

    def test_they_cannot_drift(self):
        for key in ('epochs', 'batch_size', 'test_size',
                    'constraint_loss_weight'):
            with self.subTest(key=key):
                self.assertEqual(to_game_parameters(DEFAULT)[key],
                                 to_training_parameters(DEFAULT)[key])


class TestSensingRadiusSplit(unittest.TestCase):
    """agent_range_radius used to serve two distinct concepts."""

    def test_agent_and_octopus_sensing_are_separate_fields(self):
        cfg = replace(
            DEFAULT,
            agents=replace(DEFAULT.agents, sensing_radius=2.0),
            octopus=replace(DEFAULT.octopus, sensing_radius=9.0),
        )
        self.assertEqual(cfg.agents.sensing_radius, 2.0)
        self.assertEqual(cfg.octopus.sensing_radius, 9.0)


class TestModelPathResolution(unittest.TestCase):
    """Call sites shouldn't index a path table by MLMode by hand."""

    def test_inference_model_path_resolves(self):
        self.assertEqual(DEFAULT.inference_model_path,
                         default_models[MLMode.SUCKER])

    def test_no_model_resolves_to_none(self):
        cfg = replace(DEFAULT,
                      inference=replace(DEFAULT.inference,
                                        model=MLMode.NO_MODEL))
        self.assertIsNone(cfg.inference_model_path)


class TestLegacyViewParity(unittest.TestCase):
    """The flat dicts are a compat shim; they must stay complete while call
    sites are migrated."""

    EXPECTED_GAME_KEYS = 45
    EXPECTED_TRAINING_KEYS = 18

    def test_game_parameters_key_count(self):
        self.assertEqual(len(to_game_parameters(DEFAULT)),
                         self.EXPECTED_GAME_KEYS)

    def test_training_parameters_key_count(self):
        self.assertEqual(len(to_training_parameters(DEFAULT)),
                         self.EXPECTED_TRAINING_KEYS)

    def test_mode_specific_knobs_reach_the_legacy_view(self):
        cfg = replace(
            DEFAULT,
            octopus=replace(
                DEFAULT.octopus,
                limb=replace(
                    DEFAULT.octopus.limb,
                    chain=replace(DEFAULT.octopus.limb.chain, spring_k=4.2),
                    lumped=replace(DEFAULT.octopus.limb.lumped,
                                   arm_stiffness=1.5),
                ),
            ),
        )
        game = to_game_parameters(cfg)
        self.assertEqual(game['octo_chain_spring_k'], 4.2)
        self.assertEqual(game['octo_arm_stiffness'], 1.5)


if __name__ == '__main__':
    unittest.main()
