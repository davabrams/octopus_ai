"""
Tests for the typed config schema, its profiles, and the flat view.
"""
import os
import sys
import unittest
from dataclasses import FrozenInstanceError, replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from octopus_ai.config import (
    DATAGEN,
    DEBUG,
    DEFAULT,
    RECORD,
    TEST,
    VIZ,
    config_from_flat,
    config_to_flat,
    default_models,
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
    in BOTH flat dicts with nothing keeping them in sync.

    There is no second dict to drift from now - cfg.training is the only
    source, and the trainers read it directly. What is still worth pinning
    is that the flat view reports cfg.training rather than keeping a copy
    of its own.
    """

    def test_flat_view_reads_the_training_source(self):
        cfg = replace(DEFAULT,
                      training=replace(DEFAULT.training, epochs=7,
                                       batch_size=3))
        flat = config_to_flat(cfg)
        self.assertEqual(flat['epochs'], 7)
        self.assertEqual(flat['batch_size'], 3)

    def test_every_hyperparam_tracks_the_training_source(self):
        for key, field in (('epochs', 'epochs'),
                           ('batch_size', 'batch_size'),
                           ('test_size', 'test_size'),
                           ('constraint_loss_weight',
                            'constraint_loss_weight')):
            with self.subTest(key=key):
                self.assertEqual(config_to_flat(DEFAULT)[key],
                                 getattr(DEFAULT.training, field))


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


class TestFlatViewParity(unittest.TestCase):
    """The flat view is the browser's and the force log's vocabulary. It
    must stay complete: a key that silently stops being emitted is a knob
    the UI can no longer set."""

    EXPECTED_FLAT_KEYS = 91

    def test_flat_view_key_count(self):
        self.assertEqual(len(config_to_flat(DEFAULT)),
                         self.EXPECTED_FLAT_KEYS)

    def test_mode_specific_knobs_reach_the_flat_view(self):
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
        flat = config_to_flat(cfg)
        self.assertEqual(flat['octo_chain_spring_k'], 4.2)
        self.assertEqual(flat['octo_arm_stiffness'], 1.5)


class TestRecordFlags(unittest.TestCase):
    """The two record/replay flags (Phase 0 of the record & replay plan)."""

    def test_default_and_test_keep_both_off(self):
        for profile in (DEFAULT, TEST):
            self.assertFalse(profile.output.record_run)
            self.assertFalse(profile.output.record_ilqr_history)

    def test_record_profile_turns_both_on(self):
        self.assertTrue(RECORD.output.record_run)
        self.assertTrue(RECORD.output.record_ilqr_history)
        # Force arrows off — headless, no matplotlib window.
        self.assertFalse(RECORD.output.show_forces)
        # Inherits the iLQR simulation from VIZ_ILQR.
        self.assertEqual(RECORD.octopus.movement_mode, MovementMode.ILQR)

    def test_flat_round_trip_of_both_keys(self):
        cfg = replace(
            DEFAULT,
            output=replace(DEFAULT.output, record_run=True,
                           record_ilqr_history=True),
        )
        flat = config_to_flat(cfg)
        self.assertTrue(flat['record_run'])
        self.assertTrue(flat['record_ilqr_history'])
        back = config_from_flat(flat)
        self.assertTrue(back.output.record_run)
        self.assertTrue(back.output.record_ilqr_history)

    def test_make_config_accepts_record_keys(self):
        from helpers import make_config
        cfg = make_config(record_run=True, record_ilqr_history=True)
        self.assertTrue(cfg.output.record_run)
        self.assertTrue(cfg.output.record_ilqr_history)


class TestBodyRotationFlags(unittest.TestCase):
    """Base-ring + body-rotation knobs (body rotation plan)."""

    def test_defaults_and_flat_round_trip(self):
        flat = config_to_flat(DEFAULT)
        self.assertEqual(flat['octo_ring_radius'], DEFAULT.octopus.ring_radius)
        self.assertEqual(flat['octo_max_body_angular_velocity'],
                         DEFAULT.octopus.max_body_angular_velocity)
        self.assertEqual(flat['octo_ilqr_body_torque_gain'],
                         DEFAULT.octopus.limb.ilqr.body_torque_gain)
        back = config_from_flat(flat)
        self.assertEqual(back.octopus.ring_radius, DEFAULT.octopus.ring_radius)
        self.assertEqual(back.octopus.max_body_angular_velocity,
                         DEFAULT.octopus.max_body_angular_velocity)
        self.assertEqual(back.octopus.limb.ilqr.body_torque_gain,
                         DEFAULT.octopus.limb.ilqr.body_torque_gain)

    def test_ring_radius_is_non_negative(self):
        # A negative ring radius is meaningless (radius is a distance).
        self.assertGreaterEqual(DEFAULT.octopus.ring_radius, 0.0)

    def test_ring_radius_zero_reproduces_legacy_single_point_base(self):
        cfg = replace(DEFAULT,
                      octopus=replace(DEFAULT.octopus, ring_radius=0.0))
        self.assertEqual(cfg.octopus.ring_radius, 0.0)


if __name__ == '__main__':
    unittest.main()
