"""
Guards the test-isolation invariant.

OctoConfig is a working file: it gets flipped to SPRING_CHAIN,
save_images=True, log_forces=True etc. during interactive runs. Tests build
their params from GameParameters.copy(), so without isolation they inherit
those values - which historically broke the suite whenever the movement
default changed, and risked tests writing videos / SQLite DBs into the repo.

tests/conftest.py forces a safe baseline for every test. These tests make
sure that stays true.
"""
import inspect
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from OctoConfig import GameParameters
from simulator.frame_recorder import FrameRecorder
from simulator.simutil import MLMode, MovementMode


class TestConfigIsolation(unittest.TestCase):
    def test_side_effecting_flags_are_off_for_tests(self):
        """No test should be able to write videos, frames or force DBs."""
        self.assertFalse(GameParameters['save_images'])
        self.assertFalse(GameParameters['log_forces'])
        self.assertFalse(GameParameters['show_forces'])
        self.assertFalse(GameParameters['debug_mode'])

    def test_movement_mode_baseline_is_random(self):
        """RANDOM is the only mode where move() works without an agent, so
        it is the baseline; tests wanting a spring mode opt in explicitly."""
        self.assertEqual(GameParameters['octo_movement_mode'],
                         MovementMode.RANDOM)
        self.assertEqual(GameParameters['limb_movement_mode'],
                         MovementMode.RANDOM)
        self.assertEqual(GameParameters['agent_movement_mode'],
                         MovementMode.RANDOM)

    def test_inference_baseline_needs_no_model_on_disk(self):
        self.assertEqual(GameParameters['inference_mode'], MLMode.NO_MODEL)
        self.assertEqual(GameParameters['inference_model'], MLMode.NO_MODEL)

    def test_a_test_can_still_override_the_baseline(self):
        """Isolation must not stop a test from opting in to a mode."""
        params = GameParameters.copy()
        params['octo_movement_mode'] = MovementMode.SPRING_CHAIN
        self.assertEqual(params['octo_movement_mode'],
                         MovementMode.SPRING_CHAIN)
        # ...without leaking to other tests
        self.assertEqual(GameParameters['octo_movement_mode'],
                         MovementMode.RANDOM)

    def test_frame_recorder_supports_a_non_repo_base_dir(self):
        """Tests that exercise the recorder must be able to write to tmp
        rather than the project's logs/ directory."""
        sig = inspect.signature(FrameRecorder.__init__)
        self.assertIn('base_dir', sig.parameters)


if __name__ == '__main__':
    unittest.main()
