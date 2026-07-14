"""Shared pytest configuration for the octopus_ai test suite.

WHY THIS EXISTS
---------------
Tests build their params with GameParameters.copy(), which means they
inherit whatever happens to be in OctoConfig at the time. That config is a
working file - it gets flipped to SPRING_CHAIN, save_images=True,
log_forces=True and so on during interactive runs. Tests then silently
depend on those values: the suite used to break whenever the movement mode
default changed, and any test touching a code path that reads save_images /
log_forces would start writing videos and SQLite databases into the repo.

The autouse fixture below forces a known-safe baseline for EVERY test and
restores the original afterwards, so:
  - no test can produce persistent artifacts (videos, frames, force DBs)
  - the suite's result does not depend on the developer's working config
  - a test that genuinely wants a mode/flag sets it in its own params copy,
    which still takes precedence (tests override the baseline, not vice versa)

Tests that need a real artifact (e.g. test_frame_recorder) must write into
tmp_path, never into the project's logs/ directory.
"""
import pytest

from OctoConfig import GameParameters
from simulator.simutil import MLMode, MovementMode

# Values every test starts from unless it explicitly overrides them.
SAFE_TEST_CONFIG = {
    # side-effecting: these cause writes to disk
    'save_images': False,
    'log_forces': False,
    # affects rendering only, but keep tests off the debug paths
    'show_forces': False,
    'debug_mode': False,
    # RANDOM is the only mode where move() works without an agent; tests
    # that want a spring mode opt in explicitly
    'octo_movement_mode': MovementMode.RANDOM,
    'limb_movement_mode': MovementMode.RANDOM,
    'agent_movement_mode': MovementMode.RANDOM,
    # don't depend on a trained model existing on disk
    'inference_mode': MLMode.NO_MODEL,
    'inference_model': MLMode.NO_MODEL,
}


@pytest.fixture(autouse=True)
def isolate_game_parameters():
    """Force the safe baseline for each test, then restore the real config.

    Autouse + function scope means this wraps every test (including
    unittest.TestCase ones, whose setUp runs inside it), so a
    GameParameters.copy() in setUp already sees the safe values.
    """
    original = GameParameters.copy()
    GameParameters.update(SAFE_TEST_CONFIG)
    try:
        yield
    finally:
        GameParameters.clear()
        GameParameters.update(original)
