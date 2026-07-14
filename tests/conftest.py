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

The autouse fixture below pins the TEST profile for EVERY test and restores
the original afterwards, so:
  - no test can produce persistent artifacts (videos, frames, force DBs)
  - the suite's result does not depend on the developer's working config
  - a test that genuinely wants a mode/flag sets it in its own params copy,
    which still takes precedence (tests override the baseline, not vice versa)

The baseline is OctoConfig.TEST - a real profile in the config schema, not a
dict maintained here. Adding a side-effecting flag to the schema and
defaulting it off automatically covers the suite; there is no second list to
keep in sync.

Tests that need a real artifact (e.g. test_frame_recorder) must write into
tmp_path, never into the project's logs/ directory.
"""
import pytest

from OctoConfig import TEST, GameParameters, to_game_parameters

# The safe baseline every test starts from, derived from the TEST profile.
SAFE_TEST_CONFIG = to_game_parameters(TEST)


@pytest.fixture(autouse=True)
def isolate_game_parameters():
    """Pin the TEST profile for each test, then restore the real config.

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
