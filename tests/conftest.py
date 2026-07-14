"""Shared pytest configuration for the octopus_ai test suite.

Puts the project root and this directory on sys.path so test modules can
import both the package under test and `helpers`.

There is deliberately NO config-isolation fixture here. There used to be:
GameParameters was a working dict that got flipped to save_images=True and
SPRING_CHAIN during interactive runs, and every test inherited it - the
suite broke three separate times that way, and nothing structurally stopped
a test writing videos into the repo.

The config reorg removed the need rather than the symptom. GameParameters is
now derived from the DEFAULT profile, whose defaults are side-effect free by
construction, and experimenting means selecting a profile (CFG = DEBUG in
octo_viz) instead of editing shared state. There is no longer a global to
leak, so there is nothing to isolate.

Tests that produce real artifacts (e.g. test_frame_recorder) still write to
tmp, never into the project's logs/ directory.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
