"""
Test helpers for building configs.

WHY THIS EXISTS
---------------
Tests need to say "the standard test config, but with x_len=30 and
limb_rows=6". Spelling that with nested dataclasses is unreadable:

    replace(TEST, world=replace(TEST.world, x_len=30),
            octopus=replace(TEST.octopus,
                            limb=replace(TEST.octopus.limb, rows=6)))

So tests take flat keyword overrides and this module does the nesting:

    cfg = make_config(x_len=30, limb_rows=6)

That is shorter than the module-level params dict it replaced
(`p = GameParameters.copy(); p['x_len'] = 30`) AND it is typed underneath -
plus a misspelled key raises instead of silently doing nothing, which
`p['x_lenn'] = 30` never did.

The baseline is the TEST profile: side-effect free, RANDOM movement, no
model needed on disk.

NB: these are deliberately NOT named test_*. Pytest collects any
module-level test_* callable, so an imported `test_config` would be run
as a (vacuously passing) test in every file that imported it.
"""
from config_schema import Config
from OctoConfig import TEST, config_from_flat, config_to_flat

# The flat key surface, derived from the schema itself rather than
# hand-listed, so it cannot drift.
VALID_KEYS = frozenset(config_to_flat(TEST))


class UnknownConfigKey(KeyError):
    """Raised when an override names a key the schema doesn't have."""


def make_config(**overrides) -> Config:
    """A TEST-profile Config with flat-key overrides applied.

    Raises UnknownConfigKey on a key the schema doesn't define - a typo
    should fail loudly rather than silently not apply, which is exactly
    what the old dict-mutation pattern allowed.
    """
    unknown = set(overrides) - VALID_KEYS
    if unknown:
        raise UnknownConfigKey(
            f"unknown config key(s): {sorted(unknown)}. "
            f"Valid keys are the flat config surface; see "
            f"config_schema.py for the typed structure."
        )
    flat = {**config_to_flat(TEST), **overrides}
    return config_from_flat(flat)


def make_flat(**overrides) -> dict:
    """The flat dict form of make_config, for the few tests that genuinely
    exercise the flat<->nested boundary (e.g. the browser wire protocol or
    the force-log config snapshot)."""
    return config_to_flat(make_config(**overrides))
