"""Core Octopus AI package: typed config schema, profiles, shared utilities,
and the datagen/training entry points.

Kept intentionally import-light — importing this package must not pull in
TensorFlow or run the training pipeline. Import the concrete modules
(``octopus_ai.config_schema``, ``octopus_ai.config``, ``octopus_ai.util``)
directly.
"""
