# octopus_ai
https://davabrams.wordpress.com/

# contents
Octopus AI contains:
- A model for an octopus, consisting of a head, limbs, and suckers.  Also, backround surface image generator, and attractor/repeller agent generator.
- A simulator for simulating octopus motion using various methods.
- ML training and inference pipelines for sucker color change, ie camouflage.
- Tensorboard outputs for model training.
- An ML inference server so that various ML models can be run on a separate server or distributed cluster, instead of locally.
- Visualization tools for all this stuff.

# why are you doing this
I don't know, but I can't stop.

# what to do
`OctoConfig.py`
Configuration settings

`bazel run octo_viz`
octopus ai scenario synthesizer and visualizer

`bazel run octo_datagen`
octopus ai data generator for training

`bazel run octo_model`
octopus ai model trainer
