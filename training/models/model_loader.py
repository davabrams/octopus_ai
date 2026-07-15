import logging
import os
import pathlib

import keras

from training.models.base_loader import DefaultLoader
from OctoConfig import default_models

# Importing losses runs the @keras.saving.register_keras_serializable
# decorators for the custom objects baked into the saved models
# (DeltaColorLayer, WeightedSumLoss, ConstraintLoss, ClampedTargetLoss).
# Loading a .keras model needs those registered first. Normally
# training/__init__.py pulls them in, but that side effect is unreliable
# under runners that synthesize empty package __init__.py files (e.g.
# Bazel), so register them explicitly at the point of use.
import training.losses  # noqa: F401

local_folder = pathlib.Path(__file__).parent.resolve()

all_models = [
    filename for filename in os.listdir(local_folder) if "keras" in filename
]


class ModelLoader(DefaultLoader):
    """
    Handles loading (and storage) of keras models.
    """

    custom_objects = None
    defaults = default_models

    def _load(self, **kwargs) -> None:
        """
        Load the model into memory.
        """
        if self.object is not None:
            logging.error(
                "Model already loaded! Not going to reload, and not going to "
                "error."
            )
            return

        if "custom_objects" in kwargs:
            self.custom_objects = kwargs["custom_objects"]
            self.object = keras.models.load_model(
                self.path, self.custom_objects
            )
        else:
            self.object = keras.models.load_model(self.path)
