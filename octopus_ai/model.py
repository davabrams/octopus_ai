""" Octopus model training """
import os
import pickle
import sys
import time as tm

# This module lives in the octopus_ai/ package but is also a runnable entry
# point (`python octopus_ai/model.py`). Put the repo root on sys.path so the
# `octopus_ai.*` and sibling-package imports resolve when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from octopus_ai.config import TRAINING, print_config
from simulator.simutil import MLMode
from training.limb import LimbTrainer
from training.losses import (
    ClampedTargetLoss,
    DeltaColorLayer,
    WeightedSumLoss,
)
from training.models.model_loader import ModelLoader
from training.sucker import SuckerTrainer
from octopus_ai.util import erase_all_logs

np.set_printoptions(precision=4)
tf.config.run_functions_eagerly(False)

# %% Entry point for octopus modeling
# Select the profile here; derive a variant with dataclasses.replace() to
# experiment, e.g.
#     CFG = replace(TRAINING,
#                   training=replace(TRAINING.training, epochs=5))
CFG = TRAINING

print_config(CFG, "model.py CONFIG")

ML_MODE = CFG.training.ml_mode
RUN_DATAGEN = CFG.datagen.datagen_mode
SAVE_DATA_TO_DISK = CFG.datagen.save_to_disk

RESTORE_DATA_FROM_DISK = CFG.datagen.restore_from_disk
RUN_TRAINING = CFG.training.run_training
ERASE_OLD_TENSORBOARD_LOGS = CFG.training.erase_old_tensorboard_logs
GENERATE_TENSORBOARD = CFG.training.generate_tensorboard
SAVE_MODEL_TO_DISK = CFG.training.save_model_to_disk

RESTORE_MODEL_FROM_DISK = CFG.training.restore_model_from_disk
RUN_INFERENCE = CFG.training.run_inference

RUN_EVAL = CFG.training.run_eval

test_dataset = None
train_dataset = None

start = tm.time()
print(f"Octo Model started at {start}, setting t=0.0")

if ERASE_OLD_TENSORBOARD_LOGS:
    erase_all_logs()

# The config resolves both paths; no MLMode-keyed lookup here. Note the
# dataset is keyed by training.ml_mode and the model by
# training.training_model - the same MLMode in every shipped profile, but
# two separate knobs now.
datagen_location = CFG.training_dataset_path
model_location = CFG.training_model_path

if ML_MODE == MLMode.SUCKER:
    trainer = SuckerTrainer(CFG)
elif ML_MODE == MLMode.LIMB:
    trainer = LimbTrainer(CFG)
else:
    raise ValueError(
        "No trainer available for selected ML Mode, check the config"
    )

# %% Data Gen
data = None
if RUN_DATAGEN:
    data = trainer.datagen(SAVE_DATA_TO_DISK)
    train_dataset, test_dataset = trainer.data_format(data)
    print(f"Datagen completed at time t={tm.time() - start}")

elif RESTORE_DATA_FROM_DISK:
    assert os.path.isfile(datagen_location), (
        "Specified data file does not exist"
    )
    with open(datagen_location, 'rb') as file:
        data = pickle.load(file)
    assert data, "No data found, can't run training."
    train_dataset, test_dataset = trainer.data_format(data)
    print(f"Data load completed at time t={tm.time() - start}")

else:
    # No data is specified.  It may not be needed?
    pass

if data:
    print(f"Training model with {len(data['gt_data'])} data points")

# %% Model training
if RUN_TRAINING:

    model = trainer.train(
        train_dataset, GENERATE_TENSORBOARD=GENERATE_TENSORBOARD
    )
    print(f"Model training completed at time t={tm.time() - start:.3f}")

    # %% Model deployment (this is only run if a new model was successfully
    # trained)
    if SAVE_MODEL_TO_DISK:
        model.save(model_location)
        print(f"Model deployment completed at time t={tm.time() - start:.3f}")

# %% Model inference
if RUN_INFERENCE:
    # Load model
    if RESTORE_MODEL_FROM_DISK:
        custom_objects = {
            "WeightedSumLoss": WeightedSumLoss,
            "ClampedTargetLoss": ClampedTargetLoss,
            "DeltaColorLayer": DeltaColorLayer,
        }
        model = ModelLoader(
            model_location, custom_objects=custom_objects
        ).get_object()
        model.compile()
        print(f"Model load completed at time t={tm.time() - start:.3f}")

    trainer.inference(model)
    print(f"Model inference completed at time t={tm.time() - start:.3f}")

# %% Model eval
if RUN_EVAL:
    assert "test_dataset" in dir(), (
        "Error: eval specified but not test_dataset exists"
    )
    assert test_dataset is not None, "Error: empty test_dataset"
    # For color, eval is defined as the average of RMS of the RGB values
    # mean([rms(pred, gt) for each (pred, gt) in octopus])
    BATCH_SIZE = CFG.training.batch_size
    loss, accuracy = model.evaluate(test_dataset, batch_size=BATCH_SIZE)
    print(f"Loss: {loss:.3f}, Accuracy: {accuracy:.3f}")
    print(f"Model eval completed at time t={tm.time() - start:.3f}")

# %% End and cleanup
print(f"octo AI completed at time t={tm.time() - start:.3f}")
end = tm.time()
print(f"tensorflow took: {end - start:.3f} seconds")
