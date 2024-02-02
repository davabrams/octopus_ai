""" Octopus model training """
import pickle
import time as tm
import numpy as np
import tensorflow as tf
from tensorflow import keras
from OctoConfig import GameParameters, TrainingParameters
from util import erase_all_logs
from training.sucker import SuckerTrainer
from training.limb import LimbTrainer
from training.losses import WeightedSumLoss
from simulator.simutil import MLMode

np.set_printoptions(precision=4)
tf.config.run_functions_eagerly(False)

# %% Entry point for octopus modeling
ML_MODE = TrainingParameters['ml_mode']
RUN_DATAGEN = TrainingParameters['datagen_mode']
SAVE_DATA_TO_DISK = TrainingParameters['save_data_to_disk']

RESTORE_DATA_FROM_DISK = TrainingParameters["restore_data_from_disk"]
RUN_TRAINING = TrainingParameters["run_training"]
ERASE_OLD_TENSORBOARD_LOGS = TrainingParameters["erase_old_tensorboard_logs"]
GENERATE_TENSORBOARD = TrainingParameters["generate_tensorboard"]
SAVE_MODEL_TO_DISK = TrainingParameters["save_model_to_disk"]

RESTORE_MODEL_FROM_DISK = TrainingParameters["restore_model_from_disk"]
RUN_INFERENCE = TrainingParameters["run_inference"]

RUN_EVAL = TrainingParameters["run_eval"]

test_dataset = None
train_dataset = None

start = tm.time()
print(f"Octo Model started at {start}, setting t=0.0")

if ERASE_OLD_TENSORBOARD_LOGS:
    erase_all_logs()

if ML_MODE == MLMode.SUCKER:
    trainer = SuckerTrainer(GameParameters)
    datagen_location = TrainingParameters['sucker_datagen_location']
    model_location = TrainingParameters['sucker_model_location']
elif ML_MODE == MLMode.LIMB:
    trainer = LimbTrainer(GameParameters, TrainingParameters)
    datagen_location = TrainingParameters['limb_datagen_location']
    model_location = TrainingParameters['limb_model_location']
else:
    raise ValueError("No trainer available for selected ML Mode, check GameParameters")

# %% Data Gen
data = None
if RUN_DATAGEN:
    data = trainer.datagen(SAVE_DATA_TO_DISK)
    train_dataset, test_dataset = trainer.data_format(data)
    print(f"Datagen completed at time t={tm.time() - start}")

elif RESTORE_DATA_FROM_DISK:
    with open(datagen_location, 'rb') as file:
        data = pickle.load(file)
    assert data, "No data found, can't run training."
    train_dataset, test_dataset = trainer.data_format(data)
    print(f"Data load completed at time t={tm.time() - start}")

else:
    # No data is specified.  It may not be needed.
    pass

if data:
    print(f"Training model with {len(data['gt_data'])} data points")



# %% Model training
if RUN_TRAINING:

    model = trainer.train(train_dataset)
    print(f"Model training completed at time t={tm.time() - start:.3f}")

    # %% Model deployment (this is only run if a new model was successfully trained)
    if SAVE_MODEL_TO_DISK:
        model.save(model_location)
        print(f"Model deployment completed at time t={tm.time() - start:.3f}")

# %% Model inference
if RUN_INFERENCE:
    ####### Load model
    if RESTORE_MODEL_FROM_DISK:
        custom_objects = {"WeightedSumLoss": WeightedSumLoss}
        model = keras.models.load_model(model_location, custom_objects)
        print(f"Model load completed at time t={tm.time() - start:.3f}")

    trainer.inference(model)
    print(f"Model inference completed at time t={tm.time() - start:.3f}")

# %% Model eval
if RUN_EVAL:
    assert "test_dataset" in dir(), "Error: eval specified but not test_dataset exists"
    assert test_dataset is not None, "Error: empty test_dataset"
    # For color, eval is defined as the average of RMS of the RGB values
    # mean([rms(pred, gt) for each (pred, gt) in octopus])
    BATCH_SIZE = TrainingParameters['batch_size']
    loss, accuracy = model.evaluate(test_dataset, batch_size=BATCH_SIZE)
    print(f"Loss: {loss:.3f}, Accuracy: {accuracy:.3f}")
    print(f"Model eval completed at time t={tm.time() - start:.3f}")

# %% End and cleanup
print(f"octo AI completed at time t={tm.time() - start:.3f}")
end = tm.time()
print(f"tensorflow took: {end - start:.3f} seconds")
