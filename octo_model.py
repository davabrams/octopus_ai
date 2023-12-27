""" Octopus model training """
import pickle
import time as tm
import numpy as np
import tensorflow as tf
from tensorflow import keras
from simulator.octo_datagen import OctoDatagen
from OctoConfig import GameParameters
from util import (
    erase_all_logs,
    run_sucker_model_inference,
    convert_pytype_to_tf_dataset,
    train_test_split
)
from training import train_sucker_model, WeightedSumLoss


np.set_printoptions(precision=4)
tf.config.run_functions_eagerly(False)

# %% Entry point for octopus modeling
RUN_DATAGEN = GameParameters['datagen_mode']
SAVE_DATA_TO_DISK = False

RESTORE_DATA_FROM_DISK = False
RUN_TRAINING = False
ERASE_OLD_TENSORBOARD_LOGS = False
GENERATE_TENSORBOARD = False
SAVE_MODEL_TO_DISK = False

RESTORE_MODEL_FROM_DISK = True
RUN_INFERENCE = True

RUN_EVAL = False

start = tm.time()
print(f"Octo Model started at {start}, setting t=0.0")

if ERASE_OLD_TENSORBOARD_LOGS:
    erase_all_logs()

# %% Data Gen
data = None
test_dataset = None
train_dataset = None
if RUN_DATAGEN:
    datagen = OctoDatagen(GameParameters)
    data = datagen.run_color_datagen()
    if SAVE_DATA_TO_DISK:
        with open('datagen/sucker_data.pkl', 'wb') as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
    print(f"Datagen completed at time t={tm.time() - start}")

# %% Model training
if RUN_TRAINING:
    ####### Import Data
    if RESTORE_DATA_FROM_DISK and not RUN_DATAGEN:
        del data
        with open('datagen/sucker_data.pkl', 'rb') as file:
            data = pickle.load(file)
    assert data, "No data found, can't run training."
    print(f"Training model with {len(data['gt_data'])} data points")


    ####### Format Data for Train and Val
    batch_size = GameParameters['batch_size']

    state_data = np.array([data['state_data']], dtype='float32') #sucker's current color
    gt_data = np.array([data['gt_data']], dtype='float32') #sucker's ground truth

    train_state_data, train_gt_data, test_state_data, test_gt_data = train_test_split(state_data, gt_data)

    train_dataset = convert_pytype_to_tf_dataset(np.transpose(np.stack((train_state_data, train_gt_data))),
                                                 batch_size)
    test_dataset = convert_pytype_to_tf_dataset(np.transpose(np.stack((test_state_data, test_gt_data))),
                                               batch_size)
    sucker_model = train_sucker_model(GameParameters=GameParameters,
                                      train_dataset=train_dataset,
                                      GENERATE_TENSORBOARD=GENERATE_TENSORBOARD)

    print(f"Model training completed at time t={tm.time() - start:.3f}")

    # %% Model deployment (this is only run if a new model was successfully trained)
    if SAVE_MODEL_TO_DISK:
        sucker_model.save('models/sucker.keras')
        print(f"Model deployment completed at time t={tm.time() - start:.3f}")

# %% Model inference
if RUN_INFERENCE:
    ####### Load model
    if RESTORE_MODEL_FROM_DISK:
        custom_objects = {"WeightedSumLoss": WeightedSumLoss}
        sucker_model = keras.models.load_model('models/sucker.keras', custom_objects)

    run_sucker_model_inference(sucker_model=sucker_model, GameParameters=GameParameters)

    print(f"Model inference completed at time t={tm.time() - start:.3f}")

# %% Model eval
if RUN_EVAL:
    # For sucker color, eval is defined as the average of RMS of the RGB values
    # mean([rms(pred, gt) for each (pred, gt) in octopus])
    batch_size = GameParameters['batch_size']
    loss, accuracy = sucker_model.evaluate(test_dataset, batch_size=batch_size)

    print(f"Loss: {loss:.3f}, Accuracy: {accuracy:.3f}")

    print(f"Model eval completed at time t={tm.time() - start:.3f}")

# %% End and cleanup
print(f"octo AI completed at time t={tm.time() - start:.3f}")
end = tm.time()
print(f"tensorflow took: {end - start:.3f} seconds")
