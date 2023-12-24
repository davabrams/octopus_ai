""" Octopus model training """
import datetime
import pickle
import time as tm
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing
from tensorflow import keras
from simulator.octo_datagen import OctoDatagen
from OctoConfig import GameParameters
from util import (
    sucker_model_data_constructor, 
    ConstraintLoss, 
    octo_norm, 
    train_test_split, 
    convert_pytype_to_model_input_type
)


# %% Entry point for octopus modeling
RUN_DATAGEN = GameParameters['datagen_mode']
SAVE_DATA_TO_DISK = True

RESTORE_DATA_FROM_DISK = True
RUN_TRAINING = True
GENERATE_TENSORBOARD = True
SAVE_MODEL_TO_DISK = False

RESTORE_MODEL_FROM_DISK = False
RUN_INFERENCE = True

RUN_EVAL = False

start = tm.time()
print(f"Octo Model started at {start}, setting t=0.0")

# %% Data Gen
data = None
if RUN_DATAGEN:
    datagen = OctoDatagen(GameParameters)
    data = datagen.run_color_datagen()
    if SAVE_DATA_TO_DISK:
        with open('datagen/sucker_data.pkl', 'wb') as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
    print(f"Datagen completed at time t={tm.time() - start}")

# %% Model training
if RUN_TRAINING:
    tf.config.run_functions_eagerly(False)
    if RESTORE_DATA_FROM_DISK:
        del data
        with open('datagen/sucker_data.pkl', 'rb') as file:
            data = pickle.load(file)
    assert data, "No data found, can't run training."
    print(f"Training model with {len(data['gt_data'])} data points")
    constraint_loss_weight = GameParameters['constraint_loss_weight']
    max_hue_change = GameParameters['octo_max_hue_change']
    scaler = preprocessing.MinMaxScaler()
    state_data = np.array([data['state_data']]) #sucker's current color
    gt_data = np.array([data['gt_data']]) #sucker's ground truth
    state_data_norm = np.array(list(map(octo_norm, state_data)))
    gt_data_norm = np.array(list(map(octo_norm, gt_data)))
    train_data, train_labels, val_data, val_labels = train_test_split(state_data_norm,
                                                                      gt_data_norm)

    train_input_data, train_gt_data, train_orig_data = sucker_model_data_constructor(train_data, train_labels)
    val_input_data, val_gt_data, val_orig_data = sucker_model_data_constructor(val_data, val_labels)

    sucker_model = keras.Sequential()
    sucker_model.add(keras.layers.Dense(units=3, input_dim=2, activation="relu", name="hidden_layer1"))
    sucker_model.add(keras.layers.Dense(units=3, activation="relu", name="hidden_layer2"))
    sucker_model.add(keras.layers.Dense(units=3, activation="relu", name="hidden_layer3"))
    sucker_model.add(keras.layers.Dense(units=1, activation="tanh", name="prediction"))

    losses = [
        ConstraintLoss(original_values=train_orig_data,
                       threshold=max_hue_change,
                       weight=constraint_loss_weight),
        keras.losses.MeanSquaredError()]
    # Tensorboard configuration. To start tensorboard, use:
    # tensorboard serve --logdir <log directory>
    callbacks = []
    if GENERATE_TENSORBOARD:
        log_dir = "models/logs/sucker/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard_callback)

    sucker_model.compile(optimizer="sgd",
                loss=losses,
                metrics=["mse"])

    sucker_model.fit(x=train_input_data,
            y=train_gt_data,
            epochs=GameParameters['epochs'],
            batch_size=GameParameters['batch_size'],
            callbacks=callbacks)

    if GENERATE_TENSORBOARD:
        print(f"Tensorboard generated, run with:\n\n\ttensorboard serve --logdir {log_dir}\n")

    # These need to change because now our ConstraintLoss original value should be the val set 
    eval_losses = [
        ConstraintLoss(original_values=val_orig_data,
                       threshold=max_hue_change,
                       weight=constraint_loss_weight),
        keras.losses.MeanSquaredError()]
    sucker_model.compile(optimizer="sgd",
                loss=eval_losses,
                metrics=["mse"])
    loss, accuracy = sucker_model.evaluate(x=val_input_data,
                                           y=val_gt_data,
                                           batch_size=GameParameters['batch_size'],
                                           callbacks=callbacks)

    print(f"Loss: {loss:.3f}, Accuracy: {accuracy:.3f}")
    print(f"Model training completed at time t={tm.time() - start:.3f}")

    # %% Model deployment (this is only run if a new model was successfully trained)
    if SAVE_MODEL_TO_DISK:
        sucker_model.save('models/sucker.keras')
        print(f"Model deployment completed at time t={tm.time() - start:.3f}")

# %% Model inference
if RUN_INFERENCE:
    np.set_printoptions(precision=4)
    if RESTORE_MODEL_FROM_DISK:
        custom_objects = {"ConstraintLoss": ConstraintLoss}
        sucker_model = keras.models.load_model('models/sucker.keras', custom_objects)

    assert sucker_model, "No model found, can't run inference"
    range_vals = [0.0,0.25,0.5,0.75,1.0]
    res = []
    for curr in map(octo_norm, range_vals):
        row = []
        for gt in map(octo_norm, range_vals):
            test_input = np.array([[curr, gt]])

            #computes prediction output
            pred = sucker_model.predict(test_input, verbose = 0)[0][0]
            row.append(pred)

            #computes loss
            original_value = convert_pytype_to_model_input_type(curr)
            gt_inference_input = convert_pytype_to_model_input_type(gt)
            pred_inference_input = convert_pytype_to_model_input_type(pred)

            inference_losses = [ConstraintLoss(original_values=original_value,
                                    threshold=max_hue_change,
                                    weight=constraint_loss_weight),
                                keras.losses.MeanSquaredError()]

            loss_str = ""
            for loss_func in inference_losses:
                with tf.GradientTape() as tape:
                    loss = loss_func(gt_inference_input, pred_inference_input)
                gradients = tape.gradient(loss, tf.constant(pred))
                loss_str += f"\t{float(loss):.3f} \t<𝝳={gradients}>"

            print(f"{curr:.2f}, \t{gt:.2f} -> \t{pred:.3f} \t(losses = {loss_str})")
        res.append(octo_norm(row, True))

    #plots out the results
    plt.figure(figsize = (10,7))
    sn.heatmap(res, annot=True)
    plt.xlabel('surface color')
    locs, labels = plt.xticks()
    plt.xticks(locs, range_vals)
    plt.ylabel('suckers previous color')
    locs, labels = plt.yticks()
    plt.yticks(locs, range_vals)
    plt.show()
    
    print(f"Model inference completed at time t={tm.time() - start:.3f}")

# %% Model eval
if RUN_EVAL:
    # For sucker color, eval is defined as the average of RMS of the RGB values
    # mean([rms(pred, gt) for each (pred, gt) in octopus])
    print(f"Model eval completed at time t={tm.time() - start:.3f}")

# %% End and cleanup
print(f"octo AI completed at time t={tm.time() - start:.3f}")
end = tm.time()
print(f"tensorflow took: {end - start:.3f} seconds")
