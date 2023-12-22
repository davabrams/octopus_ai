""" Octopus model training """
import datetime
import pickle
import time as tm
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tensorflow import keras
from simulator.octo_datagen import OctoDatagen
from OctoConfig import GameParameters
from util import ConstraintLoss, octo_norm, train_test_split


# %% Entry point for octopus modeling
RUN_DATAGEN = GameParameters['datagen_mode']
SAVE_DATA_TO_DISK = True

RESTORE_DATA_FROM_DISK = True
RUN_TRAINING = True
GENERATE_TENSORBOARD = True
SAVE_MODEL_TO_DISK = True

RESTORE_MODEL_FROM_DISK = True
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
    if RESTORE_DATA_FROM_DISK:
        del data
        with open('datagen/sucker_data.pkl', 'rb') as file:
            data = pickle.load(file)

    assert data, "No data found, can't run training."
    print(f"Training model with {len(data['gt_data'])} data points")
    scaler = preprocessing.MinMaxScaler()
    input_data = np.array([data['state_data']]) #sucker's current color
    label_data = np.array([data['gt_data']]) #sucker's ground truth
    input_data_norm = np.array(list(map(octo_norm, input_data)))
    label_data_norm = np.array(list(map(octo_norm, label_data)))
    train_data, train_labels, val_data, val_labels = train_test_split(input_data_norm,
                                                                      label_data_norm)
    data_input = np.transpose(np.stack((train_data, train_labels)))
    data_val = np.transpose(np.stack((val_data, val_labels)))

    sucker_model = keras.Sequential()
    sucker_model.add(keras.layers.Dense(units=5, input_dim=2, activation="relu", name="hidden_layer1"))
    sucker_model.add(keras.layers.Dense(units=5, activation="relu", name="hidden_layer2"))
    # sucker_model.add(keras.layers.Dense(units=5, activation="relu", name="hidden_layer3"))
    sucker_model.add(keras.layers.Dense(units=1, activation="tanh", name="prediction"))

    losses = [ConstraintLoss(original_values=input_data), keras.losses.MeanSquaredError()]
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

    sucker_model.fit(x=data_input,
            y=train_labels,
            epochs=GameParameters['epochs'],
            batch_size=GameParameters['batch_size'],
            callbacks=callbacks)

    if GENERATE_TENSORBOARD:
        print(f"Tensorboard generated, run with:\ntensorboard serve --logdir {log_dir}\n")

    loss, accuracy = sucker_model.evaluate(x=data_val, y=val_labels)

    print(f"Loss: {loss}, Accuracy: {accuracy}")
    print(f"Model training completed at time t={tm.time() - start}")

    # %% Model deployment (this is only run if a new model was successfully trained)
    if SAVE_MODEL_TO_DISK:
        sucker_model.save('models/sucker.keras')
        print(f"Model deployment completed at time t={tm.time() - start}")

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
            loss_str = ""
            for loss_func in losses:
                loss = float(loss_func([gt], [pred]))
                loss_str += f"{loss:.3f}, "
            print(f"{curr:.2f}, {gt:.2f} -> {pred:.3f} (losses = {loss_str})")
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
    print(f"Model inference completed at time t={tm.time() - start}")

# %% Model eval
if RUN_EVAL:
    # For sucker color, eval is defined as the average of RMS of the RGB values
    # mean([rms(pred, gt) for each (pred, gt) in octopus])
    print(f"Model eval completed at time t={tm.time() - start}")

# %% End and cleanup
print(f"octo AI completed at time t={tm.time() - start}")
end = tm.time()
print(f"tensorflow took: {end - start:.4f} seconds")
