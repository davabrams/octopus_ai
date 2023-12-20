""" Octopus model training """
import pickle
import time as tm
import numpy as np
from sklearn import preprocessing
from tensorflow import keras
from simulator.octo_datagen import OctoDatagen
from OctoConfig import GameParameters
from util import ConstraintLoss, OctoNorm, train_test_split


# %% Entry point for octopus modeling
RUN_DATAGEN = True
SAVE_DATA_TO_DISK = True

RESTORE_DATA_FROM_DISK = True
RUN_TRAINING = True
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
    scaler = preprocessing.MinMaxScaler()
    input_data = np.array([data['state_data']]) #sucker's current color and the ground truth
    label_data = np.array([data['gt_data']]) #sucker's ground truth
    input_data_norm = np.array(list(map(OctoNorm, input_data)))
    label_data_norm = np.array(list(map(OctoNorm, label_data)))
    train_data, train_labels, val_data, val_labels = train_test_split(input_data_norm,
                                                                      label_data_norm)
    x1 = keras.Input(shape =(1,))
    x2 = keras.Input(shape =(1,))

    input_layer = keras.layers.concatenate([x1, x2])
    hidden_layer = keras.layers.Dense(units=4, activation="sigmoid")(input_layer)
    prediction = keras.layers.Dense(units=1, activation="sigmoid")(hidden_layer)
    model = keras.Model(inputs=[x1, x2], outputs=prediction)

    model.compile(optimizer="SGD",
                loss=["mean_squared_error", ConstraintLoss(original_values=input_data)],
                metrics=["mse"],
                run_eagerly=True)

    model.fit(x=[train_data, train_labels],
            y=train_labels,
            epochs=GameParameters['epochs'],
            batch_size=GameParameters['batch_size'])

    loss, accuracy = model.evaluate(x=[val_data, val_labels], y=val_labels)

    print(f"Loss: {loss}, Accuracy: {accuracy}")
    print(f"Model training completed at time t={tm.time() - start}")

    # %% Model deployment (this is only run if a new model was successfully trained)
    if SAVE_MODEL_TO_DISK:
        model.save('models/sucker.keras')
        print(f"Model deployment completed at time t={tm.time() - start}")

# %% Model inference
if RUN_INFERENCE:
    if RESTORE_MODEL_FROM_DISK:
        custom_objects = {"ConstraintLoss": ConstraintLoss}
        model = keras.models.load_model('models/sucker.keras', custom_objects)

    assert model, "No model found, can't run inference"
    for curr in map(OctoNorm, [0.0,0.25,0.5,0.75,1.0]):
        for gt in map(OctoNorm, [0.0,0.25,0.5,0.75,1.0]):
            test_input = [np.expand_dims(np.array(1.), 0),
                    np.expand_dims(np.array(1.), 0)]
            pred = model.predict(test_input, verbose = 0)[0][0]
            print(f"{curr:.2f}, {gt:.2f} -> {pred:.3f}")

    print(f"Model inference completed at time t={tm.time() - start}")

# %% Model eval
if RUN_EVAL:
    print(f"Model eval completed at time t={tm.time() - start}")

# %% End and cleanup
print(f"octo AI completed at time t={tm.time() - start}")
end = tm.time()
print(f"tensorflow took: {end - start:.4f} seconds")
