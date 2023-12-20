import numpy as np
import pickle
import tensorflow as tf
import time as tm
from sklearn import preprocessing
from tensorflow import keras

from OctoDatagen import OctoDatagen
from OctoConfig import GameParameters
from util import ConstraintLoss, OctoNorm, train_test_split

data = None

# %% Entry point for octopus modeling
run_datagen = True
run_model_training = True
run_model_inference = True
run_model_eval = False

start = tm.time()
print(f"Octo Model started at {start}, setting t=0.0")

# %% Data Gen
if run_datagen:
    datagen = OctoDatagen(GameParameters)
    data = datagen.run_color_datagen()
    file = open('datagen/sucker_data.pkl', 'wb')
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
    print(f"Datagen completed at time t={tm.time() - start}")

# %% Model training
if run_model_training:
    del data
    file = open('datagen/sucker_data.pkl', 'rb')
    data = pickle.load(file)
    scaler = preprocessing.MinMaxScaler()

    input_data = np.array([data['state_data']]) #sucker's current color and the ground truth
    label_data = np.array([data['gt_data']]) #sucker's ground truth
    input_data_norm = np.array(list(map(OctoNorm, input_data)))
    label_data_norm = np.array(list(map(OctoNorm, label_data)))
    train_data, train_labels, val_data, val_labels = train_test_split(input_data_norm, label_data_norm)
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

    # %% Model deployment
    if True:
        model.save('models/sucker.keras')
        print(f"Model deployment completed at time t={tm.time() - start}")

# %% Model inference
if run_model_inference:
    custom_objects = {"ConstraintLoss": ConstraintLoss}
    model = keras.models.load_model('models/sucker.keras', custom_objects)

    for curr in map(OctoNorm, [0.0,0.25,0.5,0.75,1.0]):
        for gt in map(OctoNorm, [0.0,0.25,0.5,0.75,1.0]):
            test_input = [np.expand_dims(np.array(1.), 0), 
                    np.expand_dims(np.array(1.), 0)]
            pred = model.predict(test_input, verbose = 0)[0][0]
            print(f"{curr:.2f}, {gt:.2f} -> {pred:.3f}")

    print(f"Model inference completed at time t={tm.time() - start}")

# %% Model eval
if run_model_eval:
    pass
    print(f"Model eval completed at time t={tm.time() - start}")


    x = tf.Variable(1.0)

    def f(x):
        y = x**2 + 2*x - 5
        return y

    print(f(1))

print(f"octo AI completed at time t={tm.time() - start}")











end = tm.time()
print(f"tensorflow took: {end - start:.4f} seconds")

