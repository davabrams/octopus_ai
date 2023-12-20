import tensorflow as tf
import numpy as np
import time as tm
from AgentGenerator import AgentGenerator
from OctoDatagen import OctoDatagen
from Octopus import Octopus
from RandomSurface import RandomSurface
from util import Excess20Loss
from OctoConfig import GameParameters

""" Entry point for octopus modeling """

start = tm.time()
print(f"Octo Model started at {start}, setting t=0.0")

# %% Data Gen
datagen = OctoDatagen(GameParameters)
data = datagen.run_color_datagen()

# %% Model training
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from util import train_test_split

scaler = preprocessing.MinMaxScaler()

input_data = np.array([data['state_data'], data['gt_data']]) #sucker's current color
label_data = np.array([data['gt_data']]) #sucker's target color
train_data, train_labels, val_data, val_labels = train_test_split(input_data, label_data)
x1 = keras.Input(shape =(1,))
x2 = keras.Input(shape =(1,))
# train_data = np.stack((train_data, train_labels)).transpose()
# val_data = np.stack((val_data, val_labels)).transpose()

input_shape = np.size(train_data)

input_layer = keras.layers.concatenate([x1, x2])
hidden_layer = keras.layers.Dense(units=4, activation="sigmoid")(input_layer)
prediction = keras.layers.Dense(units=1, activation="sigmoid")(hidden_layer)
model = keras.Model(inputs=[x1, x2], outputs=prediction)

# model = keras.Sequential([
#     keras.layers.Dense(units=2, activation="sigmoid", input_shape=(2,))(input_layer),
#     keras.layers.Dense(units=1, activation="sigmoid", input_shape=(2,)),
# ])

model.compile(optimizer="SGD",
              loss=["mean_squared_error", Excess20Loss(original_values=input_data[0])],
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

model.save('models/sucker.keras')
print(f"Model deployment completed at time t={tm.time() - start}")

# %% Model inference

custom_objects = {"Excess20Loss": Excess20Loss}
model = keras.models.load_model('models/sucker.keras', custom_objects)

for curr in [0.0,0.25,0.5,0.75,1.0]:
    for gt in [0.0,0.25,0.5,0.75,1.0]:
        test_input = [np.expand_dims(np.array(1.), 0), 
                 np.expand_dims(np.array(1.), 0)]
        pred = model.predict(test_input)
        print(f"{curr}, {gt} -> {pred}")

print(f"Model inference completed at time t={tm.time() - start}")

# %% Model eval
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

