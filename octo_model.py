import tensorflow as tf
import numpy as np
import time as tm
from AgentGenerator import AgentGenerator
from OctoDatagen import OctoDatagen
from Octopus import Octopus
from RandomSurface import RandomSurface
from util import MLMode, MovementMode
from OctoConfig import GameParameters

""" Entry point for octopus modeling """

start = tm.time()
print(f"Octo Model started at {start}, setting t=0.0")

# %% Data Gen
datagen = OctoDatagen(GameParameters)
data = datagen.run_color_datagen()

# %% Model training
import tensorflow as tf
from tensorflow import keras
from util import train_test_split

input_data = np.array(data['state_data'])
label_data = np.array(data['gt_data'])

train_data, train_labels, val_data, val_labels = train_test_split(input_data, label_data)
input_shape = np.size(train_data)

model = keras.Sequential([
    keras.layers.Dense(5, activation="relu", input_shape=(1,)),
    keras.layers.Dense(5, activation="relu"),
    keras.layers.Dense(1, activation="softmax")
])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_data, train_labels, epochs=GameParameters['epochs'], batch_size=GameParameters['batch_size'])

loss, accuracy = model.evaluate(val_data, val_labels)
print(f"Loss: {loss}, Accuracy: {accuracy}")

pass
print(f"Model training completed at time t={tm.time() - start}")

# %% Model deployment
pass
print(f"Model deployment completed at time t={tm.time() - start}")

# %% Model inference
pass
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

