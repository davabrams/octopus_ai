import tensorflow as tf
import numpy as np
import time as tm

""" Entry point for TF modeling """

start = tm.time()
print(f"Datagen started at {start}, setting t=0.0")

# %% Data Gen
pass
print(f"Datagen completed at time t={tm.time() - start}")

# %% Model training
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

