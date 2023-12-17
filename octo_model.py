import tensorflow as tf
import numpy as np
import time as tm
from AgentGenerator import AgentGenerator
from OctoDatagen import OctoDatagen
from Octopus import Octopus
from RandomSurface import RandomSurface
from util import print_setup, print_all, MLMode, MovementMode

""" Entry point for octopus modeling """

start = tm.time()
print(f"Octo Model started at {start}, setting t=0.0")

GameParameters: dict = {
    # General game parameters
    'x_len': 15,
    'y_len': 15,
    'rand_seed': 0,
    'debug_mode': False, #enables things like agent attract/repel regions
    'num_iterations': 10, #set this to -1 for infinite loop

    # ML datagen parameters
    'inference_mode': MLMode.SUCKER,
    'datagen_mode': True,

    # Agent parameters
    'agent_number_of_agents': 5,
    'agent_max_velocity': 0.2,
    'agent_max_theta': 0.1,
    'agent_movement_mode': MovementMode.RANDOM,
    'agent_range_radius': 5,

    # Octopus parameters
    'octo_max_body_velocity': 0.25,
    'octo_max_arm_theta': 0.1, #used for random drift movement
    'octo_max_limb_offset': 0.5, #used for attract/repel distance
    'octo_num_arms': 8,
    'octo_max_sucker_distance': 0.3,
    'octo_min_sucker_distance': 0.1,
    'octo_max_hue_change': 0.2, #max percentage of r, g, or b's total dynamic range that can change at a time
    'octo_movement_mode': MovementMode.RANDOM,

    # Limb parameters
    'limb_rows': 16,
    'limb_cols': 2,
    'limb_movement_mode': MovementMode.RANDOM,
    }

# %% Configure game

# %% Data Gen
datagen = OctoDatagen(GameParameters)
data = datagen.run_color_datagen()
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

