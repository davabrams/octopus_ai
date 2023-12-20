"""
Octopus game and ML parameters
"""
from simulator.simutil import MLMode, MovementMode

GameParameters: dict = {
    # General game parameters
    'x_len': 15,
    'y_len': 15,
    'rand_seed': 0,
    'debug_mode': True, #enables things like agent attract/repel regions
    'num_iterations': 100, #set this to -1 for infinite loop

    # ML datagen parameters
    'inference_mode': MLMode.NO_MODEL,
    'sucker_model_location': 'models/sucker.keras',
    'datagen_mode': True,
    'test_size': 0.2,
    'epochs': 8,
    'batch_size': 8,

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
    'octo_max_hue_change': 0.25, #max val of r, g, or b's total dynamic range that can change at a time
    'octo_movement_mode': MovementMode.RANDOM,

    # Limb parameters
    'limb_rows': 16,
    'limb_cols': 2,
    'limb_movement_mode': MovementMode.RANDOM,
    }
