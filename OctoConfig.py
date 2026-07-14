"""
Octopus game and ML parameters
"""
import os
from simulator.simutil import MLMode, InferenceLocation, MovementMode

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

default_models = {
    MLMode.SUCKER: os.path.join(ROOT_DIR, 'training/models/sucker.keras'),
    MLMode.LIMB: os.path.join(ROOT_DIR, 'training/models/limb.keras'),
    MLMode.FULL: None,  # Placeholder for future full model
    MLMode.NO_MODEL: None
}

default_datasets = {
    MLMode.SUCKER: os.path.join(ROOT_DIR, 'training/datagen/sucker.pkl'),
    MLMode.LIMB: os.path.join(ROOT_DIR, 'training/datagen/limb.pkl'),
    MLMode.FULL: None,  # Placeholder for future full model
    MLMode.NO_MODEL: None
}

GameParameters: dict = {
    # General game parameters 🎛️
    'num_iterations': 120,  # set this to -1 for infinite loop
    'x_len': 15,
    'y_len': 15,
    'surface_grayscale': True,  # random grayscale surface values in [0, 1);
                                # False for the classic binary 0/1 grid
    'rand_seed': 0,
    'debug_mode': False,  # enables things like agent attract/repel regions
    'log_forces': False,  # write per-frame body/limb/sucker forces to a
                          # local SQLite db (logs/forces.db); off by default
    'show_forces': False,  # draw force-vector arrows (body drift + per-arm
                           # attraction/tension) on the visualizer
    'save_images': False,
    'adjacency_radius': 1.0,  # determines what distance is considered
    # 'adjacent'
    'inference_location': InferenceLocation.LOCAL,
    'inference_mode': MLMode.SUCKER,
    'inference_model': MLMode.SUCKER,
    'datagen_data_write_format': MLMode.SUCKER,

    # Agent parameters 👾
    'agent_number_of_agents': 5,
    'agent_max_velocity': 0.2,
    'agent_max_theta': 0.1,
    'agent_movement_mode': MovementMode.RANDOM,
    'agent_range_radius': 5,

    # Octopus parameters 🐙
    'octo_max_body_velocity': 0.25,
    'octo_max_arm_theta': 0.1,  # used for random drift movement
    'octo_max_arm_reach_theta': 0.3,  # per-joint bend cap when reaching
                                  # toward prey / away from threats in
                                  # LUMPED_SPRING mode (separate from the
                                  # random-drift cap above)
    'octo_max_limb_offset': 0.5,  # used for attract/repel distance
    'octo_arm_stiffness': 0.5,  # spring stiffness for LUMPED_SPRING: arm
                                  # rest length is short (wants to retract);
                                  # prey attraction stretches it, and the
                                  # stretch tension both reels the tip back
                                  # and tugs the body toward the tip. Higher
                                  # = stubbier arms that yank the body harder;
                                  # lower = longer reach, gentler body pull.
    'octo_arm_rest_fraction': 0.3,  # neutral arm length as a fraction from
                                  # min (0.0, fully tucked) to max (1.0,
                                  # fully extended). Sits above min so a
                                  # threat can COMPRESS the arm below rest,
                                  # producing outward tension that flees the
                                  # threat; prey STRETCHES it above rest.
    'octo_num_arms': 8,
    'octo_max_sucker_distance': 0.3,
    'octo_min_sucker_distance': 0.1,
    'octo_movement_mode': MovementMode.RANDOM,
    'octo_threading': True,

    # Limb parameters 💪
    'limb_rows': 16,
    'limb_cols': 2,
    'limb_movement_mode': MovementMode.RANDOM,

    # Sucker parameters 🪠
    'octo_max_hue_change': 0.25,  # max val of rgb that can change at a time,
                                  # used as constraint threshold
    'sucker_delta_model': True,  # sucker net predicts a tanh-bounded
                                  # color DELTA (constraint enforced by
                                  # architecture); False = legacy direct
                                  # prediction with WeightedSumLoss
    'datagen_randomize_colors_interval': 2,  # re-randomize sucker colors
                                  # every N datagen iterations so training
                                  # data covers mismatched (color, surface)
                                  # pairs; 0 disables

    # Training hyperparams (used by trainers via GameParameters)
    'test_size': 0.2,
    'epochs': 50,
    'batch_size': 32,
    'constraint_loss_weight': 0.95,
}


TrainingParameters = {
    # ML training & datagen parameters 🕸️
    'ml_mode': MLMode.SUCKER,

    # Datagen
    "save_data_to_disk": True,
    "restore_data_from_disk": False,
    'datagen_mode': True,

    # Tensorboard
    "erase_old_tensorboard_logs": True,
    "generate_tensorboard": True,

    # Model save and restore
    'training_model': MLMode.SUCKER,
    "save_model_to_disk": True,
    "restore_model_from_disk": False,

    "run_training": True,

    # Test & Eval
    "run_inference": False,
    "run_eval": False,

    # Model paths
    'models': default_models,

    # Generated-dataset (pickle) paths, used by save_data_to_disk /
    # restore_data_from_disk
    'datasets': default_datasets,

    # Training hyperparams
    'test_size': 0.2,
    'epochs': 50,
    'batch_size': 32,  # 32 is tf default
    'constraint_loss_weight': 0.95,
}
