"""Octopus Data Generation Class"""
import os
import pickle
import sys
import time as tm
import getpass
import socket

# This module lives in the octopus_ai/ package but is also a runnable entry
# point (`python octopus_ai/datagen.py`). Put the repo root on sys.path so the
# `octopus_ai.*` and sibling-package imports resolve when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from simulator.agent_generator import AgentGenerator
from simulator.octopus_generator import Octopus
from simulator.surface_generator import RandomSurface
from simulator.simutil import Color, MLMode, MovementMode, InferenceLocation


class OctoDatagen():
    """ Entry point for octopus datagen """
    def __init__(self, game_parameters):
        """Takes a Config or a legacy flat params dict."""
        from octopus_ai.config import as_config
        cfg = as_config(game_parameters)
        assert cfg.run.num_iterations >= 0, (
            "Error, number of iterations configured in game parameters "
            "are not compatible with data generation"
        )
        # The snapshot stored in the generated payload is whatever the
        # caller handed us, so a flat-dict caller gets a flat dict back.
        self.game_parameters = game_parameters
        self.cfg = cfg
        self.data_write_mode = cfg.datagen.write_format
        self.inference_mode = cfg.inference.mode
        # The config resolves the path; no MLMode-keyed lookup here.
        self.model_path = cfg.inference_model_path
        # imported lazily to avoid a circular import: a module-level import
        # of training.models.model_loader runs training/__init__, which
        # imports the trainers, which import this module
        from training.models.model_loader import ModelLoader
        self.model = ModelLoader(self.model_path).get_object()
        print(
            f"Instantiated OctDatagen with inference type "
            f"\n\t{self.inference_mode} \nand model \n\t{self.model_path}"
        )

    def run_color_datagen(self):
        cfg = self.cfg

        start = tm.time()
        print(f"Octo datagen started at {start}, setting t=0.0")

        # %% Configure game
        surf = RandomSurface(cfg)
        ag = AgentGenerator(cfg)
        ag.generate(num_agents=cfg.agents.count)
        octo = Octopus(cfg)

        # Always start with no-model
        octo.set_color(surf, inference_mode=MLMode.NO_MODEL, model=self.model)

        # %% Data Gen
        sucker_state = []
        sucker_gt = []
        sucker_test = []
        run_iterations = 0
        # Without periodic randomization, the color feedback loop converges
        # to the surface within a few iterations and nearly every recorded
        # sample has state == gt, so the model never sees the mismatched
        # (previous color, surface color) pairs it must handle right after
        # the octopus moves. Interval N re-randomizes all sucker colors
        # every N iterations (0 disables).
        randomize_interval = cfg.datagen.randomize_colors_interval

        while run_iterations != cfg.run.num_iterations:
            print(f'Datagen Iteration {run_iterations}')

            if randomize_interval > 0 and \
                    run_iterations % randomize_interval == 0:
                for limb in octo.limbs:
                    for s in limb.suckers:
                        v = float(np.random.rand())
                        s.c = Color(v, v, v)

            run_iterations += 1

            ag.increment_all(octo)
            octo.move(ag)

            for limb in octo.limbs:
                for s in limb.suckers:
                    # Different ML modes will want to capture different
                    # state info
                    if self.data_write_mode == MLMode.SUCKER:
                        state = s.c.r
                    elif self.data_write_mode == MLMode.LIMB:
                        radius = cfg.octopus.sucker.adjacency_radius
                        state = {}
                        state['color'] = s.c.r
                        state['adjacents'] = limb.find_adjacents(s, radius)

                    sucker_state.append(state)
                    sucker_gt.append(s.get_surf_color_at_this_sucker(surf))

            # run inference using the selected mode
            octo.set_color(surf, self.inference_mode, self.model)

            # log the test results (the output of inference)
            for limb in octo.limbs:
                for s in limb.suckers:
                    sucker_test.append(s.c.r)

        # Encapsulate data for use in training
        metadata = {
            'datetime': tm.time(),
            'user': getpass.getuser(),
            'machine': socket.gethostname(),
        }
        data = {
            'metadata': metadata,
            'game_parameters': self.game_parameters,
            'state_data': sucker_state,
            'gt_data': sucker_gt,
        }

        # this data is not used
        res = sucker_test

        print(f"Datagen completed at time t={tm.time() - start}")
        print(f"{len(res)} datapoints generated")

        return data


if __name__ == "__main__":

    default_params: dict = {
        # General game parameters 🎛️
        'num_iterations': 2,  # set this to -1 for infinite loop
        'x_len': 15,
        'y_len': 15,
        'rand_seed': 0,
        'debug_mode': False,  # enables things like agent attract/repel regions
        'save_images': False,
        'adjacency_radius': 1.0,  # determines what distance is
        # considered 'adjacent',
        'inference_mode': MLMode.SUCKER,
        'inference_model': MLMode.SUCKER,
        'inference_location': InferenceLocation.LOCAL,
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
        'octo_max_limb_offset': 0.5,  # used for attract/repel distance
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
        'octo_max_hue_change': 0.25,  # max val of rgb that can change at a
                                      # time, used as constraint threshold

    }

    datagen = OctoDatagen(default_params)
    synthetic_data = datagen.run_color_datagen()
    from octopus_ai.config import default_datasets
    f_location = default_datasets[MLMode.SUCKER]
    print("Writing generated data to", f_location)
    with open(f_location, 'wb') as file:
        pickle.dump(synthetic_data, file, pickle.HIGHEST_PROTOCOL)
