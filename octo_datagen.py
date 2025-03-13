"""Octopus Data Generation Class"""
import os
import pickle
import time as tm
import getpass
import socket
from training.models.model_loader import ModelLoader
from training.datagen.data_loader import DataLoader
from simulator.agent_generator import AgentGenerator
from simulator.octopus_generator import Octopus
from simulator.surface_generator import RandomSurface
from simulator.simutil import MLMode, MovementMode, InferenceLocation

class OctoDatagen():
    """ Entry point for octopus datagen """
    def __init__(self, game_parameters: dict):
        assert game_parameters['num_iterations'] >= 0, "Error, number of iterations configured in game parameters are not compatible with data generation"
        self.game_parameters = game_parameters
        self.data_write_mode = game_parameters['datagen_data_write_format']
        self.inference_mode = game_parameters['inference_mode']
        self.model_path = self.game_parameters['inference_model']
        self.model = ModelLoader(self.model_path).get_object()
        print(f"Instantiated OctDatagen with inference type \n\t{self.inference_mode} \nand model \n\t{self.model_path}")

    def run_color_datagen(self):
        params = self.game_parameters

        start = tm.time()
        print(f"Octo datagen started at {start}, setting t=0.0")

        # %% Configure game
        surf = RandomSurface(params)
        ag = AgentGenerator(params)
        ag.generate(num_agents=params['agent_number_of_agents'])
        octo=Octopus(params)

        # Always start with no-model
        octo.set_color(surf, inference_mode=MLMode.NO_MODEL, model = self.model) 

        # %% Data Gen
        sucker_state = []
        sucker_gt = []
        sucker_test = []
        run_iterations = 0
        while run_iterations != params['num_iterations']:
            print(f'Datagen Iteration {run_iterations}')
            run_iterations += 1

            ag.increment_all(octo)
            octo.move(ag)

            for l in octo.limbs:
                for s in l.suckers:
                    # Different ML modes will want to capture different state info
                    if self.data_write_mode == MLMode.SUCKER:
                        state = s.c.r
                    elif self.data_write_mode == MLMode.LIMB:
                        radius = params['adjacency_radius']
                        state = {}
                        state['color'] = s.c.r
                        state['adjacents'] = l.find_adjacents(s, radius)

                    sucker_state.append(state)
                    sucker_gt.append(s.get_surf_color_at_this_sucker(surf))

            # run inference using the selected mode
            octo.set_color(surf, self.inference_mode, self.model)

            # log the test results (the output of inference)
            for l in octo.limbs:
                for s in l.suckers:
                    sucker_test.append(s.c.r)

        # Encapsulate data for use in training
        metadata = {
            'datetime': tm.time(),
            'user': getpass.getuser(),
            'machine': socket.gethostname(),
        }
        data = {
            'metadata': metadata,
            'game_parameters': params,
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
    'num_iterations': 2, #set this to -1 for infinite loop
    'x_len': 15,
    'y_len': 15,
    'rand_seed': 0,
    'debug_mode': False, #enables things like agent attract/repel regions
    'save_images': False,
    'adjacency_radius': 1.0, #determines what distance is considered 'adjacent',
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

    # # Octopus parameters 🐙
    'octo_max_body_velocity': 0.25,
    'octo_max_arm_theta': 0.1, #used for random drift movement
    'octo_max_limb_offset': 0.5, #used for attract/repel distance
    'octo_num_arms': 8,
    'octo_max_sucker_distance': 0.3,
    'octo_min_sucker_distance': 0.1,
    'octo_movement_mode': MovementMode.RANDOM,
    'octo_threading': True,

    # # Limb parameters 💪
    'limb_rows': 16,
    'limb_cols': 2,
    'limb_movement_mode': MovementMode.RANDOM,

    # Sucker parameters 🪠
    'octo_max_hue_change': 0.25, #max val of rgb that can change at a time, 
                                 # used as constraint threshold

    }

    datagen = OctoDatagen(default_params)
    synthetic_data = datagen.run_color_datagen()
    f_location = DataLoader.defaults[MLMode.SUCKER]
    print("Writing default parameters to", f_location)
    try:
        with open(f_location, 'x') as file:
            print("Creating new file")
            file.write()
    except Exception as e:
        print("Error", e)
        print(os.getcwd())
        print(f_location)

    with open(f_location, 'wb+') as file:
        print("Appending to file")
        pickle.dump(synthetic_data, file, pickle.HIGHEST_PROTOCOL)