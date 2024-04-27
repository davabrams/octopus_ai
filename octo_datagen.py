"""Octopus Data Generation Class"""
import time as tm
import getpass
import socket
from tensorflow import keras
from simulator.agent_generator import AgentGenerator
from simulator.octopus import Octopus
from simulator.random_surface import RandomSurface
from simulator.simutil import MLMode

class OctoDatagen():
    """ Entry point for octopus datagen """
    def __init__(self, game_parameters: dict):
        assert game_parameters['num_iterations'] >= 0, "Error, number of iterations configured in game parameters are not compatible with data generation"
        self.game_parameters = game_parameters
        self.data_write_mode = game_parameters['datagen_data_write_format']
        self.inference_mode = game_parameters['inference_mode']
        self.model_path = self.game_parameters['models'][self.inference_mode]
        print(f"Instantiated OctDatagen with inference type {self.inference_mode} and model {self.model_path}")

    def run_color_datagen(self):
        params = self.game_parameters
        model = keras.models.load_model(self.model_path) if self.model_path else None

        start = tm.time()
        print(f"Octo datagen started at {start}, setting t=0.0")

        # %% Configure game
        surf = RandomSurface(params)
        ag = AgentGenerator(params)
        ag.generate(num_agents=params['agent_number_of_agents'])
        octo=Octopus(params)

        # Always start with no-model
        octo.set_color(surf, inference_mode=MLMode.NO_MODEL, model = model) 

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
            octo.set_color(surf, self.inference_mode, model)

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
        print(f"{len(res)} datapoints written")

        return data
