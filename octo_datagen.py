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
        print(f"Instantiated OctDatagen with inference type {game_parameters['inference_mode']}")

    def run_color_datagen(self):
        GameParameters = self.game_parameters
        if GameParameters['inference_mode'] == MLMode.SUCKER:
            model_path = GameParameters['sucker_model_location']
            model = keras.models.load_model(model_path)
        elif GameParameters['inference_mode'] == MLMode.LIMB:
            model_path = GameParameters['limb_model_location']
            model = keras.models.load_model(model_path)
        else:
            model_path = None
            model = None


        start = tm.time()
        print(f"Octo datagen started at {start}, setting t=0.0")

        # %% Configure game
        surf = RandomSurface(GameParameters)
        ag = AgentGenerator(GameParameters)
        ag.generate(num_agents=GameParameters['agent_number_of_agents'])
        octo=Octopus(GameParameters)
        octo.set_color(surf, inference_mode=GameParameters['inference_mode'], model = model)

        # %% Data Gen
        sucker_state = []
        sucker_gt = []
        sucker_test = []
        run_iterations = 0
        while run_iterations != GameParameters['num_iterations']:
            print(f'Datagen Iteration {run_iterations}')
            run_iterations += 1

            ag.increment_all(octo)
            octo.move(ag)

            for l in octo.limbs:
                for s in l.suckers:
                    # Different ML modes will want to capture different state info
                    if GameParameters['ml_mode'] == MLMode.SUCKER:
                        state = s.c.r
                    elif GameParameters['ml_mode'] == MLMode.LIMB:
                        radius = GameParameters['adjacency_radius']
                        state = {}
                        state['color'] = s.c.r
                        adjacents = []
                        for adj in l.find_adjacents(s, radius):
                            adjacents.append(adj)
                        state['adjacents'] = adjacents

                    sucker_state.append(state)
                    sucker_gt.append(s.get_surf_color_at_this_sucker(surf))

            # run inference using the selected mode
            octo.set_color(surf, GameParameters['inference_mode'])

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
            'game_parameters': GameParameters,
            'state_data': sucker_state,
            'gt_data': sucker_gt,
        }

        # this data is not used
        res = sucker_test

        print(f"Datagen completed at time t={tm.time() - start}")
        print(f"{len(res)} datapoints written")

        return data
