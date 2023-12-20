import tensorflow as tf
import time as tm
from simulator.AgentGenerator import AgentGenerator
from simulator.Octopus import Octopus
from simulator.RandomSurface import RandomSurface

class OctoDatagen():
    """ Entry point for octopus datagen """
    def __init__(self, game_parameters: dict):
        assert game_parameters['num_iterations'] >= 0, "Error, number of iterations configured in game parameters are not compatible with data generation"
        self.game_parameters = game_parameters
        print(f"Instantiated OctDatagen with inference type {game_parameters['inference_mode']}")

    def run_color_datagen(self):
        GameParameters = self.game_parameters

        surf = RandomSurface(GameParameters)
        ag = AgentGenerator(GameParameters)
        ag.generate(num_agents=GameParameters['agent_number_of_agents'])
        octo=Octopus(GameParameters)
        octo.set_color(surf)

        start = tm.time()
        print(f"Octo datagen started at {start}, setting t=0.0")

        # %% Configure game
        surf = RandomSurface(GameParameters)
        ag = AgentGenerator(GameParameters)
        ag.generate(num_agents=GameParameters['agent_number_of_agents'])
        octo=Octopus(GameParameters)
        octo.set_color(surf)

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
                    sucker_state.append(s.c.r)
                    sucker_gt.append(s.get_surf_color_at_this_sucker(surf))

            # run inference using the selected mode
            octo.set_color(surf, GameParameters['inference_mode'])

            # log the test results (the output of inference)
            for l in octo.limbs:
                for s in l.suckers:
                    sucker_test.append(s.c.r)

        # Encapsulate data for use in training
        import getpass
        import socket
        import time
        metadata = {
            'datetime': time.time(),
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

