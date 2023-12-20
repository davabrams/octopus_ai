""" Octopus visualizer """
import matplotlib.pyplot as plt
from OctoConfig import GameParameters
from simulator.agent_generator import AgentGenerator
from simulator.octopus import Octopus
from simulator.random_surface import RandomSurface
from simulator.simutil import print_setup, print_all, MLMode


# %% Generate game scenario %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
surf = RandomSurface(GameParameters)
ag = AgentGenerator(GameParameters)
ag.generate(num_agents=GameParameters['agent_number_of_agents'])
octo=Octopus(GameParameters)
octo.set_color(surf)

model = None
if GameParameters['inference_mode'] == MLMode.SUCKER:
    # Override `model` with the model from disk
    from tensorflow import keras
    from util import ConstraintLoss
    custom_objects = {"ConstraintLoss": ConstraintLoss}
    model_path = GameParameters['sucker_model_location']
    model = keras.models.load_model(model_path, custom_objects)


# %% Visualizer %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig, ax = print_setup()

i: int = 0
while i != GameParameters['num_iterations']:
    i += 1

    print_all(ax, octo, ag, surf, GameParameters['debug_mode'])

    fig.canvas.draw()
    # if fig.waitforbuttonpress(timeout=10) d:
        # break
    plt.pause(0.1)

    ag.increment_all(octo)
    octo.move(ag)

    # run inference using the selected mode and model
    octo.set_color(surf, GameParameters['inference_mode'], model)
