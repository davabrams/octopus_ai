import matplotlib.pyplot as plt 
from simulator.AgentGenerator import AgentGenerator
from OctoConfig import GameParameters
from simulator.Octopus import Octopus
from simulator.RandomSurface import RandomSurface
from simulator.simutil import print_setup, print_all
from simulator.simutil import MLMode

""" Entry point for octopus visualizer """

# %% Generate game scenario %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
surf = RandomSurface(GameParameters)
ag = AgentGenerator(GameParameters)
ag.generate(num_agents=GameParameters['agent_number_of_agents'])
octo=Octopus(GameParameters)
octo.set_color(surf)
model = None
if GameParameters['inference_mode'] == MLMode.SUCKER:
    from tensorflow import keras
    from util import ConstraintLoss
    custom_objects = {"ConstraintLoss": ConstraintLoss}
    model_path = GameParameters['sucker_model_location']
    model = keras.models.load_model(model_path, custom_objects)


# %% Visualizer %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig, ax = print_setup()

run_iterations = 0
while run_iterations != GameParameters['num_iterations']:
    run_iterations += 1

    print_all(ax, octo, ag, surf, GameParameters['debug_mode'])

    fig.canvas.draw()
    # if fig.waitforbuttonpress(timeout=10):
        # break
    plt.pause(0.1)

    ag.increment_all(octo)
    octo.move(ag)

    # run inference using the selected mode
    octo.set_color(surf, GameParameters['inference_mode'], model)
1
