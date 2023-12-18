import matplotlib.pyplot as plt 
from AgentGenerator import AgentGenerator
from OctoConfig import GameParameters
from Octopus import Octopus
from RandomSurface import RandomSurface
from util import print_setup, print_all

""" Entry point for octopus visualizer """

# %% Generate game scenario %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
surf = RandomSurface(GameParameters)
ag = AgentGenerator(GameParameters)
ag.generate(num_agents=GameParameters['agent_number_of_agents'])
octo=Octopus(GameParameters)
octo.set_color(surf)


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
    octo.set_color(surf, GameParameters['inference_mode'])
1
