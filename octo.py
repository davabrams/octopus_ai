import matplotlib.pyplot as plt 
import tensorflow as tf
import time as tm
from AgentGenerator import AgentGenerator, AgentType
from Octopus import Octopus
from RandomSurface import RandomSurface
from util import print_setup, print_all


GameParameters: dict = {
    'x_len': 10,
    'y_len': 12,
    'rand_seed': 0,
    'agent_max_velocity': 0.2,
    'agent_max_theta': 0.1,
    'octo_max_body_velocity': 0.3,
    'octo_max_arm_theta': 0.1,
    'octo_num_arms': 8,
    'octo_max_sucker_distance': 0.3,
    'octo_min_sucker_distance': 0.1,
    'octo_max_hue_change': 0.2, #max percentage of r, g, or b's total dynamic range that can change at a time
    'limb_rows': 16,
    'limb_cols': 2
    }

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if False:
    start = tm.time()

    x = tf.Variable(1.0)

    def f(x):
        y = x**2 + 2*x - 5
        return y

    print(f(1))


    end = tm.time()
    print(f"tensorflow took: {end - start:.4f} seconds")


# %% Generate game scenario %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
surf = RandomSurface(GameParameters)
ag = AgentGenerator(GameParameters)
ag.generate(num_agents=3)
octo=Octopus(GameParameters)
octo.set_color(surf)


# %% Visualizer %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig, ax = print_setup()
while True:
    print_all(ax, octo, ag, surf)

    fig.canvas.draw()
    # if fig.waitforbuttonpress(timeout=10):
        # break
    plt.pause(0.1 )

    ag.increment_all()
    octo.move()