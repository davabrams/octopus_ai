""" Octopus visualizer """
import time    
import matplotlib.pyplot as plt
from OctoConfig import GameParameters
from simulator.agent_generator import AgentGenerator
from simulator.octopus import Octopus
from simulator.random_surface import RandomSurface
from simulator.simutil import setup_display, display_refresh, MLMode

# %% Generate game scenario %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_agents = GameParameters['agent_number_of_agents']
model_path = GameParameters['sucker_model_location']

ag = AgentGenerator(GameParameters)
octo = Octopus(GameParameters)
surf = RandomSurface(GameParameters)
ag.generate(num_agents=num_agents)
octo.set_color(surf)

model = None
if GameParameters['inference_mode'] is not MLMode.NO_MODEL:
    # Override `model` with the model from disk
    from tensorflow import keras
    from training.losses import ConstraintLoss
    custom_objects = {"ConstraintLoss": ConstraintLoss}
    model = keras.models.load_model(model_path, custom_objects)


# %% Visualizer %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
inference_mode = GameParameters['inference_mode']
NUM_ITERATIONS = GameParameters['num_iterations']
debug_mode = GameParameters['debug_mode']
SAVE_IMAGES = GameParameters['save_images']

fig, ax = setup_display()
font = {'family':'monospace','color':'green','size':20}
y = fig.text(.1, .025, "", fontdict=font)

i: int = 0
while i != NUM_ITERATIONS:
    i += 1

    display_refresh(ax, octo, ag, surf, debug_mode=debug_mode)

    y.set_text(f"Visibility = {octo.visibility(surf):.4f}")
    fig.canvas.draw()
    # if fig.waitforbuttonpress(timeout=-1):
    #     break
    if SAVE_IMAGES:
        plt.savefig(f'foo{time.time()}.png')
    print(f"Iteration: {i}")
    plt.pause(0.1)
    ag.increment_all(octo)
    octo.move(ag)

    # run inference using the selected mode and model
    octo.set_color(surf, inference_mode=inference_mode, model=model)
