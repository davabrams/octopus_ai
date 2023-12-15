import matplotlib.pyplot as plt
from AgentGenerator import AgentType


def print_setup():
    fig = plt.figure()
    fig.show()
    ax = fig.add_subplot(1,1,1)
    ax.set_xticks
    ax.set_xticks([]) 
    ax.set_xticks([]) 
    ax.set_title("Octopus AI Visualizer") 
    return fig, ax

def print_all(ax, octo, ag, surf):

    ax.clear()

    # Print the patterned surface
    ax.imshow(surf.grid.astype(float), cmap="binary_r") 

    # Print the octopus
    for limb in octo.limbs:
        for sucker in limb.suckers:
            ax.plot(sucker.x, sucker.y, marker='.', markersize = 10, mfc=sucker.c.to_rgb(), mec=[0.5, 0.5, 0.5], lw=1)
                
        for c_row in range(len(limb.center_line) - 1):
            pt_1 = limb.center_line[c_row]
            pt_2 = limb.center_line[c_row + 1]
            ax.plot([pt_1.x, pt_2.x], [pt_1.y, pt_2.y], color = 'brown')

    # Print the agents
    for agent in ag.agents:
        color: str = 'violet'
        if agent.Type == AgentType.PREY:
            color = 'lightgreen'                
        ax.plot(agent.x, agent.y, marker='o', color=color) 