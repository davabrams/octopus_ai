import tensorflow as tf
import numpy as np
import time as tm

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
# %%
start = tm.time()

x = tf.Variable(1.0)

def f(x):
  y = x**2 + 2*x - 5
  return y

print(f(1))


end = tm.time()
print(f"tensorflow took: {end - start:.4f} seconds")
# %% Patterned Surface Generator

class RandomSurface:
    # Generates a random surface of size
    # 2*x_half_len, 2*y_half_len
    # filled with 0 (black) and 1 (white)
    def __init__(self, GameParameters: dict) -> None:
        x_len = GameParameters['x_len']
        y_len = GameParameters['y_len']
        rand_seed = GameParameters['rand_seed']
        assert x_len > 0, f"x must be >0"
        assert y_len > 0, f"y must be >0"
        
        np.random.seed(seed = rand_seed)
        
        self._x_len = x_len
        self._y_len = y_len
        self.grid = np.random.randint(2, size=(self._y_len, self._x_len), dtype=np.int8)
        
    def get_val(self, x: float, y: float) -> int:
        # Gets the value 0 (black) and 1 (white) at any location
        # within the boundary of the grid
        assert x > 0 and x < self._x_len, f"x must be between 0 and {self._x_len}"
        assert y > 0 and y < self._y_len, f"y must be between 0 and {self._y_len}"
        
        return self.grid[int(x)][int(y)]
        
        
surf = RandomSurface(GameParameters)


# %% Agent Generator
from enum import Enum



class AgentType(Enum):
    PREY = 0
    THREAT = 1
    
    
class Agent:
    x: float = 0
    y: float = 0
    vel: float = 0
    t: float = 0
    Type: AgentType = None
    
    def __repr__(self):
        return f"<Agent\n\tType: {self.Type}, \n\tLoc: ({self.x}, {self.y}), \n\tVel: {self.vel}, \n\tTheta = {self.t}>\n"
    
    
class AgentGenerator:
    # Generates an agent at a location.
    # Default is a random agent type at a random location.
    
    agents: list[Agent] = []

    def __init__(self, GameParameters: dict) -> None:
        np.random.seed(seed = GameParameters['rand_seed'])
        self._x_len = GameParameters['x_len']
        self._y_len = GameParameters['y_len']
        self.max_velocity = GameParameters['agent_max_velocity']
        self.max_theta = GameParameters['agent_max_theta']

    def generate(self, agent_type: AgentType = None):
        # Generates a new agent with a random type if unspecified
        if not agent_type:
            flip = np.random.randint(0, 2)
            if (flip == 0):
                agent_type = AgentType.PREY
            else:
                agent_type = AgentType.THREAT
        
        new_agent = Agent()
        new_agent.Type = agent_type
        new_agent.x = np.random.uniform(0, self._x_len)
        new_agent.y = np.random.uniform(0, self._y_len)
        new_agent.vel = np.random.uniform(0, self.max_velocity)
        new_agent.t = np.random.uniform(0, 2 * np.pi)
        self.agents.append(new_agent)
        
    def increment_all(self):
        self.agents = [self._increment(agent) for agent in self.agents]
            
    def _increment(self, agent: Agent) -> Agent:
        new_agent = agent
        new_agent.x = new_agent.x + new_agent.vel * np.cos(new_agent.t)
        new_agent.x = min(max(new_agent.x, 0), self._x_len)
        new_agent.y = new_agent.y + new_agent.vel * np.sin(new_agent.t)
        new_agent.y = min(max(new_agent.y, 0), self._y_len)
        
        new_agent.vel = np.random.uniform(0, self.max_velocity)
        new_agent.t = (new_agent.t + np.random.uniform(0, self.max_theta * np.pi)) % (2 * np.pi)
        return new_agent
    
a = AgentGenerator(GameParameters)
a.generate()
a.generate()
a.generate()
print(a.agents)
a.increment_all()
print(a.agents)
a.increment_all()
print(a.agents)
a.increment_all()
print(a.agents)


# %% Octopus Class
from dataclasses import dataclass, field
import matplotlib


@dataclass
class CenterPoint:
    x: float = 0
    y: float = 0
    t: float = 0

@dataclass
class Color:
    r: float = 0.5
    g: float = 0.5
    b: float = 0.5
    
    def to_rgb(self):
        return [self.r, self.g, self.b]
    
@dataclass
class Sucker:
    x: float
    y: float
    c: Color = field(default_factory=Color)
    max_hue_change: float = 0.25
    
    def __repr__(self):
        return "S:{" + str(self.x) + ", " + str(self.y) + "}"
    
    def set_color(self, surf: RandomSurface):
        x_grid_location = int(round(self.x))
        y_grid_location = int(round(self.y))
        c_val = surf.grid[y_grid_location][x_grid_location] * 1.0
        self.c.r = self.find_color_change(self.c.r, c_val)
        self.c.g = self.find_color_change(self.c.g, c_val)
        self.c.b = self.find_color_change(self.c.b, c_val)
        
    
    def find_color_change(self, c_start, c_target):
        d_max = self.max_hue_change
        dc = c_target - c_start
        dc = min(dc, d_max)
        dc = max(dc, -d_max)
        
        new_c = c_start + dc
        new_c = min(new_c, 1.0)
        new_c = max(new_c, 0.0)
        return new_c


class Limb:
    suckers = []
    
    def __init__(self, x_octo: float, y_octo: float, init_angle: float, GameParameters: dict):
        self.max_sucker_distance = GameParameters['octo_max_sucker_distance']
        self.min_sucker_distance = GameParameters['octo_min_sucker_distance']
        self.sucker_distance = self.min_sucker_distance
        self.rows = GameParameters['limb_rows']
        self.cols = GameParameters['limb_cols']
        self.center_line = [CenterPoint() for _ in range(self.rows)]
        self.x_len = GameParameters['x_len']
        self.y_len = GameParameters['y_len']
        self.max_hue_change = GameParameters['octo_max_hue_change']


        """" generate the initial sucker positions within the arm"""
        self.gen_centerline(x_octo, y_octo, init_angle)
        self.refresh_sucker_locations()

    def gen_centerline(self, x_octo: float, y_octo: float, init_angle):
        """" given an initial angle and some distance parameters, calculate a
        starting center line for octopus sucker locations and angles"""
        for row in range(self.rows):
            # add 1 to leave room for octopus center
            # generates a horizontal row of sukers and then rotates it
            # that's why there is no y_init component used in the calculation
            x_init = (1 + row) * self.min_sucker_distance
            x_prime = x_init * np.cos(init_angle) + x_octo
            y_prime = x_init * np.sin(init_angle) + y_octo
            self.center_line[row] = CenterPoint(x_prime, y_prime, init_angle)

    def refresh_sucker_locations(self):
        """ given a center line with x,y,theta, recalculate individual sucker
        locations """
        if not self.suckers:
            #the very first time we hit this, generate all sucker objects
            self.suckers = [Sucker(0, 0) for _ in range(self.cols * self.rows)]
            
        
        for row in range(self.rows):
            pt = self.center_line[row]
            x = pt.x
            y = pt.y
            t = pt.t
            t += np.pi / 2

            for col in range(self.cols):
                col_offset = (col - ((self.cols - 1) / 2))
                offset = self.sucker_distance * col_offset

                x_prime = x + offset * np.cos(t)
                y_prime = y + offset * np.sin(t)
                
                x_prime = max(x_prime, -0.5)
                x_prime = min(x_prime, self.x_len - 0.51)
                y_prime = max(y_prime, -0.5)
                y_prime = min(y_prime, self.y_len - 0.51)
                
                self.suckers[row + self.rows * col].x = x_prime
                self.suckers[row + self.rows * col].y = y_prime
                
                
    def drift(self, x_octo: float, y_octo: float):
        """randomly shift thetas, reconstruct centerline, and refresh sucker 
        locations """
        
        # TODO(davabrams): instead of doing it this way, adjust the final 
        # centerpoint to move towards prey and away from threats, and then
        # adjust the rest of the center points accordingly as a spline or 
        # something
        self.sucker_distance += np.random.uniform(-.05, .05)
        self.sucker_distance = max(self.sucker_distance, self.min_sucker_distance)
        self.sucker_distance = min(self.sucker_distance, self.max_sucker_distance)
        
        for row in range(self.rows):
            pt = self.center_line[row]
            t = pt.t
            t += np.random.uniform(-.1, .1)
            
            x_prime = x_octo + self.sucker_distance * np.cos(t)
            y_prime = y_octo + self.sucker_distance * np.sin(t)
            self.center_line[row].x = x_prime
            self.center_line[row].y = y_prime
            self.center_line[row].t = t
            x_octo = x_prime
            y_octo = y_prime
        self.refresh_sucker_locations()
        
    def set_color(self, surf: RandomSurface):
        for sucker in self.suckers:
            sucker.set_color(surf)

class Octopus:
    def __init__(self, GameParameters: dict):
        self.x = GameParameters['x_len'] / 2.0
        self.y = GameParameters['y_len'] / 2.0

        self.max_body_velocity = GameParameters['octo_max_body_velocity']
        self.max_arm_theta = GameParameters['octo_max_arm_theta']
        self.num_arms = GameParameters['octo_num_arms']

        self.limbs = [
            Limb(
                self.x,
                self.y,
                float(ix/self.num_arms * 2 * np.pi),
                GameParameters
                )
            for ix in range(self.num_arms)
        ]
        
    def move(self):
        self.x += np.random.uniform(-0.25, 0.25)
        self.y += np.random.uniform(-0.25, 0.25)
        for l in self.limbs:
            l.drift(self.x, self.y)

    def set_color(self, surf: RandomSurface):
        for l in self.limbs:
            l.set_color(surf)

octo=Octopus(GameParameters)

# %% Visualizer
# stolen from https://www.geeksforgeeks.org/chess-board-using-matplotlib-python/

import numpy as np 
import matplotlib.pyplot as plt 
plt.clf()

# Print the patterned surface
res = surf.grid
bg = plt.imshow(res.astype(float), cmap="binary_r") 

octo.set_color(surf)

# Print the octopus
for limb in octo.limbs:
    for sucker in limb.suckers:
        plt.plot(sucker.x, sucker.y, marker='.', markersize = 10, mfc=sucker.c.to_rgb(), mec=[0.5, 0.5, 0.5], lw=1)
            
    for c_row in range(len(limb.center_line) - 1):
        pt_1 = limb.center_line[c_row]
        pt_2 = limb.center_line[c_row + 1]
        plt.plot([pt_1.x, pt_2.x], [pt_1.y, pt_2.y], color =
                 'brown')


# Print the agents
for agent in a.agents:
    color: str = ''
    if agent.Type == AgentType.PREY:
        color = 'lightgreen'
    else:
        color = 'violet'
    plt.plot(agent.x, agent.y, marker='o', color=color) 
    

plt.xticks([]) 
plt.yticks([]) 
plt.title("Octopus AI Visualizer") 
plt.show() 
octo.move()



