# %% Agent Generator
from enum import Enum
import numpy as np


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

    def generate(self, num_agents: int = 1, fixed_agent_type: AgentType = None):
        # Generates (a/some) new agent(s) with a random type if unspecified
        for _ in range(num_agents):
            if not fixed_agent_type:
                flip = np.random.randint(0, 2)
                print(flip)
                if (flip == 0):
                    agent_type = AgentType.PREY
                else:
                    agent_type = AgentType.THREAT
            else:
                agent_type = fixed_agent_type
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