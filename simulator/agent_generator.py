"""Agent Generator"""
import numpy as np
from simulator.simutil import MovementMode, AgentType, Agent


class AgentGenerator:
    """
    Generates an agent at a location.
    Default is a random agent type at a random location.
    """
    agents: list[Agent] = []

    def __init__(self, params: dict) -> None:
        np.random.seed(seed=params['rand_seed'])
        self._x_len = params['x_len']
        self._y_len = params['y_len']
        self.max_velocity = params['agent_max_velocity']
        self.max_theta = params['agent_max_theta']
        self.movement_mode = params['agent_movement_mode']
        self.range_radius = params['agent_range_radius']

    def generate(self, num_agents: int = 1,
                 fixed_agent_type: AgentType = None):
        """Generates (a/some) new agent(s) with a random type if unspecified"""
        for _ in range(num_agents):
            if not fixed_agent_type:
                flip = np.random.randint(0, 2)
                if (flip == 0):
                    agent_type = AgentType.PREY
                else:
                    agent_type = AgentType.THREAT
            else:
                agent_type = fixed_agent_type
            x, y, t = (np.random.uniform(0, self._x_len - 1),
                       np.random.uniform(0, self._y_len - 1),
                       np.random.uniform(0, 2 * np.pi))
            vx, vy, vel_t = (np.random.uniform(0, self.max_velocity),
                                   np.random.uniform(0, self.max_velocity),
                                   np.random.uniform(0, self.max_velocity))

            new_agent = Agent(x, y, t, vx, vy, vel_t, agent_type)
            self.agents.append(new_agent)

    def increment_all(self, octo=None):
        if self.movement_mode == MovementMode.RANDOM:
            self.agents = [self._increment_random(agent)
                           for agent in self.agents]
        elif self.movement_mode == MovementMode.ATTRACT_REPEL:
            if not octo:
                assert False, ("movement mode set to attract/repel but no "
                              "octopus object passed")
            self.agents = [self._increment_attract_repel(agent, octo)
                           for agent in self.agents]

    def _increment_random(self, agent: Agent) -> Agent:
        # (1) move the agent forward in the direction it's facing by the
        # previous velocity
        # (2) pick a new velocity at random
        # (3) pivot a random angle

        new_agent = agent
        new_agent.update_kinematics()

        new_agent.vx = np.random.uniform(0, self.max_velocity)
        new_agent.vy = np.random.uniform(0, self.max_velocity)
        new_agent.w = (np.random.uniform(0, self.max_theta * np.pi)
                        % (2 * np.pi))
        return new_agent

    def _increment_attract_repel(self, agent: Agent, octo) -> Agent:
        return agent
