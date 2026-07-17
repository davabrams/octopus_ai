"""Agent Generator"""
import numpy as np

from octopus_ai.config import as_config
from simulator.simutil import MovementMode, AgentType, Agent


class AgentGenerator:
    """
    Generates an agent at a location.
    Default is a random agent type at a random location.
    """
    def __init__(self, params) -> None:
        # instance attribute (a class-level list would be shared
        # across all AgentGenerator instances in the process)
        self.agents: list[Agent] = []
        cfg = as_config(params)
        np.random.seed(seed=cfg.run.rand_seed)
        self._x_len = cfg.world.x_len
        self._y_len = cfg.world.y_len
        self.max_velocity = cfg.agents.max_velocity
        self.max_theta = cfg.agents.max_theta
        self.movement_mode = cfg.agents.movement_mode
        # How far an AGENT senses the octopus. The octopus's own sensing
        # radius is octopus.sensing_radius - a separate knob now.
        self.range_radius = cfg.agents.sensing_radius
        self.prey_capture_radius = cfg.agents.prey_capture_radius
        self.respawn_captured_prey = cfg.agents.respawn_captured_prey
        # running tally of prey captured over this generator's lifetime
        self.prey_captured = 0

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

    def place_agents(self, positions):
        """Place agents at explicit positions.

        positions: list of dicts, each with keys x, y, type ("prey"|"predator").
        """
        for p in positions:
            agent_type = (AgentType.PREY if p.get("type", "prey") == "prey"
                          else AgentType.THREAT)
            t = np.random.uniform(0, 2 * np.pi)
            new_agent = Agent(float(p["x"]), float(p["y"]), t, 0.0, 0.0, 0.0,
                              agent_type)
            self.agents.append(new_agent)

    # Modes where agents react to the octopus. Which spring model the
    # octopus's arms use (lumped vs chain) is irrelevant to the agents, so
    # both map to the same reactive path.
    REACTIVE_MODES = (MovementMode.LUMPED_SPRING, MovementMode.SPRING_CHAIN)

    def increment_all(self, octo=None, visibility=0.0):
        """Advance all agents one step.

        visibility: the octopus's current visibility score (0 = invisible,
        higher = more visible).  In the visibility-gated REACTIVE_MODES this
        controls how strongly agents pursue or flee — a well-camouflaged octopus
        is effectively invisible, so agents wander randomly. PURSUIT_FLEE
        ignores it (always reacts at full strength inside the sense window).
        """
        if self.movement_mode == MovementMode.RANDOM:
            self.agents = [self._increment_random(agent)
                           for agent in self.agents]
        elif (self.movement_mode in self.REACTIVE_MODES
              or self.movement_mode == MovementMode.PURSUIT_FLEE):
            if not octo:
                assert False, ("agent movement mode reacts to the octopus "
                               "but no octopus object was passed")
            # Gather sucker positions ONCE per frame rather than per agent.
            sucker_xy = np.array(
                [[s.x, s.y] for limb in octo.limbs for s in limb.suckers],
                dtype=float,
            )
            # PURSUIT_FLEE reacts regardless of camouflage: force full reaction
            # weight so agents commit to pursuing/fleeing inside the sense
            # window. The reactive spring modes stay visibility-gated.
            react_vis = (1.0 if self.movement_mode == MovementMode.PURSUIT_FLEE
                         else visibility)
            self.agents = [self._increment_reactive(agent, sucker_xy,
                                                    react_vis)
                           for agent in self.agents]
        else:
            assert False, f"Unknown agent movement mode: {self.movement_mode}"

    def _clamp_to_grid(self, agent: Agent) -> None:
        """Keep an agent inside the play area.

        Directed movement (fleeing especially) will otherwise walk an agent
        off the grid and it never comes back.
        """
        agent.x = float(min(max(agent.x, 0.0), self._x_len - 1.0))
        agent.y = float(min(max(agent.y, 0.0), self._y_len - 1.0))

    def remove_captured_prey(self, octo) -> int:
        """Remove PREY agents touched by any of the octopus's suckers.

        "Touched" means a sucker is within prey_capture_radius of the prey.
        Suckers (not the body centre) are the test because they are the
        octopus's actual physical extent - this is what makes an arm reaching
        out and contacting prey the thing that catches it.

        THREAT agents are never captured. Returns the number captured this
        call; also accumulates into self.prey_captured. When
        respawn_captured_prey is set, an equal number of fresh prey are
        generated so a run does not simply run out of food.
        """
        if not self.agents or self.prey_capture_radius <= 0 or octo is None:
            return 0

        prey = [a for a in self.agents if a.agent_type == AgentType.PREY]
        if not prey:
            return 0

        # All sucker positions once, as an (n, 2) array - vectorizing keeps
        # this cheap even at 256 suckers x several agents per frame.
        sucker_xy = np.array(
            [[s.x, s.y] for limb in octo.limbs for s in limb.suckers],
            dtype=float,
        )
        if sucker_xy.size == 0:
            return 0

        survivors = []
        captured = 0
        for agent in self.agents:
            if agent.agent_type != AgentType.PREY:
                survivors.append(agent)
                continue
            dists = np.hypot(sucker_xy[:, 0] - agent.x,
                             sucker_xy[:, 1] - agent.y)
            if float(dists.min()) <= self.prey_capture_radius:
                captured += 1
            else:
                survivors.append(agent)

        if captured:
            self.agents = survivors
            self.prey_captured += captured
            if self.respawn_captured_prey:
                self.generate(num_agents=captured,
                              fixed_agent_type=AgentType.PREY)
        return captured

    def _increment_random(self, agent: Agent) -> Agent:
        """Random walk: step by the current velocity, then re-roll it.

        The velocity ranges are SYMMETRIC about zero. They used to be
        uniform(0, max_velocity), which is never negative - so every agent
        drifted monotonically +x/+y and eventually piled into the top-right
        corner instead of wandering.
        """
        new_agent = agent
        new_agent.update_kinematics()

        new_agent.vx = np.random.uniform(-self.max_velocity,
                                         self.max_velocity)
        new_agent.vy = np.random.uniform(-self.max_velocity,
                                         self.max_velocity)
        new_agent.w = np.random.uniform(-self.max_theta * np.pi,
                                        self.max_theta * np.pi)
        self._clamp_to_grid(new_agent)
        return new_agent

    def _increment_reactive(self, agent: Agent, sucker_xy,
                            visibility: float) -> Agent:
        """Agent reaction to the octopus: PREY flee it, THREATs hunt it.

        Agents sense the NEAREST SUCKER rather than the body centre: the
        suckers are the octopus's actual physical extent, so a prey flees
        the arm reaching for it (and it is a sucker that captures it),
        while a threat closes on whatever part is nearest.

        The strength of the reaction is proportional to the octopus's
        visibility: at visibility 0 the octopus is perfectly camouflaged
        and agents wander randomly; as visibility rises agents react more
        purposefully.  Visibility is clamped to [0, 1] to produce a blend
        weight.

        Beyond range_radius the octopus hasn't been noticed, so the agent
        just wanders (same as RANDOM mode). Shared by every reactive mode
        (see REACTIVE_MODES) - the agent doesn't care whether the octopus's
        arms use the lumped spring or the spring chain.
        """
        if sucker_xy.size == 0:
            return self._increment_random(agent)

        dx = sucker_xy[:, 0] - agent.x
        dy = sucker_xy[:, 1] - agent.y
        dists = np.hypot(dx, dy)
        nearest = int(np.argmin(dists))
        dist = float(dists[nearest])

        # Out of sensing range -> hasn't noticed the octopus; wander.
        if dist > self.range_radius:
            return self._increment_random(agent)

        # How much the agent reacts vs wanders: visibility clamped to [0,1].
        react_weight = float(np.clip(visibility, 0.0, 1.0))

        # Effectively invisible -> pure random walk.
        if react_weight < 1e-6:
            return self._increment_random(agent)

        # Degenerate: sitting exactly on a sucker. Prey bolts in a random
        # direction; a threat has nothing to close on, so it just wanders.
        if dist < 1e-9:
            if agent.agent_type == AgentType.PREY:
                theta = np.random.uniform(0, 2 * np.pi)
                ux, uy = np.cos(theta), np.sin(theta)
            else:
                return self._increment_random(agent)
        else:
            # unit vector from the agent toward the nearest sucker
            ux, uy = dx[nearest] / dist, dy[nearest] / dist
            if agent.agent_type == AgentType.PREY:
                ux, uy = -ux, -uy  # flee: point away

        # Blend directed movement with random noise based on visibility.
        rand_theta = np.random.uniform(0, 2 * np.pi)
        rx, ry = np.cos(rand_theta), np.sin(rand_theta)
        bx = react_weight * ux + (1.0 - react_weight) * rx
        by = react_weight * uy + (1.0 - react_weight) * ry
        mag = np.hypot(bx, by)
        if mag > 1e-9:
            bx /= mag
            by /= mag

        agent.vx = float(bx * self.max_velocity)
        agent.vy = float(by * self.max_velocity)
        agent.t = float(np.arctan2(by, bx))
        agent.w = 0.0
        agent.update_kinematics()
        self._clamp_to_grid(agent)
        return agent
