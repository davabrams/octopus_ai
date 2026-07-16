"""Octopus Class"""
from typing import List
import tensorflow as tf
from tensorflow import keras
from multiprocessing.pool import ThreadPool
import numpy as np
from simulator.surface_generator import RandomSurface
from simulator.simutil import (
    MovementMode,
    Agent,
    AgentType,
    Color,
    MLMode,
    CenterPoint,
    agent_influence_vector,
    convert_adjacents_to_ragged_tensor
)
from simulator.spring_chain import solve_chain, base_reaction
from simulator.ilqr.arm import ArmController
from octopus_ai.config import as_config

class Sucker:
    """
    Stores location and color of a sucker, and includes methods to change the sucker color.
    This object is instantiated by Limb objects.
    """

    def __init__(self, x: float, y: float, c: Color = Color(), params = None):
        self.x = x
        self.y = y
        self.c = c
        self.prev: "Sucker" = None
        if params is not None:
            self.max_hue_change = as_config(params).octopus.sucker.\
                max_hue_change
        else:
            self.max_hue_change = float(0.25)

    def __repr__(self):
        return "S:{" + str(self.x) + ", " + str(self.y) + "}"

    def force_color(self, c: Color):
        """
        Forces the sucker's color
        """
        assert isinstance(c, Color)
        self.c = c

    def find_color(self, *args):
        # surf, inference_mode, model, adjacents, ix):
        """ 
        Sets the sucker's new color using either a heuristic or an ML model. 
        """
        surf, inference_mode, model, adjacents, ix = args[0]
        
        if inference_mode is not MLMode.NO_MODEL:
            assert model is not None, "model inference specified but no model was specified"
            assert isinstance(model, keras.Model), f"Expected keras model, got {type(model)}"
        c_val = self.get_surf_color_at_this_sucker(surf)
        c_ret = Color()

        if inference_mode == MLMode.NO_MODEL:
            c_ret.r = self._find_color_change(self.c.r, c_val.r)
        elif inference_mode == MLMode.SUCKER:
            c_ret.r = model.predict(np.array([[self.c.r, c_val.r]]), verbose=0)[0][0]
        elif inference_mode == MLMode.LIMB:
            fixed_test_input = np.array([[self.c.r, c_val.r]])
            ragged_test_input = convert_adjacents_to_ragged_tensor(adjacents)
            c_ret.r = model.predict([fixed_test_input, ragged_test_input])[0][0]
        c_ret.g = c_ret.r
        c_ret.b = c_ret.r
        return c_ret, ix

    def set_loc(self, x: float, y: float):
        """
        Sets the sucker location and iterates it
        """
        self.prev = self
        self.prev.prev = None #Prevents unbound memory
        self.x = x
        self.y = y

    def get_surf_color_at_this_sucker(self, surf: RandomSurface) -> Color:
        """Gets the RGB color of the surface underneath the sucker"""
        x_grid_location = int(round(self.x))
        y_grid_location = int(round(self.y))
        rgb = surf.get_val(x_grid_location, y_grid_location)
        return Color(float(rgb[0]), float(rgb[1]), float(rgb[2]))

    def _find_color_change(self, c_start: float, c_target: float):
        d_max = self.max_hue_change
        dc = c_target - c_start
        dc = min(dc, d_max)
        dc = max(dc, -d_max)

        new_c = c_start + dc
        new_c = min(new_c, 1.0)
        new_c = max(new_c, 0.0)
        return new_c

    def distance_to(self, s: "Sucker"):
        """
        Finds the distance from this sucker to another one
        Uses euclidean distance
        """
        x = s.x - self.x
        y = s.y - self.y
        dist = np.sqrt(x * x + y * y, dtype=float)
        return dist


class Limb:
    """
    Stores limb spline, and includes methods to change the limb spline.
    This class is instantiated by Octopus objects, and instantiates Sucker objects.
    """

    def __init__(self, x_octo: float, y_octo: float, init_angle: float,
                 params):
        """params may be a Config or a legacy flat params dict."""
        cfg = as_config(params)
        limb = cfg.octopus.limb

        self.suckers: list[Sucker] = []
        self.max_sucker_distance = limb.max_sucker_distance
        self.min_sucker_distance = limb.min_sucker_distance
        self.sucker_distance = self.min_sucker_distance
        self.rows = limb.rows
        self.cols = limb.cols
        self.center_line = [CenterPoint() for _ in range(self.rows)]
        self.x_len = cfg.world.x_len
        self.y_len = cfg.world.y_len
        self.max_hue_change = cfg.octopus.sucker.max_hue_change
        self.movement_mode = limb.movement_mode

        # Mode-specific knobs, read from the block that owns them.
        self.max_arm_theta = limb.random.max_arm_theta
        self.max_arm_reach_theta = limb.lumped.max_arm_reach_theta
        self.max_limb_offset = limb.lumped.max_limb_offset
        self.arm_stiffness = limb.lumped.arm_stiffness
        self.arm_rest_fraction = limb.lumped.arm_rest_fraction
        self.chain_spring_k = limb.chain.spring_k
        self.chain_agent_k = limb.chain.agent_k
        self.chain_move_k = limb.chain.move_k
        self.ilqr_cfg = limb.ilqr

        # How close two suckers must be to count as neighbours for the LIMB
        # model; datagen builds its training adjacents with this.
        self.adjacency_radius = cfg.octopus.sucker.adjacency_radius
        # How far THIS ARM senses agents. Distinct from the agent's own
        # sensing radius, which it used to be forced to equal.
        self.agent_range_radius = cfg.octopus.sensing_radius
        self.threading = cfg.run.threading

        # Last-frame force capture (populated by _move_lumped_spring).
        # Shared source of truth for on-screen arrows and the force DB.
        # f_attract: prey/threat pull at the tip; f_spring: restoring force;
        # net: their sum; tension: base reaction fed to the body.
        self.last_f_attract = np.zeros(2, dtype=float)
        self.last_f_spring = np.zeros(2, dtype=float)
        self.last_net = np.zeros(2, dtype=float)
        self.last_tension = np.zeros(2, dtype=float)
        self.last_arm_length = 0.0

        # iLQR motor-control state (MovementMode.ILQR). Each limb owns its own
        # controller and warm-start controls, functioning autonomously of the
        # others (ARCHITECTURE.md §11.4). Built lazily on first move so the
        # compile cost is only paid by limbs that actually use iLQR.
        self._ilqr_controller = None
        self._ilqr_u = None            # warm-start control sequence
        # horizon/iters/weights come from self.ilqr_cfg; body_stiffness is read
        # out here because the base-reaction path uses it every frame.
        self.ilqr_body_stiffness = limb.ilqr.body_stiffness  # how hard this
        # arm's reach/flee tugs the body (feeds _drift_body_by_tension)

        # Base-ring geometry (body rotation plan). Each limb's base is pinned to
        # a fixed angular slot on a ring around the body, so the arms fan out
        # from DISTINCT roots (never collocating at the center). base_angle is
        # the limb's fixed offset; the body's orientation rotates the whole ring.
        self.base_angle = float(init_angle)
        self.ring_radius = cfg.octopus.ring_radius
        # Offset of this arm's base from the body center last frame (the moment
        # arm the body integrates into torque); rebound each _move_ilqr.
        self.last_base_offset = np.zeros(2, dtype=float)

        # Opt-in per-iteration iLQR recording (record & replay). When on, each
        # _move_ilqr solve captures its solve history + per-solve metadata for
        # the recorder to drain between frames; off = zero overhead.
        self.record_ilqr_history = cfg.output.record_ilqr_history
        self.last_ilqr_history = None  # list[ILQRIterationRecord] | None
        self.last_ilqr_meta = None     # dict | None, per-solve metadata

        """" generate the initial sucker positions within the arm"""
        self._gen_centerline(x_octo, y_octo, init_angle)
        self._refresh_sucker_locations()

    def _gen_centerline(self, x_octo: float, y_octo: float, init_angle):
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

    def _refresh_sucker_locations(self):
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
                col_offset = col - ((self.cols - 1) / 2)
                offset = self.sucker_distance * col_offset

                x_prime = x + offset * np.cos(t)
                y_prime = y + offset * np.sin(t)

                x_prime = max(x_prime, -0.5)
                x_prime = min(x_prime, self.x_len - 0.51)
                y_prime = max(y_prime, -0.5)
                y_prime = min(y_prime, self.y_len - 0.51)

                self.suckers[row + self.rows * col].set_loc(x_prime, y_prime)

    def move(self, x_octo: float, y_octo: float,
             agents: list = None, coordinated_influence=None,
             body_theta: float = 0.0, body_dtheta: float = 0.0):
        """Move the limb, then refresh sucker locations.

        x_octo, y_octo: the (possibly just-moved) body position; the arm
            base is anchored here.
        body_theta: the body's orientation, which rotates this arm's base
            around the ring (ILQR mode; see _move_ilqr).
        body_dtheta: how much the body just rotated this frame; the arm is
            carried rigidly with it (its free nodes rotate about the body
            center by this) so rotation adds no spurious strain (ILQR mode).
        agents: list of Agent objects the arm may sense (LUMPED_SPRING).
        coordinated_influence: optional (dx, dy) supplied by the Octopus in
            full-body mode so arms share a target rather than each chasing
            independently. When None (limb mode), the arm senses agents on
            its own.
        """
        if self.movement_mode == MovementMode.RANDOM:
            self._move_random(x_octo, y_octo)
        elif self.movement_mode == MovementMode.LUMPED_SPRING:
            self._move_lumped_spring(
                x_octo, y_octo, agents, coordinated_influence)
        elif self.movement_mode == MovementMode.SPRING_CHAIN:
            self._move_spring_chain(x_octo, y_octo, agents)
        elif self.movement_mode == MovementMode.ILQR:
            self._move_ilqr(x_octo, y_octo, agents, body_theta, body_dtheta)
        else:
            assert False, "Unknown movement mode"

        self._refresh_sucker_locations()

    def _move_random(self, x_octo: float, y_octo: float):
        self.sucker_distance += np.random.uniform(-.05, .05)
        self.sucker_distance = max(self.sucker_distance, self.min_sucker_distance)
        self.sucker_distance = min(self.sucker_distance, self.max_sucker_distance)

        for row in range(self.rows):
            pt = self.center_line[row]
            t = pt.t
            t += np.random.uniform(-self.max_arm_theta, self.max_arm_theta)

            x_prime = x_octo + self.sucker_distance * np.cos(t)
            y_prime = y_octo + self.sucker_distance * np.sin(t)
            self.center_line[row].x = x_prime
            self.center_line[row].y = y_prime
            self.center_line[row].t = t
            x_octo = x_prime
            y_octo = y_prime

    def _rest_length(self) -> float:
        """The arm's spring rest (neutral) length.

        Sits a fraction (arm_rest_fraction) of the way from the fully-tucked
        minimum to the fully-extended maximum, so the arm can BOTH stretch
        above rest (reaching prey) and compress below rest (recoiling from a
        threat). If rest were the minimum, threats could never compress the
        arm and the body would never flee.
        """
        n = len(self.center_line) - 1
        min_len = self.min_sucker_distance * n
        max_len = self.max_sucker_distance * n
        return min_len + self.arm_rest_fraction * (max_len - min_len)

    def _max_length(self) -> float:
        """Fully-stretched arm length (segments at max_sucker_distance)."""
        return self.max_sucker_distance * (len(self.center_line) - 1)

    def tension_vector(self):
        """Signed spring tension reaction at the base - the ONLY thing the
        body feels from this arm.

        Magnitude is stiffness * |length - rest|; the sign follows real
        spring mechanics:

        - Arm STRETCHED past rest (reaching prey): tension pulls the base
          toward the tip. Since a prey-seeking tip points at the prey, the
          body is drawn toward the prey.
        - Arm COMPRESSED below rest (recoiling from a threat): the spring
          pushes the base away from the tip. A threat pulls the tip inward
          toward the threat side, so "away from tip" is "away from the
          threat" - the body flees. This is why threat avoidance needs no
          separate term: it emerges from the same base tension (per the
          "only the base connection moves the base" model).

        Returned as an (dx, dy) numpy vector; ~zero at rest length.
        """
        base = self.center_line[0]
        tip = self.center_line[-1]
        dx, dy = tip.x - base.x, tip.y - base.y
        length = float(np.hypot(dx, dy))
        if length < 1e-9:
            return np.zeros(2, dtype=float)
        signed_stretch = length - self._rest_length()  # + stretch, - compress
        # unit vector base -> tip; scaling by signed_stretch flips direction
        # automatically under compression (pushes base away from tip)
        return self.arm_stiffness * signed_stretch * np.array(
            [dx / length, dy / length])

    def _move_lumped_spring(self, x_octo: float, y_octo: float,
                            agents: list = None,
                            coordinated_influence=None):
        """Spring-tension reach (overdamped).

        The arm is a spring with a short rest length. Two forces act on the
        tip: prey/threat attraction pulls it out, and spring tension
        (stiffness * stretch, directed tip -> base) reels it back. The tip
        moves to the overdamped balance of the two, which sets the new arm
        length (hence sucker_distance). With no attraction, tension wins and
        the arm retracts toward rest length. The base stays pinned to the
        body; the chain is reflowed with a per-joint bend cap.
        """
        n = len(self.center_line)

        # 1) Anchor the base to the (already-updated) body position.
        base = self.center_line[0]
        base.x = x_octo
        base.y = y_octo

        # 2) Attraction at the tip (or the shared body influence in
        #    coordinated full-octopus mode).
        tip = self.center_line[-1]
        if coordinated_influence is not None:
            f_attract = np.asarray(coordinated_influence, dtype=float)
        else:
            f_attract = agent_influence_vector(
                tip.x, tip.y, agents, self.agent_range_radius)

        # 3) Spring restoring force at the tip: magnitude stiffness * stretch,
        #    directed from tip back toward base (i.e. -[base->tip]).
        dx, dy = tip.x - base.x, tip.y - base.y
        length = float(np.hypot(dx, dy))
        rest = self._rest_length()
        if length > 1e-9:
            axis = np.array([dx / length, dy / length])
        else:
            # degenerate: use the base heading so the arm has a direction
            axis = np.array([np.cos(base.t), np.sin(base.t)])
        stretch = length - rest
        f_spring = -self.arm_stiffness * stretch * axis  # reels tip inward

        # 4) Overdamped tip update: step by the net force, capped so the tip
        #    can't jump more than max_limb_offset in one step.
        net = f_attract + f_spring
        nm = float(np.hypot(net[0], net[1]))
        if nm > 1e-9:
            step = min(self.max_limb_offset, nm)
            target_x = tip.x + step * net[0] / nm
            target_y = tip.y + step * net[1] / nm
        else:
            target_x, target_y = tip.x, tip.y

        # 5) New arm length -> new segment spacing (sucker_distance).
        #    The arm may compress below rest (threat recoil, tucking in) down
        #    to its fully-tucked minimum, or stretch up to max. The min floor
        #    is the compact length, not rest, so threats can shorten the arm.
        min_len = self.min_sucker_distance * (n - 1)
        new_len = float(np.hypot(target_x - base.x, target_y - base.y))
        new_len = min(max(new_len, min_len), self._max_length())
        seg_len = new_len / (n - 1) if n > 1 else new_len
        seg_len = min(max(seg_len, self.min_sucker_distance),
                      self.max_sucker_distance)
        self.sucker_distance = seg_len

        # 6) Reflow the chain base -> tip: each joint aims at the tip target
        #    but turns at most max_arm_reach_theta from the previous heading,
        #    and sits seg_len from its parent.
        prev_x, prev_y = base.x, base.y
        prev_angle = base.t
        for i in range(1, n):
            desired_angle = np.arctan2(target_y - prev_y, target_x - prev_x)
            dtheta = (desired_angle - prev_angle + np.pi) % (2 * np.pi) - np.pi
            dtheta = max(-self.max_arm_reach_theta,
                         min(self.max_arm_reach_theta, dtheta))
            angle = prev_angle + dtheta

            new_x = prev_x + seg_len * np.cos(angle)
            new_y = prev_y + seg_len * np.sin(angle)

            new_x = min(max(new_x, -0.5), self.x_len - 0.51)
            new_y = min(max(new_y, -0.5), self.y_len - 0.51)

            self.center_line[i].x = new_x
            self.center_line[i].y = new_y
            self.center_line[i].t = angle

            prev_x, prev_y, prev_angle = new_x, new_y, angle

        # Capture this frame's forces for logging / visualization.
        self.last_f_attract = np.asarray(f_attract, dtype=float).copy()
        self.last_f_spring = np.asarray(f_spring, dtype=float).copy()
        self.last_net = np.asarray(net, dtype=float).copy()
        self.last_tension = self.tension_vector().copy()
        self.last_arm_length = float(new_len)

    def _move_spring_chain(self, x_octo: float, y_octo: float,
                           agents: list = None):
        """Mass-spring chain solved by ONE direct linear solve per frame.

        Each centerline node is a sucker mass; node 0 is the base, pinned to
        the body. Free nodes 1..n feel: neighbor springs (chain_spring_k),
        a prey spring-to-target (chain_agent_k, tip-weighted linear ramp),
        a threat constant push, and a movement-cost anchor to their
        start-of-frame position (chain_move_k, uniform). All forces are
        linear, so equilibrium is K x = f (K symmetric positive-definite);
        solve_chain does it in one shot. This becomes iterative only when
        forces go nonlinear.

        The base reaction (spring between base and node 1) is stored as the
        arm's tension for the body to follow, mirroring LUMPED_SPRING so the
        two modes share body dynamics.
        """
        # Pin the base to the body.
        base = self.center_line[0]
        base.x = x_octo
        base.y = y_octo
        base_xy = np.array([base.x, base.y], dtype=float)

        free = self.center_line[1:]
        n = len(free)
        if n == 0:
            return

        # Start-of-frame positions (the movement-cost anchor targets).
        prev_xy = np.array([[pt.x, pt.y] for pt in free], dtype=float)

        # Per-node prey target (NaN where none) and threat push (0 where
        # none), sensed at each node's own position within its range.
        targets = np.full((n, 2), np.nan, dtype=float)
        threat_force = np.zeros((n, 2), dtype=float)
        if agents:
            for i, pt in enumerate(free):
                nearest_prey = None
                nearest_d = None
                for ag in agents:
                    d = np.hypot(ag.x - pt.x, ag.y - pt.y)
                    if d > self.agent_range_radius:
                        continue
                    if ag.agent_type == AgentType.PREY:
                        # closest prey becomes this node's spring target
                        if nearest_d is None or d < nearest_d:
                            nearest_d = d
                            nearest_prey = ag
                    else:  # THREAT: constant push away, weighted by proximity
                        if d > 1e-9:
                            w = 1.0 - (d / self.agent_range_radius)
                            ux = (pt.x - ag.x) / d
                            uy = (pt.y - ag.y) / d
                            threat_force[i, 0] += (self.chain_agent_k * w * ux)
                            threat_force[i, 1] += (self.chain_agent_k * w * uy)
                if nearest_prey is not None:
                    targets[i] = (nearest_prey.x, nearest_prey.y)

        new_xy = solve_chain(
            base_xy, prev_xy, targets, threat_force,
            self.chain_spring_k, self.chain_agent_k, self.chain_move_k)

        # Clamp to the grid and write back; set a heading for each node so
        # _refresh_sucker_locations orients the sucker pairs sensibly.
        prev_px, prev_py = base.x, base.y
        for i, pt in enumerate(free):
            nx = min(max(float(new_xy[i, 0]), -0.5), self.x_len - 0.51)
            ny = min(max(float(new_xy[i, 1]), -0.5), self.y_len - 0.51)
            pt.x = nx
            pt.y = ny
            pt.t = float(np.arctan2(ny - prev_py, nx - prev_px))
            prev_px, prev_py = nx, ny

        # Base reaction -> the body's tension signal for this arm.
        node1_xy = np.array([free[0].x, free[0].y], dtype=float)
        tension = base_reaction(base_xy, node1_xy, self.chain_spring_k)

        # Capture for logging / visualization (reuse the same attributes).
        self.last_tension = np.asarray(tension, dtype=float).copy()
        self.last_f_attract = np.zeros(2, dtype=float)
        self.last_f_spring = np.zeros(2, dtype=float)
        self.last_net = np.asarray(tension, dtype=float).copy()
        self.last_arm_length = float(
            np.hypot(free[-1].x - base.x, free[-1].y - base.y))

    def _ilqr_target(self, tip, agents):
        """The reach target for this arm's tip this frame.

        Nearest sensed PREY within the arm's range as [x, y], or None when no
        prey is in range. Sensing is measured from the arm's TIP, so an
        extended arm reaches prey well beyond the body's own sensing radius -
        which is exactly why the body must follow the ARM's reach rather than
        re-sensing from its centre (see _move_ilqr). Threats are handled
        separately by _ilqr_nearest_threat.
        """
        if not agents:
            return None
        nearest = None
        nearest_d = None
        for ag in agents:
            if ag.agent_type != AgentType.PREY:
                continue
            d = np.hypot(ag.x - tip.x, ag.y - tip.y)
            if d > self.agent_range_radius:
                continue
            if nearest_d is None or d < nearest_d:
                nearest_d = d
                nearest = ag
        return [nearest.x, nearest.y] if nearest is not None else None

    def _ilqr_nearest_threat(self, tip, agents):
        """Nearest sensed THREAT within the arm's range as [x, y], or None.

        Fed to the iLQR repulsion cost so the arm bends away from it.
        """
        if not agents:
            return None
        nearest = None
        nearest_d = None
        for ag in agents:
            if ag.agent_type != AgentType.THREAT:
                continue
            d = np.hypot(ag.x - tip.x, ag.y - tip.y)
            if d > self.agent_range_radius:
                continue
            if nearest_d is None or d < nearest_d:
                nearest_d = d
                nearest = ag
        return [nearest.x, nearest.y] if nearest is not None else None

    def _move_ilqr(self, x_octo: float, y_octo: float, agents: list = None,
                   body_theta: float = 0.0, body_dtheta: float = 0.0):
        """Per-limb iLQR reach, receding-horizon (MPC) style.

        Each frame the arm re-plans a short trajectory toward its target with
        its own compiled iLQR controller (ARCHITECTURE.md §11.4), applies just
        the first planned step, and warm-starts next frame from the shifted
        plan. The base (centerline node 0) is pinned to this arm's point on the
        BODY RING - a fixed angular slot (base_angle) rotated by body_theta,
        ring_radius out from the body center - so the arms fan out from distinct
        roots instead of all sharing the center. The free centerline nodes are
        the optimizer's decision variables. Springs hold the chain near its rest
        spacing, an effort cost keeps motion economical, and a tip attractor
        pulls toward nearby prey.
        """
        n_free = self.rows - 1
        if n_free < 1:
            return

        if self._ilqr_controller is None:
            ic = self.ilqr_cfg
            self._ilqr_controller = ArmController(
                n_free=n_free,
                rest_length=self.max_sucker_distance,
                horizon=ic.horizon,
                max_iters=ic.max_iters,
                w_spring=ic.w_spring,
                w_bend=ic.w_bend,
                w_effort=ic.w_effort,
                w_reach_run=ic.w_reach_run,
                w_reach_terminal=ic.w_reach_terminal,
                w_repel=ic.w_repel,
                repel_radius=ic.repel_radius,
            )

        # Carry the arm rigidly with the body's rotation this frame: rotate the
        # free nodes about the body center by body_dtheta BEFORE warm-starting.
        # Without this, rotating theta moves each base tangentially while the
        # warm start stays in world coords - a spurious strain that feeds back
        # into more torque and spins the body away (a runaway limit cycle).
        # Carried rigidly, rotation adds no strain, so the body only rotates
        # from genuine off-axis reaching (rotate-to-face, then settle).
        if body_dtheta != 0.0:
            cos_d, sin_d = np.cos(body_dtheta), np.sin(body_dtheta)
            for i in range(1, self.rows):
                px = self.center_line[i].x - x_octo
                py = self.center_line[i].y - y_octo
                self.center_line[i].x = x_octo + cos_d * px - sin_d * py
                self.center_line[i].y = y_octo + sin_d * px + cos_d * py

        # Pin the base to this arm's point on the body ring: a fixed angular
        # slot (base_angle) rotated by the body's orientation, ring_radius out
        # from the body center. R=0 reproduces the legacy single-point base.
        psi = self.base_angle + body_theta
        base_x = x_octo + self.ring_radius * np.cos(psi)
        base_y = y_octo + self.ring_radius * np.sin(psi)
        self.last_base_offset = np.array([base_x - x_octo, base_y - y_octo],
                                         dtype=float)
        base = self.center_line[0]
        base.x = base_x
        base.y = base_y
        x0 = np.array([[self.center_line[i].x, self.center_line[i].y]
                       for i in range(1, self.rows)],
                      dtype=np.float32).reshape(-1)

        tip = self.center_line[-1]
        prey = self._ilqr_target(tip, agents)      # [x, y] or None
        threat = self._ilqr_nearest_threat(tip, agents)
        # The arm reaches toward its prey; with none, it holds at the tip.
        solve_target = prey if prey is not None else [tip.x, tip.y]

        # Snapshot the warm-start controls BEFORE the solve: the np.roll below
        # builds a fresh array, so this reference stays stable for recording.
        u_init = self._ilqr_u

        res = self._ilqr_controller.solve(
            base_xy=[base_x, base_y], target=solve_target, x0=x0,
            threat=threat, u_init=self._ilqr_u,
            record_history=self.record_ilqr_history)

        # Receding horizon: apply only the first planned step (x_traj[1]).
        new_free = tf.reshape(res.x_traj[1], (-1, 2)).numpy()
        prev_x, prev_y = x_octo, y_octo
        for i in range(1, self.rows):
            nx = min(max(float(new_free[i - 1, 0]), -0.5), self.x_len - 0.51)
            ny = min(max(float(new_free[i - 1, 1]), -0.5), self.y_len - 0.51)
            self.center_line[i].x = nx
            self.center_line[i].y = ny
            self.center_line[i].t = float(np.arctan2(ny - prev_y, nx - prev_x))
            prev_x, prev_y = nx, ny

        # Warm-start next frame: shift the plan forward one step, pad with zero.
        u_np = res.u_traj.numpy()
        self._ilqr_u = np.roll(u_np, -1, axis=0)
        self._ilqr_u[-1] = 0.0

        # Drain solve history + metadata for the recorder (rebound every frame;
        # the recorder must read them between octo.move() and the next frame).
        if self.record_ilqr_history:
            self.last_ilqr_history = res.history
            self.last_ilqr_meta = {
                "base_xy": (float(base_x), float(base_y)),
                "target": (float(solve_target[0]), float(solve_target[1])),
                # Without target_kind an idle frame (holding at its own tip)
                # looks like the arm "reaching" its own tip.
                "target_kind": "prey" if prey is not None else "hold",
                "threat": (None if threat is None
                           else (float(threat[0]), float(threat[1]))),
                "threat_active": threat is not None,
                "x0": x0,                 # (2*n_free,) float32
                "u_init": u_init,         # (horizon, 2*n_free) float32 | None
                "iterations": res.iterations,
                "converged": res.converged,
                "final_cost": float(res.cost),
            }

        # Base reaction: the ONLY thing the body feels is the tensile force in
        # the spring between the base (node 0, pinned to the body) and its
        # adjacent node (node 1). The body is never influenced by agents
        # directly - prey/threats act on the arm's centerline nodes (the tip
        # attractor pulls toward prey, the repulsion barrier pushes off
        # threats), and the inter-node springs carry that as tension down the
        # chain to node 1. When the arm strains toward far prey the whole chain
        # stretches, so this first segment is stretched and its tension pulls
        # the body toward node 1 (up the arm, toward the prey); when the arm
        # recoils from a threat, node 1 is shoved clear and the tension pulls
        # the body after it. At rest the segment sits at its rest length and the
        # body feels nothing. Summed across arms by _drift_body_by_tension.
        # Tensile only: the segment PULLS the body when stretched (a taut arm
        # reaching prey / recoiling from a threat) but does not PUSH when
        # compressed - it is a rope, not a rod. This matters: an idle arm
        # settles a hair under rest length, and a two-sided spring would turn
        # that tiny compression into a persistent inward shove that, once the
        # body drifts and the arms trail into alignment, runs away into a
        # wander. Clamping to tension (stretch > 0) leaves an idle octopus
        # still, and only a genuinely straining arm moves the body.
        node1 = self.center_line[1]
        dx, dy = node1.x - base_x, node1.y - base_y  # base<->node1 (ring base)
        seg_len = float(np.hypot(dx, dy))
        rest = self.max_sucker_distance  # the iLQR segment rest length
        stretch = seg_len - rest
        if seg_len > 1e-9 and stretch > 0.0:
            self.last_tension = self.ilqr_body_stiffness * stretch \
                * np.array([dx / seg_len, dy / seg_len])
        else:
            self.last_tension = np.zeros(2, dtype=float)
        self.last_f_attract = np.asarray(self.last_tension, dtype=float).copy()
        self.last_f_spring = np.zeros(2, dtype=float)
        self.last_net = np.asarray(self.last_tension, dtype=float).copy()
        tip = self.center_line[-1]
        self.last_arm_length = float(np.hypot(tip.x - x_octo, tip.y - y_octo))

    def find_adjacents(self, s_target: Sucker, radius: float):
        """
        Finds all suckers within a specified radius
        """
        adjacents = []
        for s in self.suckers:
            dist = s.distance_to(s_target)
            if dist <= radius:
                adjacents.append(tuple((s, dist)))
        return adjacents

    def find_color(self,
                   surf: RandomSurface,
                   inference_mode: MLMode,
                   model
                    ):
        """
        Finds the color of the suckers in this limb, in parallel
        """
        pool = ThreadPool()

        find_color_parameter_list = [(surf, inference_mode, model,
                self.find_adjacents(s, self.adjacency_radius)
                if inference_mode == MLMode.LIMB
                else None,
                ix,) for ix, s in enumerate(self.suckers)]
        pool_iterable = zip(self.suckers, find_color_parameter_list)
        color_tuple_array = pool.imap_unordered(
            func = lambda x: x[0].find_color(x[1]),
            iterable = pool_iterable)
        color_tuple_array_sorted = sorted(color_tuple_array, key = lambda x: x[1])
        color_array_sorted = [c[0] for c in color_tuple_array_sorted]
        return color_array_sorted

    def force_color(self,
                    color_array: List[Color]):
        """
        Directly changes the color of the suckers in the limb to a specified Color in a list of colors
        """
        assert len(color_array) == len(self.suckers), "The length of the color list does not match the length of the sucker list"
        assert isinstance(color_array, list)
        assert isinstance(color_array[0], Color)
        for ix, s in enumerate(self.suckers):
            s.force_color(color_array[ix])


class Octopus:
    """
    Instantiate this object by passing in a game parameter configuration.
    This class stores the octopus's position.
    move(Agent): moves the octopus "body", and then calls upon it's child 
                    Limb objects' movement methods.
    set_color(RandomSurface, MLMode, model): passes its set_color parameters
                    to its child Limb object's set_color() function
    """
    def __init__(self, params):
        """params may be a Config or a legacy flat params dict."""
        cfg = as_config(params)
        self.x = cfg.world.x_len / 2.0
        self.y = cfg.world.y_len / 2.0
        self.max_body_velocity = cfg.octopus.max_body_velocity
        self.movement_mode = cfg.octopus.movement_mode
        self.model = None
        self.threading = cfg.run.threading

        # Body ORIENTATION (body rotation plan): the angular twin of (x, y).
        # The base ring rotates with theta, and theta integrates the summed arm
        # torque so the fan can turn (the "top" limb is not always on top).
        self.theta = 0.0
        self.ring_radius = cfg.octopus.ring_radius
        self.max_body_angular_velocity = cfg.octopus.max_body_angular_velocity
        self.body_torque_gain = cfg.octopus.limb.ilqr.body_torque_gain

        # Last-frame body force capture (populated by _move_lumped_spring):
        # last_body_force is the summed arm tension; last_body_drift is the
        # capped displacement actually applied to the body this frame.
        self.last_body_force = np.zeros(2, dtype=float)
        self.last_body_drift = np.zeros(2, dtype=float)
        self.last_body_torque = 0.0   # summed arm torque last frame
        self.last_body_dtheta = 0.0   # capped rotation actually applied

        num_arms = cfg.octopus.num_arms
        self.limbs = [
            Limb(
                self.x,
                self.y,
                float(ix/num_arms * 2 * np.pi),
                cfg  # already parsed; Limb's as_config passes it through
                )
            for ix in range(num_arms)
        ]

    def move(self, ag: Agent = None):
        """
        Moves the octopus using different movement modes.
        RANDOM: randomly moves the octopus in the x and y dimension
        LUMPED_SPRING: randomly moves the octopus, but considers agents.
            - Attracted to prey
            - Repelled by threats
        """
        agent_modes = (MovementMode.LUMPED_SPRING, MovementMode.SPRING_CHAIN,
                       MovementMode.ILQR)
        if self.movement_mode in agent_modes and not ag:
            assert False, ("movement mode needs agents but no agent object "
                           "was passed")

        agents = ag.agents if ag is not None else None
        coordinated_influence = None

        if self.movement_mode == MovementMode.RANDOM:
            self.x += np.random.uniform(-self.max_body_velocity, self.max_body_velocity)
            self.y += np.random.uniform(-self.max_body_velocity, self.max_body_velocity)
        elif self.movement_mode in agent_modes:
            # These modes share body dynamics: drift by summed arm base tension.
            # For LUMPED_SPRING the tension is computed live; SPRING_CHAIN and
            # ILQR each stored their base reaction in last_tension on the
            # previous frame. First frame (all-zero) just doesn't move.
            coordinated_influence = self._drift_body_by_tension()
        else:
            assert False, "Unknown movement mode"

        for l in self.limbs:
            l.move(self.x, self.y, agents, coordinated_influence,
                   body_theta=self.theta, body_dtheta=self.last_body_dtheta)

    def _drift_body_by_tension(self):
        """Drift the body by the summed base tension of its arms.

        Shared by both spring modes (pure-tension coupling, no direct
        attraction): the body only moves because its arms are straining.

        - LUMPED_SPRING: each arm's live tension_vector() (signed stretch/
          compression along base->tip).
        - SPRING_CHAIN / ILQR: each arm's stored base reaction from last frame
          (last_tension) - the base<->tip spring strain the arm reported after
          its own solve. Same one-step lag as LUMPED_SPRING: arms replan after
          this call, the body chases the reported strain, converges over the
          loop. This is how eight independent per-limb controllers reconcile
          into one body trajectory - the body is a passive mass integrating the
          vector sum of their base reactions, no central negotiation.

        Relaxed arms contribute ~nothing; arms reaching hard tug strongly.
        The body eases along the summed tension, capped by max_body_velocity.
        Returns None (arms sense agents at their own nodes; full-octopus
        coordination can later pass a shared influence).
        """
        stored_tension_modes = (MovementMode.SPRING_CHAIN, MovementMode.ILQR)
        total = np.zeros(2, dtype=float)
        torque = 0.0
        for limb in self.limbs:
            if self.movement_mode in stored_tension_modes:
                f = np.asarray(limb.last_tension, dtype=float)
            else:
                f = np.asarray(limb.tension_vector(), dtype=float)
            total += f
            # Torque about the body center: the tangential component of each
            # arm's base reaction applied at its ring point (moment arm r_i).
            # r_i is zero for modes that don't sit on the ring, so they add no
            # rotation. This is the ANGULAR twin of the linear tension sum.
            r = np.asarray(limb.last_base_offset, dtype=float)
            torque += float(r[0] * f[1] - r[1] * f[0])

        # Linear: ease the body along the summed tension, capped.
        drift = np.zeros(2, dtype=float)
        mag = float(np.hypot(total[0], total[1]))
        if mag > 1e-9:
            step = min(self.max_body_velocity, mag)
            drift = np.array([step * total[0] / mag, step * total[1] / mag])
            self.x += drift[0]
            self.y += drift[1]

        # Angular: rotate the body by the summed torque, capped. As theta turns,
        # the whole base ring rotates, so the fan re-orients (the top limb is
        # not permanently on top).
        dtheta = self.body_torque_gain * torque
        cap = self.max_body_angular_velocity
        dtheta = max(-cap, min(cap, dtheta))
        self.theta = float((self.theta + dtheta) % (2.0 * np.pi))

        self.last_body_force = total.copy()
        self.last_body_drift = drift.copy()
        self.last_body_torque = float(torque)
        self.last_body_dtheta = float(dtheta)
        return None

    def find_color(
            self,
            surf: RandomSurface,
            inference_mode: MLMode = MLMode.NO_MODEL,
            model = None
    ) -> List[List[Color]]:
        """
        Finds the color of the suckers in the limbs.

        NO_MODEL and SUCKER inference vectorize cleanly: every sucker is
        independent (no cross-sucker interaction), so the whole octopus is
        evaluated in a single TensorFlow pass over an (N, 2) tensor - one
        forward call instead of 256 single-row `model.predict` calls. LIMB
        inference still uses the per-sucker ThreadPool path because its model
        consumes a ragged, per-sucker neighbourhood (a stateful RNN branch)
        that does not batch trivially.
        """
        if inference_mode in (MLMode.NO_MODEL, MLMode.SUCKER):
            return self._find_color_batched(surf, inference_mode, model)
        return self._find_color_threadpool(surf, inference_mode, model)

    def _find_color_threadpool(
            self,
            surf: RandomSurface,
            inference_mode: MLMode = MLMode.NO_MODEL,
            model = None
    ) -> List[List[Color]]:
        """
        Finds the color of the suckers in the limbs, in parallel (per-sucker).

        The original inference path: one Keras call per sucker, dispatched
        through nested ThreadPools. Retained for LIMB mode and as the
        reference implementation the batched path is validated against.
        """
        pool = ThreadPool(processes = 2)
        pool_iterable = [(l, ix) for ix, l in enumerate(self.limbs)]
        result = pool.imap_unordered(
            func = lambda x: (x[0].find_color(surf, inference_mode, model), x[1],),
            iterable = pool_iterable)
        ret = sorted(result, key = lambda x: x[1])
        ret = [x[0] for x in ret]
        return ret

    def _find_color_batched(
            self,
            surf: RandomSurface,
            inference_mode: MLMode,
            model = None
    ) -> List[List[Color]]:
        """
        Whole-octopus color inference in a single TensorFlow pass.

        Gathers every sucker's (current_color, surface_color) into one
        (N, 2) tensor and computes all N new colors at once, then scatters
        them back into the per-limb Color lists in original order. All of the
        math - the surface lookup, the constraint clamp, and the model forward
        pass - runs as TensorFlow ops on one batched tensor. Only the initial
        read of Python-float object attributes and the final write back into
        Color objects cross the tensor boundary.

        Semantically identical to the per-sucker path: the batch axis carries
        independent suckers, so batching == calling the single-agent model
        once per sucker.
        """
        if inference_mode == MLMode.SUCKER:
            assert model is not None, \
                "model inference specified but no model was specified"
            assert isinstance(model, keras.Model), \
                f"Expected keras model, got {type(model)}"

        # Flatten suckers, remembering how many belong to each limb so the
        # nested (per-limb) list can be rebuilt in the same order.
        limb_sizes = [len(l.suckers) for l in self.limbs]
        suckers = [s for l in self.limbs for s in l.suckers]

        # Read the Python-float object state into tensors (the one unavoidable
        # host->device gather while suckers remain Python objects). RGB: each
        # sucker contributes its current [r, g, b].
        cur = tf.constant([[s.c.r, s.c.g, s.c.b] for s in suckers],
                          dtype=tf.float32)  # (N, 3)
        # float64 positions so the round-to-grid matches the per-sucker path
        # bit-for-bit: at float32, a position at x.5+epsilon collapses onto the
        # .5 boundary and banker's-rounds the other way, reading a neighbouring
        # grid cell. The reference path rounds Python float64s, so we do too.
        xy = tf.constant([[s.x, s.y] for s in suckers],
                         dtype=tf.float64)  # (N, 2)

        # Surface color under each sucker, in one on-device gather. The grid is
        # RGB, indexed [y][x] -> (3,); positions are rounded and clamped to the
        # grid, mirroring Sucker.get_surf_color_at_this_sucker / get_val.
        grid = tf.constant(surf.grid, dtype=tf.float32)  # (y_len, x_len, 3)
        y_len, x_len = int(grid.shape[0]), int(grid.shape[1])
        ix = tf.clip_by_value(
            tf.cast(tf.round(xy[:, 0]), tf.int32), 0, x_len - 1)
        iy = tf.clip_by_value(
            tf.cast(tf.round(xy[:, 1]), tf.int32), 0, y_len - 1)
        surf_c = tf.gather_nd(grid, tf.stack([iy, ix], axis=1))  # (N, 3)

        if inference_mode == MLMode.NO_MODEL:
            # Vectorized Sucker._find_color_change, applied to every channel
            # independently: step each of r/g/b toward the surface colour,
            # capped at +/- each sucker's max_hue_change, then clip to [0, 1].
            d_max = tf.constant([s.max_hue_change for s in suckers],
                                dtype=tf.float32)[:, None]  # (N, 1) broadcasts
            dc = tf.clip_by_value(surf_c - cur, -d_max, d_max)
            new_c = tf.clip_by_value(cur + dc, 0.0, 1.0)  # (N, 3)
        else:  # MLMode.SUCKER
            # The sucker model is a single-channel (current, surface) -> new
            # map; apply it to each RGB channel independently.
            channels = [
                tf.reshape(
                    model(tf.stack([cur[:, ch], surf_c[:, ch]], axis=1),
                          training=False),
                    [-1])
                for ch in range(3)
            ]
            new_c = tf.stack(channels, axis=1)  # (N, 3)

        # Single host pull, then scatter back into Color objects, rebuilding
        # the per-limb nesting in order.
        new_c = new_c.numpy()
        out: List[List[Color]] = []
        k = 0
        for size in limb_sizes:
            row = [Color(float(v[0]), float(v[1]), float(v[2]))
                   for v in new_c[k:k + size]]
            out.append(row)
            k += size
        return out


    def set_color(self,
                  surf: RandomSurface,
                  inference_mode: MLMode = MLMode.NO_MODEL,
                  model = None
                  ):
        """
        Main entry point for setting color.  Calls the child Limb object's find_color method,
        and then its force_color method
        """
        color_matrix = self.find_color(surf, inference_mode, model)
        for limb, c_array in zip(self.limbs, color_matrix):
            limb.force_color(c_array)

    def visibility(self, surf: RandomSurface):
        """Computes octopus visibility as mean of square of octopus color error"""
        sum_of_squares = 0.0
        num_suckers = 0
        for l in self.limbs:
            num_suckers += (l.rows * l.cols)
            for s in l.suckers:
                pred = s.c.to_rgb()
                truth = s.get_surf_color_at_this_sucker(surf).to_rgb()
                diff = pred - truth
                diff_sq = np.power(diff, 2)
                error = np.sum(diff_sq)
                sum_of_squares += error
        mse = sum_of_squares / num_suckers
        return mse
