"""Octopus Class"""
from typing import List
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
            self.max_hue_change = params['octo_max_hue_change']
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
        """Gets the color of the surface underneath the sucker"""
        x_grid_location = int(round(self.x))
        y_grid_location = int(round(self.y))
        c_val = surf.get_val(x_grid_location, y_grid_location) * 1.0
        return Color(c_val, c_val, c_val)

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

    def __init__(self, x_octo: float, y_octo: float, init_angle: float, params: dict):
        self.suckers: list[Sucker] = []
        self.max_sucker_distance = params['octo_max_sucker_distance']
        self.min_sucker_distance = params['octo_min_sucker_distance']
        self.sucker_distance = self.min_sucker_distance
        self.rows = params['limb_rows']
        self.cols = params['limb_cols']
        self.center_line = [CenterPoint() for _ in range(self.rows)]
        self.x_len = params['x_len']
        self.y_len = params['y_len']
        self.max_hue_change = params['octo_max_hue_change']
        self.movement_mode = params['limb_movement_mode']
        self.max_arm_theta = params['octo_max_arm_theta']
        self.max_arm_reach_theta = params['octo_max_arm_reach_theta']
        self.max_limb_offset = params['octo_max_limb_offset']
        self.arm_stiffness = params['octo_arm_stiffness']
        self.arm_rest_fraction = params['octo_arm_rest_fraction']
        self.agent_range_radius = params['agent_range_radius']
        self.threading = params['octo_threading']

        # Last-frame force capture (populated by _move_lumped_spring).
        # Shared source of truth for on-screen arrows and the force DB.
        # f_attract: prey/threat pull at the tip; f_spring: restoring force;
        # net: their sum; tension: base reaction fed to the body.
        self.last_f_attract = np.zeros(2, dtype=float)
        self.last_f_spring = np.zeros(2, dtype=float)
        self.last_net = np.zeros(2, dtype=float)
        self.last_tension = np.zeros(2, dtype=float)
        self.last_arm_length = 0.0

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
             agents: list = None, coordinated_influence=None):
        """Move the limb, then refresh sucker locations.

        x_octo, y_octo: the (possibly just-moved) body position; the arm
            base is anchored here.
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
                self.find_adjacents(s, self.agent_range_radius)
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
    def __init__(self, params: dict):
        self.x = params['x_len'] / 2.0
        self.y = params['y_len'] / 2.0
        self.max_body_velocity = params['octo_max_body_velocity']
        self.movement_mode = params['octo_movement_mode']
        self.model = None
        self.threading = params['octo_threading']

        # Last-frame body force capture (populated by _move_lumped_spring):
        # last_body_force is the summed arm tension; last_body_drift is the
        # capped displacement actually applied to the body this frame.
        self.last_body_force = np.zeros(2, dtype=float)
        self.last_body_drift = np.zeros(2, dtype=float)

        num_arms = params['octo_num_arms']
        self.limbs = [
            Limb(
                self.x,
                self.y,
                float(ix/num_arms * 2 * np.pi),
                params
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
        if self.movement_mode == MovementMode.LUMPED_SPRING and not ag:
            assert False, "movement mode set to attract/repel but no agent object passed"

        agents = ag.agents if ag is not None else None
        coordinated_influence = None

        if self.movement_mode == MovementMode.RANDOM:
            self.x += np.random.uniform(-self.max_body_velocity, self.max_body_velocity)
            self.y += np.random.uniform(-self.max_body_velocity, self.max_body_velocity)
        elif self.movement_mode == MovementMode.LUMPED_SPRING:
            coordinated_influence = self._move_lumped_spring(ag)
        else:
            assert False, "Unknown movement mode"

        for l in self.limbs:
            l.move(self.x, self.y, agents, coordinated_influence)

    def _move_lumped_spring(self, ag=None):
        """Drift the body by the summed spring tension of its arms.

        Pure-tension coupling (no direct attraction term): the body only
        moves because its arms are straining. Each arm contributes its
        tension reaction (stiffness * stretch, directed toward that arm's
        tip); relaxed arms contribute ~nothing, arms reaching hard toward
        prey tug strongly. The body eases along the summed tension, capped
        by max_body_velocity.

        Note the ordering in Octopus.move: arms are reflowed *after* this,
        so the tension summed here reflects last step's arm strain. Over the
        loop this is a stable one-step lag - the body chases the strain the
        arms reported, they re-strain toward prey, and it converges.

        Returns None: arms sense agents at their own tips (limb autonomy).
        Full-octopus coordination can later pass a shared influence here.
        """
        total = np.zeros(2, dtype=float)
        for limb in self.limbs:
            total += limb.tension_vector()

        drift = np.zeros(2, dtype=float)
        mag = float(np.hypot(total[0], total[1]))
        if mag > 1e-9:
            step = min(self.max_body_velocity, mag)
            drift = np.array([step * total[0] / mag, step * total[1] / mag])
            self.x += drift[0]
            self.y += drift[1]

        # Capture for logging / visualization.
        self.last_body_force = total.copy()
        self.last_body_drift = drift.copy()

        return None

    def find_color(
            self,
            surf: RandomSurface,
            inference_mode: MLMode = MLMode.NO_MODEL,
            model = None
    ) -> List[List[Color]]:
        """
        Finds the color of the suckers in the limbs, in parallel
        """
        pool = ThreadPool(processes = 2)
        pool_iterable = [(l, ix) for ix, l in enumerate(self.limbs)]
        result = pool.imap_unordered(
            func = lambda x: (x[0].find_color(surf, inference_mode, model), x[1],),
            iterable = pool_iterable)
        ret = sorted(result, key = lambda x: x[1])
        ret = [x[0] for x in ret]
        return ret


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
