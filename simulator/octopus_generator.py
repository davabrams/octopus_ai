"""Octopus Class"""
from dataclasses import dataclass, field
from typing import List
from tensorflow import keras
from multiprocessing.pool import ThreadPool
import numpy as np
from simulator.surface_generator import RandomSurface
from simulator.simutil import (
    MovementMode,
    Agent,
    Color,
    MLMode,
    CenterPoint,
    convert_adjacents_to_ragged_tensor
)

@dataclass
class Sucker:
    """
    Stores location and color of a sucker, and includes methods to change the sucker color.
    This object is instantiated by Limb objects.
    """

    x: float
    y: float
    c: Color = field(default_factory=Color)
    prev: "Sucker" = field(default_factory="Sucker")

    def __init__(self, x: float, y: float, c: Color = Color(), params = None):
        self.x = x
        self.y = y
        self.c = c
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
            assert isinstance(model, keras.models.Sequential), f"Expected sequential keras model, got {type(model)}"
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
    suckers = []

    def __init__(self, x_octo: float, y_octo: float, init_angle: float, params: dict):
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
        self.agent_range_radius = params['agent_range_radius']
        self.threading = params['octo_threading']

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

    def move(self, x_octo: float, y_octo: float):
        """randomly shift thetas, reconstruct centerline, and refresh sucker 
        locations """
        # TODO(davabrams): instead of doing it this way, adjust the final
        # centerpoint to move towards prey and away from threats, and then
        # adjust the rest of the center points accordingly as a spline or
        # something
        if self.movement_mode == MovementMode.RANDOM:
            self._move_random(x_octo, y_octo)
        elif self.movement_mode == MovementMode.ATTRACT_REPEL:
            self._move_attract_repel(x_octo, y_octo)
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

    def _move_attract_repel(self, x_octo: float, y_octo: float):
        raise NotImplementedError("Not implemented yet")
        # step 1: move the last centerpoint towards prey and away from threats
        # step 2: shift the first centerpoint relative to the octopus new postion
        # step 3:

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
        ATTRACT_REPEL: randomly moves the octopus, but considers agents.
            - Attracted to prey
            - Repelled by threats
        """
        if self.movement_mode == MovementMode.ATTRACT_REPEL and not ag:
            assert False, "movement mode set to attract/repel but no agent object passed"

        if self.movement_mode == MovementMode.RANDOM:
            self.x += np.random.uniform(-self.max_body_velocity, self.max_body_velocity)
            self.y += np.random.uniform(-self.max_body_velocity, self.max_body_velocity)
        elif self.movement_mode == MovementMode.ATTRACT_REPEL:
            self._move_attract_repel(ag)
        else:
            assert False, "Unknown movement mode"

        for l in self.limbs:
            l.move(self.x, self.y)

    def _move_attract_repel(self, ag: Agent = None):
        print("Attract/Repel movement mode not complete", ag.agent_type)

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
        map(lambda l, c_array: l.force_color[c_array], self.limbs, color_matrix)

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
