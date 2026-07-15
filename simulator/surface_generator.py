# %% Patterned Surface Generator
import numpy as np

from octopus_ai.config import as_config

class RandomSurface:
    """Generates a random surface of size (y_len, x_len).

    Two modes, selected by world.surface_grayscale:
    - grayscale (the default): cells are uniform random floats in [0, 1)
    - binary: cells are 0 (black) or 1 (white), int8

    Takes a Config or a legacy flat params dict.
    """
    def __init__(self, params) -> None:
        cfg = as_config(params)
        x_len = cfg.world.x_len
        y_len = cfg.world.y_len
        rand_seed = cfg.run.rand_seed
        assert x_len > 0, "x must be >0"
        assert y_len > 0, "y must be >0"

        np.random.seed(seed = rand_seed)

        self._x_len = x_len
        self._y_len = y_len
        self.grayscale = bool(cfg.world.surface_grayscale)
        if self.grayscale:
            self.grid = np.random.rand(
                self._y_len, self._x_len).astype(np.float32)
        else:
            self.grid = np.random.randint(
                2, size=(self._y_len, self._x_len), dtype=np.int8)

    def get_val(self, x: float, y: float):
        """Gets the value at any location within the boundary of the grid:
        0/1 int in binary mode, a float in [0, 1) in grayscale mode.
        Takes in a float and quantizes it."""
        if not ((x >= 0.0) and (x < self._x_len)):
            raise ValueError(f"x ({x}) must be between 0 and {self._x_len}")
        if not ((y >= 0.0) and (y < self._y_len)):
            raise ValueError(f"y ({y}) must be between 0 and {self._y_len}")

        val = self.grid[int(round(y))][int(round(x))]
        return float(val) if self.grayscale else int(val)
