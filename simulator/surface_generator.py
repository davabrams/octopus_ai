# %% Patterned Surface Generator
import logging

import numpy as np

from octopus_ai.config import as_config


class RandomSurface:
    """Generates a surface of size (y_len, x_len).

    Three sources, in priority order:
    - world.background_image set: the image, grayscaled and resized to the grid
      (values in [0, 1]); the octopus camouflages to the picture.
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

        image_grid = None
        if cfg.world.background_image:
            image_grid = self._load_image(
                cfg.world.background_image, x_len, y_len)

        if image_grid is not None:
            self.grid = image_grid
            self.grayscale = True  # an image is continuous-tone
        elif self.grayscale:
            self.grid = np.random.rand(
                self._y_len, self._x_len).astype(np.float32)
        else:
            self.grid = np.random.randint(
                2, size=(self._y_len, self._x_len), dtype=np.int8)

    @staticmethod
    def _load_image(path, x_len, y_len):
        """Load an image as a (y_len, x_len) grayscale grid in [0, 1].

        Returns None (so the caller falls back to a random surface) if the
        image can't be read, rather than crashing a run over a bad path.
        """
        try:
            from PIL import Image
            with Image.open(path) as img:
                # resize takes (width, height) = (x_len, y_len); the resulting
                # array is (height, width) = (y_len, x_len), i.e. grid[y][x].
                gray = img.convert("L").resize((x_len, y_len))
            return np.asarray(gray, dtype=np.float32) / 255.0
        except Exception as e:
            logging.warning(
                "Could not load background image %r (%s); "
                "falling back to a random surface.", path, e)
            return None

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
