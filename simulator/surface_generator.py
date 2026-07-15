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

        # The grid is always RGB: shape (y_len, x_len, 3), float in [0, 1]. A
        # grayscale surface simply has r == g == b, so all downstream colour
        # matching is full-colour and grayscale is just a special case.
        image_grid = None
        if cfg.world.background_image:
            image_grid = self._load_image(
                cfg.world.background_image, x_len, y_len)

        if image_grid is not None:
            self.grid = image_grid  # full-colour picture (r,g,b independent)
        elif self.grayscale:
            # Grayscale noise: one random channel broadcast to r == g == b.
            gray = np.random.rand(self._y_len, self._x_len).astype(np.float32)
            self.grid = np.repeat(gray[:, :, None], 3, axis=2)
        else:
            # Full-colour noise: independent r, g, b per cell.
            self.grid = np.random.rand(
                self._y_len, self._x_len, 3).astype(np.float32)

    @staticmethod
    def _load_image(path, x_len, y_len):
        """Load an image as a (y_len, x_len, 3) RGB grid in [0, 1].

        Returns None (so the caller falls back to a random surface) if the
        image can't be read, rather than crashing a run over a bad path.
        """
        try:
            from PIL import Image
            with Image.open(path) as img:
                # resize takes (width, height) = (x_len, y_len); the resulting
                # array is (height, width, 3) = (y_len, x_len, 3), grid[y][x].
                rgb = img.convert("RGB").resize((x_len, y_len))
            return np.asarray(rgb, dtype=np.float32) / 255.0
        except Exception as e:
            logging.warning(
                "Could not load background image %r (%s); "
                "falling back to a random surface.", path, e)
            return None

    def get_val(self, x: float, y: float):
        """RGB colour at (x, y) as a float32 array [r, g, b] in [0, 1].

        Takes floats and quantizes to the nearest grid cell.
        """
        if not ((x >= 0.0) and (x < self._x_len)):
            raise ValueError(f"x ({x}) must be between 0 and {self._x_len}")
        if not ((y >= 0.0) and (y < self._y_len)):
            raise ValueError(f"y ({y}) must be between 0 and {self._y_len}")

        return self.grid[int(round(y))][int(round(x))]
