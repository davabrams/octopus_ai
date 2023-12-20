# %% Patterned Surface Generator
import numpy as np


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
        
        return self.grid[int(round(y))][int(round(x))]