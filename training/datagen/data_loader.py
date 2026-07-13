import logging
import os
import pathlib
import pickle

from simulator.simutil import MLMode
from training.models.base_loader import DefaultLoader

local_folder = pathlib.Path(__file__).parent.resolve()

all_data = [
    filename for filename in os.listdir(local_folder) if "pkl" in filename
]


class DataLoader(DefaultLoader):
    """
    Handles loading (and storage) of generated data pickle files.
    """

    local_folder = pathlib.Path(__file__).parent.resolve()
    defaults = {
        MLMode.NO_MODEL: None,
        MLMode.SUCKER: str(local_folder / "sucker.pkl"),
        MLMode.LIMB: str(local_folder / "limb.pkl"),
    }

    def _load(self, **kwargs) -> None:
        """
        Load the data into memory.
        """
        if self.object is not None:
            logging.error(
                "Model already loaded! Not going to reload, and not going to "
                "error."
            )
            return
        if not self._confirm_file_exists():
            raise FileNotFoundError
        with open(self.path, "rb") as f:
            self.object = pickle.load(f)
