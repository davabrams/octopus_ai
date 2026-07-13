"""
Base loader for Keras models and data files.
"""
import os
import pathlib
from abc import ABC
from typing import Optional, Union, Any

from simulator.simutil import MLMode


class AlreadyLoadedError(BaseException):
    """Error to throw if a model load is attempted twice."""


class DefaultLoader(ABC):
    """
    Handles loading (and storage) of keras models and data files.
    """

    path: Optional[str] = None
    object: Optional[Any] = None
    defaults: dict = {
        MLMode.NO_MODEL: None,
        MLMode.SUCKER: None,
        MLMode.LIMB: None,
        MLMode.FULL: None,
    }
    LOCAL_DIR = pathlib.Path(__file__).parent.resolve()

    def __init__(
        self,
        file_name_or_ml_mode: Optional[Union[str, MLMode]],
        **kwargs: dict,
    ):
        if file_name_or_ml_mode is None:
            return

        if isinstance(file_name_or_ml_mode, MLMode):
            file_name_or_ml_mode = self._convert_ml_mode_to_file_name(
                file_name_or_ml_mode
            )
        if file_name_or_ml_mode is None:
            return
        self.path = file_name_or_ml_mode
        if "/" not in self.path:
            self.path = self._convert_filename_to_full_path(self.path)

        self._confirm_file_exists()
        self._load(**kwargs)

    def _confirm_file_exists(self) -> None:
        print(self.path)
        if os.path.isfile(self.path):
            return
        raise FileNotFoundError(f"Not found: {self.path}")

    def _convert_ml_mode_to_file_name(self, ml_mode: MLMode) -> str:
        return self.defaults[ml_mode]

    def _convert_filename_to_full_path(self, filename: str) -> str:
        filename = os.path.join(self.LOCAL_DIR, filename)
        return filename

    def _load(self, **kwargs: dict) -> None:
        """Load the object into memory. Must be implemented by subclasses."""
        raise NotImplementedError

    def get_path(self) -> str:
        """Returns the path of the encapsulated object."""
        return self.path

    def get_object(self) -> Any:
        """Returns the loaded object."""
        return self.object
