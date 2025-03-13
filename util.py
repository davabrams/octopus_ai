""" Utilities for Octopus ML modeling """
import os
import shutil
import logging
from typing import Optional, Union, Any
from abc import ABC
import pathlib
import numpy as np
import tensorflow as tf
from simulator.simutil import MLMode

np.set_printoptions(precision=4)

def erase_all_logs():
    """
    Erases tensorboard logs from log folder generated during training.
    """
    log_dir = "./logs"
    log_prefix = "events.out.tfevents"
    if not os.path.exists(log_dir):
        print("Log folder not found, nothing erased from." + log_dir)
    else:
        onlyfiles = [f for f in os.listdir(log_dir) if
                    os.path.isfile(os.path.join(log_dir, f)) and
                    len(f) >= len(log_prefix) and
                    f.startswith(log_prefix)]
        if len(onlyfiles) == 0:
            print("No log files found in log folder, nothing erased.")
        for f in onlyfiles:
            try:
                os.remove(log_dir + "/" + f)
            except Exception as e:
                print(f"Could not remove log file: {f}\n Found error {e}")
            else:
                print(f"Removed log file: {f}")
    log_dir = "./models/logs/sucker/fit"
    if not os.path.exists(log_dir):
        print("Log folder not found, nothing erased from." + log_dir)
    else:
        onlyfolders = [f for f in os.listdir(log_dir) if
                        os.path.isdir(os.path.join(log_dir, f))]
        if len(onlyfolders) == 0:
            print("No log folders found in sucker training log folder, nothing erased.")
        for f in onlyfolders:
            try:
                shutil.rmtree(log_dir + "/" + f) # Be careful changing this line!
            except:
                print(f"Could not remove log folder: {f}")
            else:
                print(f"Removed log folder: {f}")

def octo_norm(_x: np.array, reverse=False):
    """
    Normalizer for convertinbg color gradients (0..1) to tf inputs (-1..+1)
    Also includes the reverse operation
    """
    if reverse:
        return np.divide(np.add(_x, 1), 2)
    else:
        return np.subtract(np.multiply(_x, 2), 1)

def train_test_split(state_data, gt_data, test_size=0.2, random_state=None):
    """
    Splits data and labels into training and test sets.

    Args:
        data: NumPy array of data points.
        labels: NumPy array of corresponding labels.
        test_size: Proportion of data to be used for testing (between 0 and 1).
        random_state: Seed for random shuffling (optional).

    Returns:
        train_data, train_labels, test_data, test_labels: 
                NumPy arrays of training and test data and labels.
    """
    if not isinstance(state_data, np.ndarray) or not isinstance(gt_data, np.ndarray):
        raise TypeError("Both data and labels must be NumPy arrays.")

    if test_size < 0 or test_size > 1:
        raise ValueError("test_size must be between 0 and 1.")

    num_data_fields = state_data.shape[0]
    num_labels_fields = gt_data.shape[0]
    num_samples = state_data.shape[1]
    shuffle_indices = np.arange(num_samples)
    if random_state is not None:
        # np.random.seed(random_state)
        pass
    np.random.shuffle(shuffle_indices)

    split_point = int(num_samples * test_size)

    for _df in range(num_data_fields):
        train_state_data = state_data[_df][shuffle_indices[:split_point]]
        test_state_data = state_data[_df][shuffle_indices[split_point:]]

    for _df in range(num_labels_fields):
        train_gt_data = gt_data[_df][shuffle_indices[:split_point]]
        test_gt_data = gt_data[_df][shuffle_indices[split_point:]]

    return train_state_data, train_gt_data, test_state_data, test_gt_data

def train_test_split_multiple_state_vectors(state_data, gt_data, test_size=0.2, random_state=None):
    """
    Splits data and labels into training and test sets, but takes an array of state_data

    Args:
        data: NumPy array of data points.
        labels: NumPy array of corresponding labels.
        test_size: Proportion of data to be used for testing (between 0 and 1).
        random_state: Seed for random shuffling (optional).

    Returns:
        [train_data], train_labels, [test_data], test_labels: 
                NumPy arrays of training and test data and labels.
    """
    if not isinstance(state_data, list) or not isinstance(gt_data, list):
        raise TypeError("State data and labels must be a list.")

    for state_type in state_data:
        if not isinstance(state_type, list):
            raise TypeError("State data fields must be a list.")

    if test_size < 0 or test_size > 1:
        raise ValueError("test_size must be between 0 and 1.")

    num_samples = len(gt_data)

    shuffle_indices = np.arange(num_samples)
    if random_state is not None:
        np.random.seed(random_state)
    np.random.shuffle(shuffle_indices)

    split_point = int(num_samples * (1 - test_size))

    train_state_data = [[] for _ in range(len(state_data))]
    test_state_data = [[] for _ in range(len(state_data))]
    for state_type_ix, state_type in enumerate(state_data):
        state_type_shuffled = [state_type[ix] for ix in shuffle_indices]
        train_state_data[state_type_ix] = state_type_shuffled[:split_point]
        test_state_data[state_type_ix] = state_type_shuffled[split_point:]

    gt_data_shuffled = [gt_data[ix] for ix in shuffle_indices]
    train_gt_data = gt_data_shuffled[:split_point]
    test_gt_data = gt_data_shuffled[split_point:]

    return train_state_data, train_gt_data, test_state_data, test_gt_data


def convert_pytype_to_tf_dataset(input_np_array, batch_size):
    """
    Converts native types to 2 dimensional tensors for tf model input
    """
    shaped_tensor = tf.convert_to_tensor(input_np_array, dtype='float32')
    while len(shaped_tensor.shape) < 2:
        shaped_tensor = tf.expand_dims(shaped_tensor, axis=len(shaped_tensor.shape))
    output_tensor = tf.data.Dataset.from_tensor_slices((shaped_tensor, shaped_tensor))
    output_tensor = output_tensor.batch(batch_size=batch_size)
    return output_tensor

class AlreadyLoadedError(BaseException):
    """
    Error to throw if a model load is attempted twice
    """

class DefaultLoader (ABC):
    """
    Handles loading (and storage) of keras models
    """
    path: Optional[str] = None
    object: Optional[Any] = None
    defaults: dict = {
        MLMode.NO_MODEL: None,
        MLMode.SUCKER: None,
        MLMode.LIMB: None,
        MLMode.FULL: None
    }
    LOCAL_DIR = pathlib.Path(__file__).parent.resolve()

    def __init__(self, file_name_or_ml_mode: Optional[Union[str, MLMode]], **kwargs: dict):
        if file_name_or_ml_mode is None:
            return

        if isinstance(file_name_or_ml_mode, MLMode):
            file_name_or_ml_mode = self._convert_ml_mode_to_file_name(file_name_or_ml_mode)
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
        """
        Load the model into memort
        """
        raise NotImplementedError

    def get_path(self) -> str:
        """
        Returns the name of the encapsulated model
        """
        return self.path

    def get_object(self) -> Any:
        """
        Returns the model object
        """
        return self.object
