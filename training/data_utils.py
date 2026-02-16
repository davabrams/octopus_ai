"""
Data utilities for training — train/test splitting and dataset conversion.
"""
import numpy as np
import tensorflow as tf


def train_test_split(state_data, gt_data, test_size=0.2, random_state=None):
    """
    Splits data and labels into training and test sets.

    Args:
        state_data: NumPy array of data points.
        gt_data: NumPy array of corresponding labels.
        test_size: Proportion of data to be used for testing (between 0 and 1).
        random_state: Seed for random shuffling (optional).

    Returns:
        train_data, train_labels, test_data, test_labels:
                NumPy arrays of training and test data and labels.
    """
    if not isinstance(state_data, np.ndarray) or not isinstance(
        gt_data, np.ndarray
    ):
        raise TypeError("Both data and labels must be NumPy arrays.")

    if test_size < 0 or test_size > 1:
        raise ValueError("test_size must be between 0 and 1.")

    num_data_fields = state_data.shape[0]
    num_labels_fields = gt_data.shape[0]
    num_samples = state_data.shape[1]
    shuffle_indices = np.arange(num_samples)
    if random_state is not None:
        pass
    np.random.shuffle(shuffle_indices)

    split_point = int(num_samples * test_size)
    train_state_data = test_state_data = np.ndarray((0,))
    train_gt_data = test_gt_data = np.ndarray((0,))
    for _df in range(num_data_fields):
        train_state_data = state_data[_df][shuffle_indices[:split_point]]
        test_state_data = state_data[_df][shuffle_indices[split_point:]]

    for _df in range(num_labels_fields):
        train_gt_data = gt_data[_df][shuffle_indices[:split_point]]
        test_gt_data = gt_data[_df][shuffle_indices[split_point:]]

    return train_state_data, train_gt_data, test_state_data, test_gt_data


def train_test_split_multiple_state_vectors(
    state_data, gt_data, test_size=0.2, random_state=None
):
    """
    Splits data and labels into training and test sets, but takes an array
    of state_data.

    Args:
        state_data: List of lists of data points.
        gt_data: List of corresponding labels.
        test_size: Proportion of data to be used for testing (between 0 and 1).
        random_state: Seed for random shuffling (optional).

    Returns:
        [train_data], train_labels, [test_data], test_labels
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
    Converts native types to 2 dimensional tensors for tf model input.
    """
    shaped_tensor = tf.convert_to_tensor(input_np_array, dtype="float32")
    while len(shaped_tensor.shape) < 2:
        shaped_tensor = tf.expand_dims(
            shaped_tensor, axis=len(shaped_tensor.shape)
        )
    output_tensor = tf.data.Dataset.from_tensor_slices(
        (shaped_tensor, shaped_tensor)
    )
    output_tensor = output_tensor.batch(batch_size=batch_size)
    return output_tensor
