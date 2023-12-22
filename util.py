""" Utilities for Octopus ML modeling """
import numpy as np
from tensorflow import keras
from keras.utils import losses_utils
import tensorflow as tf

np.set_printoptions(precision=4)


def octo_norm(_x: np.array, reverse=False):
    """
    Normalizer for convertinbg color gradients (0..1) to tf inputs (-1..+1)
    Also includes the reverse operation
    """
    if reverse:
        return np.divide(np.add(_x, 1), 2)
    else:
        return np.subtract(np.multiply(_x, 2), 1)

def train_test_split(data, labels, test_size=0.2, random_state=None):
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
    if not isinstance(data, np.ndarray) or not isinstance(labels, np.ndarray):
        raise TypeError("Both data and labels must be NumPy arrays.")

    if test_size < 0 or test_size > 1:
        raise ValueError("test_size must be between 0 and 1.")

    num_data_fields = data.shape[0]
    num_labels_fields = labels.shape[0]
    num_samples = data.shape[1]
    shuffle_indices = np.arange(num_samples)
    if random_state is not None:
        # np.random.seed(random_state)
        np.random.shuffle(shuffle_indices)

    split_point = int(num_samples * test_size)

    for _df in range(num_data_fields):
        train_data = data[_df][shuffle_indices[:split_point]]
        test_data = data[_df][shuffle_indices[split_point:]]

    for _df in range(num_labels_fields):
        train_labels = labels[_df][shuffle_indices[:split_point]]
        test_labels = labels[_df][shuffle_indices[split_point:]]

    return train_data, train_labels, test_data, test_labels

@keras.saving.register_keras_serializable(package="Octo", name="ConstraintLoss")
class ConstraintLoss(tf.keras.losses.Loss):
    """ This is the loss that maintains our color change rate constraint.
    We have defined the color change rate to be a maximum of 0.25 per iteration.
    This means that there is a cost incurred if the predicted value is 0.25
    greater than or less than the original value. The true value is never used. """
    def __init__(self, original_values, threshold=0.25):
        super().__init__()
        self.original_values = original_values
        self.threshold = threshold

    def call(self, y_true, y_pred):
        # Calculate absolute difference between predictions and original values
        if tf.is_tensor(y_pred) and tf.is_tensor(y_true):
            y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(
                y_pred, y_true
            )

        diff = tf.abs(y_pred - self.original_values)

        # Apply threshold and square for stronger penalty
        excess_penalty = tf.where(diff > self.threshold,
                                  tf.square(diff - self.threshold),
                                  tf.zeros_like(diff))

        # Return scaled excess penalty as the loss
        return 100 * tf.reduce_mean(excess_penalty)

    def get_config(self):
        config = super().get_config()
        config.update({
            "original_values": self.original_values,
            "threshold": self.threshold,
        })
        return config

    @classmethod
    def from_config(cls, config: dict):
        original_values = config.pop("original_values")
        threshold = config.pop("threshold")
        return cls(original_values=original_values, threshold=threshold)

@keras.saving.register_keras_serializable(package="Octo", name="WeightedSumLoss")
class WeightedSumLoss(tf.keras.losses.Loss):
    """Takes the weighted sum of two loss functions"""
    def __init__(self, original_values, threshold = 0.25):
        super().__init__()
        self.original_values = original_values
        self.threshold = threshold
        self.f1 = ConstraintLoss(self.original_values)
        self.f2 = keras.losses.MeanSquaredError()
        self.w1 = 0.99
        self.w2 = 0.01

    def call(self, y_true, y_pred):
        loss1 = self.f1(y_true, y_pred)
        loss2 = self.f2(y_true, y_pred)
        w_loss1 = self.w1 * loss1
        w_loss2 = self.w2 * loss2
        return w_loss1 + w_loss2

    def get_config(self):
        config = super().get_config()
        config.update({
            "original_values": self.original_values,
            "threshold": self.threshold,
        })
        return config

    @classmethod
    def from_config(cls, config: dict):
        original_values = config.pop("original_values")
        threshold = config.pop("threshold")
        return cls(original_values=original_values, threshold=threshold)
