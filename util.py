import numpy as np
from tensorflow import keras

np.set_printoptions(precision=4)

""" Utilities for ML modeling """

def OctoNorm(x: np.array):
   return np.subtract(np.multiply(x, 2), 1)

def train_test_split(data, labels, test_size=0.2, random_state=None):
    """
    Splits data and labels into training and test sets.

    Args:
        data: NumPy array of data points.
        labels: NumPy array of corresponding labels.
        test_size: Proportion of data to be used for testing (between 0 and 1).
        random_state: Seed for random shuffling (optional).

    Returns:
        train_data, train_labels, test_data, test_labels: NumPy arrays of training and test data and labels.
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

    for ix in range(num_data_fields):
        train_data = data[ix][shuffle_indices[:split_point]]
        test_data = data[ix][shuffle_indices[split_point:]]

    for ix in range(num_labels_fields):
        train_labels = labels[ix][shuffle_indices[:split_point]]
        test_labels = labels[ix][shuffle_indices[split_point:]]

    return train_data, train_labels, test_data, test_labels

import tensorflow as tf

@keras.saving.register_keras_serializable(package="Octo", name="Excess20Loss")
class Excess20Loss(tf.keras.losses.Loss):
  def __init__(self, original_values, threshold=0.2):
    super(Excess20Loss, self).__init__()
    self.original_values = original_values
    self.threshold = threshold

  def call(self, y_true, y_pred):
    # Calculate absolute difference between predictions and original values
    diff = tf.abs(y_pred - self.original_values)

    # Calculate fraction of original value exceeded
    excess_fraction = diff / self.original_values

    # Apply threshold and square for stronger penalty
    excess_penalty = tf.where(excess_fraction > self.threshold,
                              tf.square(excess_fraction - self.threshold),
                              tf.zeros_like(excess_fraction))

    # Return scaled excess penalty as the loss
    return tf.reduce_mean(excess_penalty)

  def get_config(self):
    config = super(Excess20Loss, self).get_config()
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

@keras.saving.register_keras_serializable(package="Octo", name="ConstraintLoss")
class ConstraintLoss(tf.keras.losses.Loss):
  """ This is the loss that maintains our color change rate constraint.
  We have defined the color change rate to be a maximum of 0.25 per iteration.
  This means that there is a cost incurred if the predicted value is 0.25
  greater than or less than the original value. The true value is never used. """
  def __init__(self, original_values, threshold=0.25):
    super(ConstraintLoss, self).__init__()
    self.original_values = original_values
    self.threshold = threshold

  def call(self, y_true, y_pred):

    # Calculate absolute difference between predictions and original values
    diff = tf.abs(y_pred - self.original_values)

    # Apply threshold and square for stronger penalty
    excess_penalty = tf.where(diff > self.threshold,
                              tf.square(diff - self.threshold),
                              tf.zeros_like(diff))

    # Return scaled excess penalty as the loss
    return 100 * tf.reduce_mean(excess_penalty)

  def get_config(self):
    config = super(ConstraintLoss, self).get_config()
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
