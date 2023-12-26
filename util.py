""" Utilities for Octopus ML modeling """
import os
import numpy as np
from tensorflow import keras
from keras.utils import losses_utils
import tensorflow as tf

np.set_printoptions(precision=4)

def erase_all_logs():
    """
    Erases tensorboard logs from log folder generated during training.
    """
    log_dir = "./logs"
    log_prefix = "events.out.tfevents"
    if not os.path.exists(log_dir):
        print("Log folder not found, nothing erased.")
        return
    onlyfiles = [f for f in os.listdir(log_dir) if
                 os.path.isfile(os.path.join(log_dir, f)) and
                 len(f) >= len(log_prefix) and
                 f.startswith(log_prefix)]
    if len(onlyfiles) == 0:
        print("No log files found in log folder, nothing erased.")
    for f in onlyfiles:
        try:
            os.remove(log_dir + "/" + f)
        except:
            print(f"Could not remove log file: {f}")
        else:
            print(f"Removed log file: {f}")



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

def convert_pytype_to_model_input_type(input_float):
    """
    Converts native types to 2 dimensional tensors for tf model input
    """
    shaped_tensor = tf.convert_to_tensor(input_float, dtype='float32')
    while len(shaped_tensor.shape) < 2:
        shaped_tensor = tf.expand_dims(shaped_tensor, axis=len(shaped_tensor.shape))
    output_tensor = tf.data.Dataset.from_tensor_slices((shaped_tensor, shaped_tensor))
    return output_tensor

@keras.saving.register_keras_serializable(package="Octo", name="ConstraintLoss")
class ConstraintLoss(tf.keras.losses.Loss):
    """ This is the loss that maintains our color change rate constraint.
    We have defined the color change rate to be a maximum of 0.25 per iteration.
    This means that there is a cost incurred if the predicted value is 0.25
    greater than or less than the original value. The GT value is never used. """
    def __init__(self, threshold=0.25, weight=1.0):
        super().__init__()

        # The original_values tensor should have a 2D shape of (batch_size, num_features)
        # assert tf.is_tensor(original_values), "error: original values must be a tensor"
        # ov_size = original_values.shape
        # assert len(ov_size) == 2, "error: original value must have two dimensions"
        # batch_size = ov_size[0]
        # num_features = ov_size[1]
        # assert num_features == 1, "error: The original_values tensor should have a 2D shape of (batch_size = N, num_features = 1)"
        self.threshold = threshold
        self.weight = float(weight)

    def call(self, y_true: tf.TensorArray, y_pred: tf.TensorArray):

        # Calculate absolute difference between predictions and original values
        assert tf.is_tensor(y_pred) and tf.is_tensor(y_true), "error: inputs must be tensors"

        # the ground truth input is y_true[0] and is not important for this loss
        original_value = y_true[1]
        predicted_value = y_pred[0]

        diff = tf.abs(predicted_value - original_value)

        # Apply threshold and square for stronger penalty
        excess_penalty = tf.where(diff > self.threshold,
                                  tf.square(diff - self.threshold),
                                  tf.zeros_like(diff))

        weighted_penalty =  tf.multiply(self.weight, excess_penalty)

        # Return scaled excess penalty as the loss
        return weighted_penalty

    def get_config(self):
        config = super().get_config()
        config.update({
            "threshold": self.threshold,
        })
        return config

    @classmethod
    def from_config(cls, config: dict):
        threshold = config.pop("threshold")
        return cls(threshold=threshold)

@keras.saving.register_keras_serializable(package="Octo", name="WeightedSumLoss")
class WeightedSumLoss(tf.keras.losses.Loss):
    """Takes the weighted sum of two loss functions"""
    def __init__(self, 
                 threshold = tf.convert_to_tensor(0.25),
                 weight=tf.convert_to_tensor(0.99),
                 step=0,
                 logwriter=None):
        super().__init__()
        self.threshold = threshold
        self.f1 = ConstraintLoss()
        self.f2 = keras.losses.MeanSquaredError()
        self.w1 = weight
        self.w2 = tf.convert_to_tensor(1.0) - weight
        if logwriter:
            self.writer = logwriter
        self.step = step

    def call(self, y_true, y_pred):
        loss1 = self.f1(y_true, y_pred)
        loss2 = self.f2(y_true[0], y_pred)
        w_loss1 = self.w1 * loss1
        w_loss2 = self.w2 * loss2

        if self.writer:
            with self.writer.as_default():
                tf.summary.scalar('w_loss1', w_loss1, step = self.step)
                tf.summary.scalar('w_loss2', w_loss2, step = self.step)
                tf.summary.scalar('loss1', loss1, step = self.step)
                tf.summary.scalar('loss2', loss2, step = self.step)
                self.writer.flush()

        # loss_details = tf.convert_to_tensor([[w_loss1 + w_loss2, 
        #                                      loss1, 
        #                                      loss2, 
        #                                      w_loss1, 
        #                                      w_loss2]])
        return w_loss1 + w_loss2

    def get_config(self):
        config = super().get_config()
        config.update({
            "threshold": self.threshold,
        })
        return config

    @classmethod
    def from_config(cls, config: dict):
        threshold = config.pop("threshold")
        return cls(threshold=threshold)
