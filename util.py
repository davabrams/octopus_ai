""" Utilities for Octopus ML modeling """
import os, shutil
import numpy as np
import tensorflow as tf
import seaborn as sn
import matplotlib.pyplot as plt

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
            except:
                print(f"Could not remove log file: {f}")
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

def run_sucker_model_inference(sucker_model=None, GameParameters: dict = None):
    """
    Runs a standard sweep inference on the input domain
    """
    assert sucker_model, "No model found, can't run inference"
    assert GameParameters, "No parameters found, can't run inference"

    # batch_size = GameParameters['batch_size']
    # max_hue_change = GameParameters['octo_max_hue_change']
    # constraint_loss_weight = GameParameters['constraint_loss_weight']

    ####### Iterate over domain space
    range_vals = [0.0,0.25,0.5,0.75,1.0]
    res = []
    for curr in range_vals:
        row = []
        for gt in range_vals:
            test_input = np.array([[curr, gt]])

            #computes prediction output
            pred = sucker_model.predict(test_input, verbose = 0)[0][0]
            row.append(pred)

        res.append(row)

    ####### Plot inference results
    plt.figure(figsize = (10,7))
    sn.heatmap(res, annot=True)
    plt.xlabel('surface color')
    locs, labels = plt.xticks()
    plt.xticks(locs, range_vals)
    plt.ylabel('sucker previous color')
    locs, labels = plt.yticks()
    plt.yticks(locs, range_vals)
    plt.show()

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
