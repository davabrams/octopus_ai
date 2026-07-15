""" Utilities for Octopus ML modeling """
import os
import shutil
import numpy as np

# Canonical implementations live in training/data_utils.py; re-exported here
# for backwards compatibility with older imports (tests, notebooks).
# The legacy duplicated copies of these functions — and of DefaultLoader,
# whose canonical home is training/models/base_loader.py — were removed.
from training.data_utils import (  # noqa: F401
    train_test_split,
    train_test_split_multiple_state_vectors,
    convert_pytype_to_tf_dataset,
)

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
    for log_dir in ("./models/logs/sucker/fit", "./models/logs/limb/fit"):
        if not os.path.exists(log_dir):
            print("Log folder not found, nothing erased from." + log_dir)
            continue
        onlyfolders = [f for f in os.listdir(log_dir) if
                       os.path.isdir(os.path.join(log_dir, f))]
        if len(onlyfolders) == 0:
            print(f"No log folders found in {log_dir}, nothing erased.")
        for f in onlyfolders:
            try:
                shutil.rmtree(log_dir + "/" + f)  # Be careful changing!
            except Exception:
                print(f"Could not remove log folder: {f}")
            else:
                print(f"Removed log folder: {f}")


def octo_norm(_x: np.array, reverse=False):
    """
    Normalizer for converting color gradients (0..1) to tf inputs (-1..+1)
    Also includes the reverse operation
    """
    if reverse:
        return np.divide(np.add(_x, 1), 2)
    else:
        return np.subtract(np.multiply(_x, 2), 1)
