"""
Library for limb model training
"""
import pickle
import time as tm
import numpy as np
from octo_datagen import OctoDatagen
from util import (
    convert_pytype_to_tf_dataset,
    train_test_split,
    run_sucker_model_inference
)
from training.train import train_sucker_model

from .trainutil import Trainer

class LimbTrainer(Trainer):
    def __init__(self, GameParameters):
        self.GameParameters = GameParameters

    def datagen(self, SAVE_DATA_TO_DISK):
        datagen = OctoDatagen(self.GameParameters)
        data = datagen.run_color_datagen()
        if SAVE_DATA_TO_DISK:
            with open(self.GameParameters['limb_datagen_location'], 'wb') as file:
                pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
        return data

    def data_format(self, data):
        """
        Format Data for Train and Val
        """
        batch_size = self.GameParameters['batch_size']

        raise NotImplementedError("not implemented beyond this point")

        state_data = np.array([data['state_data']], dtype='float32') #sucker's current color
        gt_data = np.array([data['gt_data']], dtype='float32') #sucker's ground truth

        train_state_data, train_gt_data, test_state_data, test_gt_data = train_test_split(state_data, gt_data)

        train_dataset = convert_pytype_to_tf_dataset(np.transpose(np.stack((train_state_data, train_gt_data))),
                                                    batch_size)
        test_dataset = convert_pytype_to_tf_dataset(np.transpose(np.stack((test_state_data, test_gt_data))),
                                                batch_size)
        return train_dataset, test_dataset

    def train(self, train_dataset, GENERATE_TENSORBOARD=False):
        return train_sucker_model(GameParameters=self.GameParameters,
                           train_dataset=train_dataset,
                           GENERATE_TENSORBOARD=GENERATE_TENSORBOARD)

    def inference(self, sucker_model):
        run_sucker_model_inference(sucker_model=sucker_model, GameParameters=self.GameParameters)
