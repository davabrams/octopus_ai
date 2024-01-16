"""
Library for limb model training
"""
import pickle
import numpy as np
import tensorflow as tf
from octo_datagen import OctoDatagen
from util import (
    train_test_split_multiple_state_vectors,
    run_sucker_model_inference
)
from simulator.simutil import MLMode
from training.train import train_limb_model
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
        if self.GameParameters['ml_mode'] == MLMode.SUCKER:
            state_data = np.array([data['state_data']], dtype='float32') #sucker's current color
        elif self.GameParameters['ml_mode'] == MLMode.LIMB:
            c_ragged_list = []
            dist_ragged_list = []
            # Iterate over data points
            for state in data['state_data']:
                # Iterate over adjacent nodes
                c_array = []
                dist_array = []
                for adj in state['adjacents']:
                    S = adj[0]
                    c = S.c.r
                    dist = adj[1]
                    c_array.append(c)
                    dist_array.append(dist)
                    # TODO(davabrams) : normalize or standardize this
                c_ragged_list.append(tf.constant(c_array, dtype='float32'))
                dist_ragged_list.append(tf.constant(dist_array, dtype='float32'))
        else:
            raise TypeError("Expected valid MLMode for training")

        state_data = [x['color'] for x in data['state_data']]
        gt_data = [x.r for x in data['gt_data']]

        train_state_data, train_gt_data, test_state_data, test_gt_data = train_test_split_multiple_state_vectors([state_data, gt_data, c_ragged_list, dist_ragged_list], gt_data)

        def gen(data: list):
            for ix in range(len(data[0])):
                yield data[0][ix], data[1][ix], tf.ragged.constant(data[2][ix], dtype='float32'), tf.ragged.constant(data[3][ix], dtype='float32')

        output_signature = (tf.TensorSpec(shape=(), dtype=tf.float32),
                            tf.TensorSpec(shape=(), dtype=tf.float32),
                            tf.RaggedTensorSpec(shape=(1, None), dtype=tf.float32),
                            tf.RaggedTensorSpec(shape=(1, None), dtype=tf.float32),)
        test_gen = gen(test_state_data)
        test_dataset = tf.data.Dataset.from_generator(lambda: map(tuple, test_gen), output_signature=output_signature)
        train_gen = gen(train_state_data)
        train_dataset = tf.data.Dataset.from_generator(lambda: map(tuple, train_gen), output_signature=output_signature)
        return train_dataset, test_dataset

    def train(self, train_dataset, GENERATE_TENSORBOARD=False):
        return train_limb_model(GameParameters=self.GameParameters,
                           train_dataset=train_dataset,
                           GENERATE_TENSORBOARD=GENERATE_TENSORBOARD)

    def inference(self, sucker_model):
        run_sucker_model_inference(sucker_model=sucker_model, GameParameters=self.GameParameters)
