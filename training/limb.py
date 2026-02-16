"""
Library for limb model training
"""
import os
import datetime
import pickle
import time
import numpy as np
import tensorflow as tf
import seaborn as sn
import matplotlib.pyplot as plt
from tensorflow import keras
from training.losses import WeightedSumLoss
from octo_datagen import OctoDatagen
from util import (
    train_test_split_multiple_state_vectors,
)
from simulator.simutil import MLMode
from .trainutil import Trainer


class LimbTrainer(Trainer):
    """
    Limb-level trainer class
    """
    def __init__(self, GameParameters, TrainingParameters):
        self.GameParameters = GameParameters
        self.TrainingParameters = TrainingParameters

    def datagen(self, SAVE_DATA_TO_DISK):
        datagen = OctoDatagen(self.GameParameters)
        data = datagen.run_color_datagen()
        ml_mode = self.TrainingParameters['ml_mode']
        datagen_path = self.TrainingParameters['datasets'][ml_mode]
        if SAVE_DATA_TO_DISK:
            with open(datagen_path, 'wb') as file:
                pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
        return data

    def data_format(self, data):
        """
        Format Data for Train and Val
        """
        datagen_data_format = data['game_parameters'][
            'datagen_data_write_format'
        ]
        print("Found data format:", datagen_data_format)
        if datagen_data_format == MLMode.SUCKER:
            # sucker's current color
            state_data = np.array([data['state_data']], dtype='float32')
        elif datagen_data_format == MLMode.LIMB:
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
                c_ragged_list.append(c_array)
                dist_ragged_list.append(dist_array)
        else:
            raise TypeError("Expected valid MLMode for training")

        state_data = [x['color'] for x in data['state_data']]
        gt_data = [x.r for x in data['gt_data']]

        (train_state_data, train_gt_data, test_state_data,
         test_gt_data) = train_test_split_multiple_state_vectors(
            [state_data, gt_data, c_ragged_list, dist_ragged_list], gt_data
        )

        def gen(data):
            for ix in range(len(data[0])):
                n = [[data[0][ix]], [data[1][ix]], data[2][ix], data[3][ix]]
                yield tf.ragged.constant(n, dtype=tf.float32)

        output_signature = tf.RaggedTensorSpec(
            shape=(4, None), dtype=tf.float32
        )
        train_dataset = tf.data.Dataset.from_generator(
            lambda: gen(train_state_data), output_signature=output_signature
        )
        test_dataset = tf.data.Dataset.from_generator(
            lambda: gen(test_state_data), output_signature=output_signature
        )

        return train_dataset, test_dataset

    def train(self, train_dataset, GENERATE_TENSORBOARD=False):
        return self.train_limb_model(
            GameParameters=self.GameParameters,
            TrainingParameters=self.TrainingParameters,
            train_dataset=train_dataset,
            GENERATE_TENSORBOARD=GENERATE_TENSORBOARD
        )

    def inference(self, sucker_model):
        """
        Runs a standard sweep inference on the input domain
        """
        assert sucker_model, "No model found, can't run inference"
        assert self.GameParameters, "No parameters found, can't run inference"

        # batch_size = GameParameters['batch_size']
        # max_hue_change = GameParameters['octo_max_hue_change']
        # constraint_loss_weight = GameParameters['constraint_loss_weight']

        # Iterate over domain space
        range_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
        res = []
        for curr in range_vals:
            row = []
            for gt in range_vals:
                fixed_test_input = np.array([[curr, gt]])
                ragged_test_input = np.array([[[0, 0]]])
                test_input = [fixed_test_input, ragged_test_input]

                # computes prediction output
                pred = sucker_model.predict(test_input, verbose=0)[0][0]
                row.append(pred)

            res.append(row)

        # Plot inference results
        plt.figure(figsize=(10, 7))
        sn.heatmap(res, annot=True)
        plt.xlabel('surface color')
        locs = plt.xticks()
        plt.xticks(locs, range_vals)
        plt.ylabel('sucker previous color')
        locs = plt.yticks()
        plt.yticks(locs, range_vals)
        plt.show()

    def train_limb_model(self, GameParameters, TrainingParameters,
                         train_dataset, GENERATE_TENSORBOARD=False):
        """
        Contains model construction, loss construction, and training loop.
        """
        batch_size = TrainingParameters['batch_size']

        # Configure loss function settings
        constraint_loss_weight = tf.constant(
            TrainingParameters['constraint_loss_weight'], dtype='float32'
        )
        max_hue_change = tf.constant(
            GameParameters['octo_max_hue_change'], dtype='float32'
        )

        # Model constructor
        limb_model = self._model_constructor()

        # Tensorboard configuration
        # tensorboard serve --logdir <log directory>
        summary_writer = []
        if GENERATE_TENSORBOARD:
            log_dir = os.path.join(
                "models/logs/limb/fit/",
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            )
            summary_writer = tf.summary.create_file_writer(logdir=log_dir)

        # Custom training loop
        epochs = TrainingParameters["epochs"]
        optimizer = keras.optimizers.SGD(learning_rate=1e-3)

        """
        Epoch = # of times we go over each element in the data set
        Batch = how many data points are passed through at once, before sgd
        Step = which index in the data set we are on, after the data is batched
        """

        start_time = time.time()
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))

            epoch_loss_mse = tf.keras.metrics.MeanSquaredError()

            # Iterate over the batches of the dataset.
            if not train_dataset:
                raise ValueError(
                    "train_dataset is empty, there is no way to train"
                )
            for step, train_data in enumerate(train_dataset):
                (state_data, gt_data, c_ragged_list,
                 dist_ragged_list) = train_data
                y_train_data = tf.stack([state_data, gt_data])
                fixed_train_input = tf.expand_dims(y_train_data, axis=0)
                ragged_train_input = tf.expand_dims(
                    tf.RaggedTensor.from_tensor(
                        tf.stack([c_ragged_list, dist_ragged_list], axis=1)
                    ), axis=0
                )

                limb_model.reset_states()

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                with tf.GradientTape() as tape:
                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.

                    # Logits for this minibatch
                    # https://en.wikipedia.org/wiki/Logit
                    logits = limb_model(
                        [fixed_train_input, ragged_train_input], training=True
                    )

                    # Recompile the loss function with the updated step
                    # (for logging)
                    loss_fn = WeightedSumLoss(
                        threshold=max_hue_change,
                        weight=constraint_loss_weight,
                        step=epoch,
                        logwriter=summary_writer
                    )

                    # Compute the loss value for this minibatch.
                    loss_value = loss_fn(y_train_data, logits)
                    epoch_loss_mse(y_train_data, logits)
                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to
                # the loss.
                grads = tape.gradient(
                    loss_value, limb_model.trainable_weights
                )

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(
                    zip(grads, limb_model.trainable_weights)
                )

                # Report every 100 steps (3,200 data points) to the console.
                if step % 100 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f | "
                        % (step, float(loss_value)),
                        f"Seen so far: {((step + 1) * batch_size)} samples"
                    )
            t = time.time()
            t_elapsed = t - start_time
            t_remaining = (t_elapsed / epoch) * epochs
            t_eta = start_time + t_remaining
            print(f"\nEpoch {epoch}/{epochs}")
            print(
                f"Time Elapsed: {t_elapsed}, Time Remaining: {t_remaining}, "
                f"ETA: {t_eta}\n\n"
            )

            if GENERATE_TENSORBOARD:
                with summary_writer.as_default():
                    # pylint: disable=not-callable
                    tf.summary.scalar(
                        'epoch_loss_mse', epoch_loss_mse.result(),
                        step=optimizer.iterations
                    )

        if GENERATE_TENSORBOARD:
            print(
                f"Tensorboard generated, run with:\n\n\t"
                f"tensorboard serve --logdir {log_dir}\n"
            )

        return limb_model
    
    def _model_constructor(self):
        # define two sets of inputs
        fixed_input = tf.keras.Input(
            shape=(2,), dtype=tf.float32, name="fixed_input", ragged=False
        )
        ragged_input = tf.keras.Input(
            shape=(None, 2), dtype=tf.float32, name="ragged_input", ragged=True
        )

        # the first branch operates on the fixed input
        fixed_layer_stack = tf.keras.layers.Dense(
            units=5, activation="relu", name="fixed_hidden_layer1"
        )(fixed_input)
        fixed_layer_stack = tf.keras.layers.Dense(
            units=5, activation="relu", name="fixed_hidden_layer2"
        )(fixed_layer_stack)
        fixed_layer_stack = tf.keras.layers.Dense(
            units=4, activation="relu", name="fixed_prediction_layer"
        )(fixed_layer_stack)
        fixed_model = tf.keras.Model(
            inputs=fixed_input, outputs=fixed_layer_stack,
            name="fixed_output_layer"
        )

        # the second branch operates on the ragged input
        ragged_layer_stack = tf.keras.layers.SimpleRNN(
            units=5, activation="relu", name="ragged_rnn_layer"
        )(ragged_input)
        ragged_layer_stack = tf.keras.layers.Dense(
            units=5, activation="relu", name="ragged_hidden_layer"
        )(ragged_layer_stack)
        ragged_layer_stack = tf.keras.layers.Dense(
            units=4, activation="relu", name="ragged_prediction_layer"
        )(ragged_layer_stack)
        ragged_model = tf.keras.Model(
            inputs=ragged_input, outputs=ragged_layer_stack,
            name="ragged_output_layer"
        )
        
        # combine the output of the two branches
        combined = tf.keras.layers.concatenate([
            fixed_model.output, ragged_model.output
        ])

        # apply a fully connected layer and then a regression prediction on the
        # combined outputs
        limb_model_output = tf.keras.layers.Dense(
            2, activation="relu", name="limb_hidden_layer1"
        )(combined)
        limb_model_output = tf.keras.layers.Dense(
            2, activation="relu", name="limb_hidden_layer2"
        )(limb_model_output)
        # limb_model_output = tf.keras.layers.Dense(
        #     2, activation="relu", name="limb_hidden_layer3"
        # )(limb_model_output)
        limb_model_output = tf.keras.layers.Dense(
            1, activation="linear", name="limb_model_output"
        )(limb_model_output)

        # our model will accept the inputs of the two branches and
        # then output a single value
        limb_model = tf.keras.Model(
            inputs=[fixed_model.input, ragged_model.input],
            outputs=limb_model_output
        )

        return limb_model
