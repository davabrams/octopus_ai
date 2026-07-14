"""
Library for sucker model training
"""
import os
import datetime
import pickle
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from training.losses import (
    ClampedTargetLoss,
    DeltaColorLayer,
    WeightedSumLoss,
)
from training.trainutil import Trainer
from training.data_utils import (
    convert_pytype_to_tf_dataset,
    train_test_split,
)
from octo_datagen import OctoDatagen
from simulator.simutil import MLMode
from OctoConfig import as_config, default_datasets


class SuckerTrainer(Trainer):
    """Sucker-level trainer class.

    Takes a single Config. The hyperparams it needs (epochs, batch_size,
    constraint_loss_weight) live under cfg.training; the physical
    constraint threshold it trains against is cfg.octopus.sucker.
    """
    def __init__(self, cfg):
        self.cfg = as_config(cfg)

    def datagen(self, SAVE_DATA_TO_DISK):
        datagen = OctoDatagen(self.cfg)
        data = datagen.run_color_datagen()
        if SAVE_DATA_TO_DISK:
            # A Config built from a bare Config() has no paths table; fall
            # back to the shipped default, as the TrainingParameters=None
            # case used to.
            datagen_path = (self.cfg.paths.dataset_paths.get(MLMode.SUCKER)
                            or default_datasets[MLMode.SUCKER])
            with open(datagen_path, 'wb') as file:
                pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
        return data

    def data_format(self, data):
        """
        Format Data for Train and Val
        """
        batch_size = self.cfg.training.batch_size

        state_data = np.array([data['state_data']],
                              dtype='float32')  # sucker's current color
        gt_data = np.array([[c.r for c in data['gt_data']]],
                           dtype='float32')  # sucker's ground truth

        train_state_data, train_gt_data, test_state_data, test_gt_data = \
            train_test_split(state_data, gt_data)

        train_dataset = convert_pytype_to_tf_dataset(
            np.transpose(np.stack((train_state_data, train_gt_data))),
            batch_size)
        test_dataset = convert_pytype_to_tf_dataset(
            np.transpose(np.stack((test_state_data, test_gt_data))),
            batch_size)
        return train_dataset, test_dataset

    def train(self, train_dataset, GENERATE_TENSORBOARD=False):
        return self.train_sucker_model(
            cfg=self.cfg,
            train_dataset=train_dataset,
            GENERATE_TENSORBOARD=GENERATE_TENSORBOARD)

    def inference(self, sucker_model):
        """
        Runs a standard sweep inference on the input domain
        """
        assert sucker_model, "No model found, can't run inference"

        # Iterate over domain space
        range_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
        res = []
        for curr in range_vals:
            row = []
            for gt in range_vals:
                test_input = np.array([[curr, gt]])

                # computes prediction output
                pred = sucker_model.predict(test_input, verbose=0)[0][0]
                row.append(pred)

            res.append(row)

        # Plot inference results
        plt.figure(figsize=(10, 7))
        sn.heatmap(res, annot=True)
        plt.xlabel('surface color')
        locs, labels = plt.xticks()
        plt.xticks(locs, range_vals)
        plt.ylabel('sucker previous color')
        locs, labels = plt.yticks()
        plt.yticks(locs, range_vals)
        plt.show()

    def train_sucker_model(self, cfg, train_dataset,
                           GENERATE_TENSORBOARD=False):
        """
        Contains model construction, loss construction, and training loop.
        """
        batch_size = cfg.training.batch_size

        # Configure loss function settings
        constraint_loss_weight = cfg.training.constraint_loss_weight
        hue_change_limit = cfg.octopus.sucker.max_hue_change
        max_hue_change = tf.constant(hue_change_limit, dtype='float32')

        # Model constructor. Two architectures:
        # - delta (default via cfg.training.sucker_delta_model): the
        #   network predicts an unbounded raw value that DeltaColorLayer
        #   squashes into a legal color step around the previous color, so
        #   the physical constraint holds by construction and training
        #   uses ClampedTargetLoss (pure target matching).
        # - legacy: direct color prediction trained with WeightedSumLoss
        #   (soft constraint penalty + weak MAE pull).
        use_delta = bool(cfg.training.sucker_delta_model)
        if use_delta:
            inp = keras.layers.Input(shape=(2,))
            hidden = keras.layers.Dense(
                units=5, activation="relu", name="hidden_layer1")(inp)
            hidden = keras.layers.Dense(
                units=5, activation="relu", name="hidden_layer2")(hidden)
            hidden = keras.layers.Dense(
                units=5, activation="relu", name="hidden_layer3")(hidden)
            raw = keras.layers.Dense(
                units=1, activation="linear", name="raw_delta")(hidden)
            outp = DeltaColorLayer(
                max_hue_change=float(hue_change_limit),
                name="prediction_layer")([inp, raw])
            sucker_model = keras.Model(inputs=inp, outputs=outp)
            delta_loss_fn = ClampedTargetLoss(
                threshold=float(hue_change_limit))
        else:
            inp = keras.layers.Input(shape=(2,), batch_size=batch_size)
            outp = keras.layers.Dense(units=1, activation="linear",
                                      name="prediction_layer")

            sucker_model = keras.Sequential()
            sucker_model.add(inp)
            sucker_model.add(keras.layers.Dense(
                units=5, activation="relu", name="hidden_layer1"))
            sucker_model.add(keras.layers.Dense(
                units=5, activation="relu", name="hidden_layer2"))
            sucker_model.add(keras.layers.Dense(
                units=5, activation="relu", name="hidden_layer3"))
            sucker_model.add(outp)

        # Tensorboard configuration
        # tensorboard serve --logdir <log directory>
        summary_writer = []
        if GENERATE_TENSORBOARD:
            log_dir = os.path.join(
                "models/logs/sucker/fit/",
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            summary_writer = tf.summary.create_file_writer(logdir=log_dir)

        # Custom training loop
        epochs = cfg.training.epochs
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)

        total_steps = 0
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))

            epoch_loss_mse = tf.keras.metrics.MeanSquaredError()

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in \
                    enumerate(train_dataset):
                # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                with tf.GradientTape() as tape:
                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.

                    # Logits for this minibatch
                    # https://en.wikipedia.org/wiki/Logit
                    logits = sucker_model(x_batch_train, training=True)

                    if use_delta:
                        loss_fn = delta_loss_fn
                    else:
                        # Recompile the loss function with the updated
                        # step (for logging)
                        loss_fn = WeightedSumLoss(
                            threshold=max_hue_change,
                            weight=constraint_loss_weight,
                            step=total_steps,
                            logwriter=summary_writer)

                    # Compute the loss value for this minibatch.
                    loss_value = loss_fn(y_batch_train, logits)
                    epoch_loss_mse(y_batch_train, logits)
                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect
                # to the loss.
                grads = tape.gradient(loss_value,
                                      sucker_model.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(
                    zip(grads, sucker_model.trainable_weights))

                total_steps += 1

                # Log every 20 batches.
                if step % 20 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %s samples" %
                          ((step + 1) * batch_size))

                if total_steps % 100 == 0:
                    print(
                        f"Total Steps: {total_steps}, total data points: "
                        f"{total_steps * batch_size}")

            if GENERATE_TENSORBOARD:
                with summary_writer.as_default():
                    tf.summary.scalar(
                        'epoch_loss_mse', epoch_loss_mse.result(),
                        step=optimizer.iterations)  # disable=not-callable

        if GENERATE_TENSORBOARD:
            print(f"Tensorboard generated, run with:\n\n"
                  f"\ttensorboard serve --logdir {log_dir}\n")

        return sucker_model
