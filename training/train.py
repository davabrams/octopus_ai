import os
import datetime
import tensorflow as tf
from tensorflow import keras
from training.losses import WeightedSumLoss

def train_sucker_model(GameParameters, train_dataset, GENERATE_TENSORBOARD=False):
    """
    Contains model construction, loss construction, and training loop.
    """
    batch_size = GameParameters['batch_size']

    ####### Configure loss function settings
    constraint_loss_weight = GameParameters['constraint_loss_weight']
    max_hue_change = tf.constant(GameParameters['octo_max_hue_change'], dtype='float32')

    ####### Model constructor
    inp = keras.layers.Input(shape=(None,2), batch_size=batch_size)
    outp = keras.layers.Dense(units=1, activation="linear", name="prediction_layer")

    sucker_model = keras.Sequential()
    sucker_model.add(inp)
    sucker_model.add(keras.layers.Dense(units=5, activation="relu", name="hidden_layer1"))
    sucker_model.add(keras.layers.Dense(units=5, activation="relu", name="hidden_layer2"))
    sucker_model.add(keras.layers.Dense(units=5, activation="relu", name="hidden_layer3"))
    sucker_model.add(outp)

    ####### Tensorboard configuration
    # tensorboard serve --logdir <log directory>
    summary_writer = []
    if GENERATE_TENSORBOARD:
        log_dir = os.path.join("models/logs/sucker/fit/", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        summary_writer = tf.summary.create_file_writer(logdir=log_dir)

    ####### Custom training loop
    epochs = GameParameters["epochs"]
    optimizer = keras.optimizers.SGD(learning_rate=1e-3)

    total_steps = 0
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        epoch_loss_mse = tf.keras.metrics.MeanSquaredError()

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            sucker_model.reset_states()

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

                # Recompile the loss function with the updated step (for logging)
                loss_fn = WeightedSumLoss(threshold=max_hue_change,
                                          weight=constraint_loss_weight,
                                          step=total_steps,
                                          logwriter=summary_writer)

                # Compute the loss value for this minibatch.
                loss_value = loss_fn(y_batch_train, logits)
                epoch_loss_mse(y_batch_train, logits)
            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, sucker_model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, sucker_model.trainable_weights))

            total_steps += 1

            # Log every 20 batches.
            if step % 20 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * batch_size))

            if total_steps % 100 == 0:
                print(f"Total Steps: {total_steps}, total data points: {total_steps * batch_size}")

        if GENERATE_TENSORBOARD:
            with summary_writer.as_default():
                tf.summary.scalar('epoch_loss_mse', epoch_loss_mse.result(), step=optimizer.iterations)

    if GENERATE_TENSORBOARD:
        print(f"Tensorboard generated, run with:\n\n\ttensorboard serve --logdir {log_dir}\n")

    return sucker_model
