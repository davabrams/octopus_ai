""" Octopus model training """
import os
import datetime
import pickle
import time as tm
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from simulator.octo_datagen import OctoDatagen
from OctoConfig import GameParameters
from util import (
    train_test_split,
    convert_pytype_to_tf_dataset,
    erase_all_logs
)
from losses import WeightedSumLoss


np.set_printoptions(precision=4)
tf.config.run_functions_eagerly(False)
batch_size = GameParameters['batch_size']

# %% Entry point for octopus modeling
RUN_DATAGEN = GameParameters['datagen_mode']
SAVE_DATA_TO_DISK = True

RESTORE_DATA_FROM_DISK = True
RUN_TRAINING = True
ERASE_OLD_TENSORBOARD_LOGS = True
GENERATE_TENSORBOARD = True
SAVE_MODEL_TO_DISK = True

RESTORE_MODEL_FROM_DISK = True
RUN_INFERENCE = True

RUN_EVAL = False

start = tm.time()
print(f"Octo Model started at {start}, setting t=0.0")

if ERASE_OLD_TENSORBOARD_LOGS:
    erase_all_logs()

# %% Data Gen
data = None
if RUN_DATAGEN:
    datagen = OctoDatagen(GameParameters)
    data = datagen.run_color_datagen()
    if SAVE_DATA_TO_DISK:
        with open('datagen/sucker_data.pkl', 'wb') as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
    print(f"Datagen completed at time t={tm.time() - start}")

# %% Model training
if RUN_TRAINING:
    ####### Import Data
    if RESTORE_DATA_FROM_DISK:
        del data
        with open('datagen/sucker_data.pkl', 'rb') as file:
            data = pickle.load(file)
    assert data, "No data found, can't run training."
    print(f"Training model with {len(data['gt_data'])} data points")

    ####### Format Data for Train and Val
    state_data = np.array([data['state_data']], dtype='float32') #sucker's current color
    gt_data = np.array([data['gt_data']], dtype='float32') #sucker's ground truth

    train_data, train_labels, val_data, val_labels = train_test_split(state_data, gt_data)

    train_dataset = convert_pytype_to_tf_dataset(np.transpose(np.stack((train_data, train_labels))),
                                                 batch_size)
    val_dataset = convert_pytype_to_tf_dataset(np.transpose(np.stack((val_data, val_labels))),
                                               batch_size)

    ####### Configure loss function settings
    constraint_loss_weight = GameParameters['constraint_loss_weight']
    logwriter = tf.summary.create_file_writer('logs')
    max_hue_change = tf.constant(GameParameters['octo_max_hue_change'], dtype='float32')

    ####### Model constructor
    inp = keras.layers.Input(shape=(None,2), batch_size=batch_size)
    outp = keras.layers.Dense(units=1, activation="sigmoid", name="prediction_layer")

    sucker_model = keras.Sequential()
    sucker_model.add(inp)
    sucker_model.add(keras.layers.Dense(units=3, activation="relu", name="hidden_layer1"))
    sucker_model.add(keras.layers.Dense(units=3, activation="relu", name="hidden_layer2"))
    sucker_model.add(keras.layers.Dense(units=3, activation="relu", name="hidden_layer3"))
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

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        epoch_loss_mse = tf.keras.metrics.MeanSquaredError()
        train_loss_results = []

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
                                          step=step,
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

            # Log every 20 batches.
            if step % 20 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * batch_size))

        if GENERATE_TENSORBOARD:
            with summary_writer.as_default():
                tf.summary.scalar('epoch_loss_mse', epoch_loss_mse.result(), step=optimizer.iterations)

    if GENERATE_TENSORBOARD:
        print(f"Tensorboard generated, run with:\n\n\ttensorboard serve --logdir {log_dir}\n")

    print(f"Model training completed at time t={tm.time() - start:.3f}")

    # %% Model deployment (this is only run if a new model was successfully trained)
    if SAVE_MODEL_TO_DISK:
        sucker_model.save('models/sucker.keras')
        print(f"Model deployment completed at time t={tm.time() - start:.3f}")

# %% Model inference
if RUN_INFERENCE:
    ####### Load model
    if RESTORE_MODEL_FROM_DISK:
        custom_objects = {"WeightedSumLoss": WeightedSumLoss}
        sucker_model = keras.models.load_model('models/sucker.keras', custom_objects)
    assert sucker_model, "No model found, can't run inference"

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

            #computes loss
            gt_inference_input = convert_pytype_to_tf_dataset(gt, batch_size)
            pred_inference_input = convert_pytype_to_tf_dataset(pred, batch_size)

            inference_losses = [WeightedSumLoss(threshold=max_hue_change,
                                    weight=constraint_loss_weight)]
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

    print(f"Model inference completed at time t={tm.time() - start:.3f}")

# %% Model eval
if RUN_EVAL:
    # For sucker color, eval is defined as the average of RMS of the RGB values
    # mean([rms(pred, gt) for each (pred, gt) in octopus])
    loss, accuracy = sucker_model.evaluate(x=tf.convert_to_tensor(val_data, dtype='float32'),
                                           y=tf.convert_to_tensor([val_labels], dtype='float32'),
                                           batch_size=GameParameters['batch_size'])

    print(f"Loss: {loss:.3f}, Accuracy: {accuracy:.3f}")

    print(f"Model eval completed at time t={tm.time() - start:.3f}")

# %% End and cleanup
print(f"octo AI completed at time t={tm.time() - start:.3f}")
end = tm.time()
print(f"tensorflow took: {end - start:.3f} seconds")
