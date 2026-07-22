import numpy as np
import tensorflow as tf
import keras
from matplotlib import pyplot as plt


@keras.saving.register_keras_serializable(
    package="Octo", name="ConstraintLoss"
)
class ConstraintLoss(tf.keras.losses.Loss):
    """This is the loss that maintains our color change rate constraint.
    
    We have defined the color change rate to be a maximum of 0.25 per
    iteration. This means that there is a cost incurred if the predicted
    value is 0.25 greater than or less than the original value. The GT
    value is never used.
    """
    def __init__(self,
                 threshold=0.25,
                 step=0,
                 logwriter=None):
        super().__init__()

        self.threshold = threshold
        self.writer = logwriter
        self.step = step

    def call(self, y_true: tf.TensorArray, y_pred: tf.TensorArray):

        # Calculate absolute difference between predictions and original values
        assert tf.is_tensor(y_pred) and tf.is_tensor(y_true), (
            "error: inputs must be tensors"
        )

        original_value = y_true
        predicted_value = y_pred

        diff = tf.abs(predicted_value - original_value)

        # Apply threshold and square for stronger penalty
        excess_penalty = tf.where(diff > self.threshold,
                                  tf.square(diff - self.threshold),
                                  tf.zeros_like(diff))

        if self.writer:
            with self.writer.as_default():
                base_step = self.step
                i = 0
                for d in diff:
                    step = base_step + i
                    tf.summary.scalar('ConstraintLoss_diff', d, step=step)
                    i = i + 1
                i = 0
                for ov in original_value:
                    step = base_step + i
                    tf.summary.scalar(
                        'ConstraintLoss_original_value', ov, step=step
                    )
                    i = i + 1
                i = 0
                for pv in predicted_value:
                    step = base_step + i
                    tf.summary.scalar(
                        'ConstraintLoss_predicted_value', pv, step=self.step
                    )
                    i = i + 1
                i = 0
                for ep in excess_penalty:
                    step = base_step + i
                    tf.summary.scalar(
                        'ConstraintLoss_excess_penalty', ep, step=self.step
                    )
                    i = i + 1

        # Return excess penalty as the loss
        return excess_penalty

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


@keras.saving.register_keras_serializable(
    package="Octo", name="WeightedSumLoss"
)
class WeightedSumLoss(tf.keras.losses.Loss):
    """Takes the weighted sum of two loss functions"""
    def __init__(self,
                 threshold=tf.convert_to_tensor(0.25),
                 weight=tf.convert_to_tensor(0.99),
                 step=0,
                 logwriter=None):
        super().__init__()
        self.threshold = threshold
        self.f1 = ConstraintLoss(threshold, step, logwriter)
        # self.f2 = keras.losses.MeanSquaredError()
        self.f2 = keras.losses.MeanAbsoluteError()
        self.f1_fields = 0  # state data
        self.f2_fields = 1  # ground truth data
        self.w1 = weight
        self.w2 = tf.convert_to_tensor(1.0) - weight
        self.step = step
        self.writer = logwriter
        if logwriter:
            tf.summary.scalar(
                'WeightedSumLoss_threshold', threshold, step=step
            )
            tf.summary.scalar('WeightedSumLoss_weight', weight, step=step)

    def call(self, y_true, y_pred):
        # the previous state is y_true[1] and is the only input used for
        # this loss
        loss1 = self.f1(y_true[self.f1_fields], y_pred[0])
        # the ground truth is y_true[0] is the only input used for this loss
        loss2 = self.f2(
            tf.expand_dims(y_true[self.f2_fields], axis=0), y_pred[0]
        )
        w_loss1 = tf.multiply(self.w1, loss1)
        w_loss2 = tf.multiply(self.w2, loss2)

        # y_true.shape[0] can be None for symbolic tensors (e.g. inside
        # model.fit); fall back to 1 so logging-step math never crashes
        batch_dim = y_true.shape[0] if y_true.shape[0] is not None else 1
        step = self.step * batch_dim

        if self.writer:
            with self.writer.as_default():
                tf.summary.scalar(
                    'WeightedSumLoss_w_loss1_(reduced)', w_loss1, step=step
                )
                tf.summary.scalar(
                    'WeightedSumLoss_w_loss2_(reduced)', w_loss2, step=step
                )
                tf.summary.scalar(
                    'WeightedSumLoss_loss1_(reduced)', loss1, step=step
                )
                tf.summary.scalar(
                    'WeightedSumLoss_loss2_(reduced)', loss2, step=step
                )
                self.writer.flush()

        # Return w1*loss1 + w2*loss2 as the loss
        return tf.math.add(w_loss1, w_loss2)

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


@keras.saving.register_keras_serializable(
    package="Octo", name="DeltaColorLayer"
)
class DeltaColorLayer(keras.layers.Layer):
    """Output layer that enforces the color-change constraint by
    construction.

    Takes [model_input, raw_head] where model_input is (batch, 2) rows of
    [previous_color, surface_color] and raw_head is an unbounded (batch, 1)
    dense output. Produces:

        prediction = clip(prev + max_hue_change * tanh(raw), 0, 1)

    so every prediction is within +/- max_hue_change of the previous color
    and inside [0, 1] regardless of the network weights. With the
    constraint guaranteed architecturally, training can use a plain
    target-matching loss (see ClampedTargetLoss) instead of a weighted
    constraint penalty.
    """
    def __init__(self, max_hue_change=0.25, **kwargs):
        super().__init__(**kwargs)
        self.max_hue_change = float(max_hue_change)

    def call(self, inputs):
        model_input, raw = inputs
        prev = model_input[:, 0:1]
        # CONDITIONED mode: if the input carries a third column it is the
        # per-example max_hue_change (budget), so the same layer honours any
        # budget passed at inference. Otherwise use the fixed build-time value.
        # Either way delta = budget * tanh(raw), so the constraint holds by
        # construction (see sucker_hue_change_conditioned).
        if model_input.shape[-1] is not None and model_input.shape[-1] >= 3:
            budget = model_input[:, 2:3]
        else:
            budget = self.max_hue_change
        delta = budget * tf.tanh(raw)
        return tf.clip_by_value(prev + delta, 0.0, 1.0)

    def get_config(self):
        config = super().get_config()
        config.update({"max_hue_change": self.max_hue_change})
        return config


@keras.saving.register_keras_serializable(
    package="Octo", name="ClampedTargetLoss"
)
class ClampedTargetLoss(tf.keras.losses.Loss):
    """MSE against the constraint-clamped target.

    y_true is the sucker pipeline's (batch, 2) packing with columns
    [previous_color, surface_color]. The target is the best *legal* color
    for this step:

        target = clip(prev + clip(surf - prev, +/- threshold), 0, 1)

    i.e. exactly what the heuristic would produce. Because the target
    never asks for a step the constraint forbids, this loss does not fight
    a constraint penalty and the full gradient budget goes to target
    matching. Intended for models whose output already satisfies the
    constraint (e.g. DeltaColorLayer).

    Note the explicit column indexing (y_true[:, 0]): WeightedSumLoss
    indexes rows (y_true[0]), which matches the limb pipeline's
    (fields, ...) format but scrambles the sucker pipeline's (batch, 2)
    format.
    """
    def __init__(self, threshold=0.25, name="clamped_target_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = float(threshold)

    def call(self, y_true, y_pred):
        prev = y_true[:, 0]
        surf = y_true[:, 1]
        # CONDITIONED training: a third column is the per-example budget, so the
        # legal target is clamped at that sample's own max_hue_change (matching
        # the conditioned DeltaColorLayer). Otherwise use the fixed threshold.
        if y_true.shape[-1] is not None and y_true.shape[-1] >= 3:
            thr = y_true[:, 2]
        else:
            thr = self.threshold
        step = tf.clip_by_value(surf - prev, -thr, thr)
        target = tf.clip_by_value(prev + step, 0.0, 1.0)
        pred = tf.squeeze(y_pred, axis=-1)
        return tf.square(pred - target)

    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config


def plot_loss_functions(cl, mae, wsl, cfg, state_value,
                        target_value):
    """
    plots the loss and gradient for these three loss functions
    over an input sweep
    """
    constraint_weight = cfg.training.constraint_loss_weight
    threshold = cfg.octopus.sucker.max_hue_change

    def get_values_to_plot(loss, y_pred, y_true):
        print(loss.name)
        x_vals = []
        y_vals = []
        for ix, pred in enumerate(range(len(y_pred))):
            x_vals.append(pred)
            y_vals.append(float(loss(y_true[ix], y_pred[ix])))
        return x_vals, y_vals

    y_pred_raw = tf.expand_dims(
        tf.convert_to_tensor(np.linspace(0.0, 1.0, 1000), dtype='float32'),
        axis=1
    )
    y_true_gt = tf.multiply(
        tf.ones_like(y_pred_raw, dtype='float32'), target_value
    )  # Ground truth (target) value
    y_true_st = tf.multiply(
        tf.ones_like(y_pred_raw, dtype='float32'), state_value
    )  # Ground truth (target) value
    y_pred = tf.convert_to_tensor([[x] for x in y_pred_raw])
    y_true = tf.convert_to_tensor(
        list([
            [x] for x in zip(
                [float(x) for x in y_true_st], [float(x) for x in y_true_gt]
            )
        ])
    )

    # This loss considers the predicted value and the previous value
    plt.subplot(2, 2, 1)
    plt.title(
        f'Constraint Loss\nTarget value (GT)={target_value}, '
        f'State (previous) value={state_value}'
    )
    loss = cl(
        threshold=tf.convert_to_tensor(threshold, dtype='float32'),
        step=0,
        logwriter=None
    )
    x_vals, y_vals = get_values_to_plot(loss, y_pred[:, 0], y_true_st)
    plt.plot(
        tf.divide(x_vals, 1000),
        tf.multiply(y_vals, constraint_weight),
        color='y'
    )
    y_vals_1 = y_vals

    # This loss considers the predicted value and the ground truth value
    plt.subplot(2, 2, 2)
    plt.title(
        f'Mean Absolute Error\nTarget value (GT)={target_value}, '
        f'State (previous) value={state_value}'
    )
    loss = mae()
    x_vals, y_vals = get_values_to_plot(loss, y_pred[:, 0], y_true_gt)
    plt.plot(
        tf.divide(x_vals, 1000),
        tf.multiply(y_vals, 1 - constraint_weight),
        color='b'
    )
    y_vals_2 = y_vals

    loss = wsl(
        threshold=tf.convert_to_tensor(threshold, dtype='float32'),
        weight=tf.convert_to_tensor(constraint_weight),
        step=0,
        logwriter=None
    )
    plt.subplot(2, 2, 3)
    plt.title(
        f'Weighted Sum Loss\nTarget value (GT)={target_value}, '
        f'State (previous) value={state_value}'
    )
    x_vals, y_vals = get_values_to_plot(loss, y_pred, y_true)
    plt.plot(tf.divide(x_vals, 1000), y_vals, color='g')
    y_vals_3 = y_vals

    plt.subplot(2, 2, 4)
    plt.plot(
        tf.divide(x_vals, 1000),
        tf.multiply(y_vals_1, constraint_weight),
        color='y'
    )
    plt.plot(
        tf.divide(x_vals, 1000),
        tf.multiply(y_vals_2, 1 - constraint_weight),
        color='b'
    )
    plt.plot(tf.divide(x_vals, 1000), y_vals_3, color='g')

    plt.title(
        f'Combined Losses\nTarget value (GT)={target_value}, '
        f'State (previous) value={state_value}'
    )

    plt.plot(tf.divide(x_vals, 1000), y_vals, color='g')

    plt.title(
        f'Target value (GT)={target_value}\n'
        f'State (previous) value={state_value}'
    )
    plt.xlabel('predicted value')
    plt.ylabel('loss')

    plt.show()


# Use this to generate the loss plots.
if False:
    from octopus_ai.config import DEFAULT
    previous_value = 0.0
    target_value = 0.0
    plot_loss_functions(ConstraintLoss,
                        keras.losses.MeanAbsoluteError,
                        WeightedSumLoss,
                        DEFAULT,
                        previous_value,
                        target_value)
