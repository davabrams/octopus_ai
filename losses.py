import tensorflow as tf
from tensorflow import keras

@keras.saving.register_keras_serializable(package="Octo", name="ConstraintLoss")
class ConstraintLoss(tf.keras.losses.Loss):
    """ This is the loss that maintains our color change rate constraint.
    We have defined the color change rate to be a maximum of 0.25 per iteration.
    This means that there is a cost incurred if the predicted value is 0.25
    greater than or less than the original value. The GT value is never used. """
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
        assert tf.is_tensor(y_pred) and tf.is_tensor(y_true), "error: inputs must be tensors"

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
                    tf.summary.scalar('ConstraintLoss_diff', d, step = step)
                    i = i + 1
                i = 0
                for ov in original_value:
                    step = base_step + i
                    tf.summary.scalar('ConstraintLoss_original_value', ov, step = step)
                    i = i + 1
                i = 0
                for pv in predicted_value:
                    step = base_step + i
                    tf.summary.scalar('ConstraintLoss_predicted_value', pv, step = self.step)
                    i = i + 1
                i = 0
                for ep in excess_penalty:
                    step = base_step + i
                    tf.summary.scalar('ConstraintLoss_excess_penalty', ep, step = self.step)
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
        self.f1 = ConstraintLoss(threshold, step, logwriter)
        self.f2 = keras.losses.MeanSquaredError()
        self.f1_fields = 1
        self.f2_fields = 0
        self.w1 = weight
        self.w2 = tf.convert_to_tensor(1.0) - weight
        if logwriter:
            self.writer = logwriter
        self.step = step

    def call(self, y_true, y_pred):
        
        # the previous state is y_true[1] and is the only input used for this loss
        loss1 = self.f1(y_true[:,self.f1_fields], y_pred[:,0])
        # the ground truth is y_true[0] is the only input used for this loss
        loss2 = self.f2(y_true[:,self.f1_fields], y_pred[:,0])
        w_loss1 = tf.math.multiply(self.w1, loss1)
        w_loss2 = tf.math.multiply(self.w2, loss2)

        step = self.step * y_true.shape[0]

        if self.writer:
            with self.writer.as_default():
                tf.summary.scalar('WeightedSumLoss_w_loss1_(reduced)', w_loss1, step = step)
                tf.summary.scalar('WeightedSumLoss_w_loss2_(reduced)', w_loss2, step = step)
                tf.summary.scalar('WeightedSumLoss_loss1_(reduced)', loss1, step = step)
                tf.summary.scalar('WeightedSumLoss_loss2_(reduced)', loss2, step = step)
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
