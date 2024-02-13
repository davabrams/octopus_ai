"""
Trainer utilities
"""
import tensorflow as tf
from tensorflow.python.util.tf_export import keras_export  # pylint: disable=no-name-in-module
from keras.engine import base_layer
from keras import activations
from keras import backend
from keras import constraints
from keras import initializers
from keras import regularizers

class Trainer:
    """
    Trainer template class
    """
    def __init__(self):
        pass

    def datagen(self):
        """
        Should implement the datagen functionality
        """
        raise RuntimeError("datagen function not implemented")

    def data_format(self, data):
        """
        Should implement data formatter.  For
        example: train/test split, normalization, etc
        """
        raise RuntimeError("data_format function not implemented")

    def train(self):
        """
        Should implement the training functionality
        """
        raise RuntimeError("train function not implemented")

    def inference(self):
        """
        Should implement the inference functionality
        """
        raise RuntimeError("inference function not implemented")

    def eval(self):
        """
        Should implement the eval functionality
        """
        raise RuntimeError("eval function not implemented")


@keras_export("keras.layers.ConcurrentRNNCell")
class ConcurrentRNNCell(base_layer.BaseRandomLayer):
    """
    Cell class for ConcurrentRNN.
    """
    def __init__(
        self,
        units,
        activation="tanh",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        **kwargs,
    ):
        if units <= 0:
            raise ValueError(
                "Received an invalid value for argument `units`, "
                f"expected a positive integer, got {units}."
            )
        super().__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
        self.state_size = self.units
        self.output_size = self.units


@keras_export("keras.layers.ConcurrentRNN")
class ConcurrentRNN(base_layer.Layer):
    """
    Class for SimpleRNN.
    """


    def __init__(
        self,
        cell,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=False,
        time_major=False,
        **kwargs,
    ):
        if "state_size" not in dir(cell):
            raise ValueError(
                "The RNN cell should have a `state_size` attribute "
                "(tuple of integers, one integer per RNN state). "
                f"Received: cell={cell}"
            )
        self.zero_output_for_mask = kwargs.pop("zero_output_for_mask", False)
        if "input_shape" not in kwargs and (
            "input_dim" in kwargs or "input_length" in kwargs
        ):
            input_shape = (
                kwargs.pop("input_length", None),
                kwargs.pop("input_dim", None),
            )
            kwargs["input_shape"] = input_shape
        super().__init__()

        self.cell = cell
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll
        self.time_major = time_major

        self.supports_masking = True
        # The input shape is unknown yet, it could have nested tensor inputs,
        # and the input spec will be the list of specs for nested inputs, the
        # structure of the input_spec will be the same as the input.
        self.input_spec = None
        self.state_spec = None
        self._states = None
        self.constants_spec = None
        self._num_constants = 0
        
        if stateful:
            if tf.distribute.has_strategy():
                raise ValueError(
                    "Stateful RNNs (created with `stateful=True`) "
                    "are not yet supported with tf.distribute.Strategy."
                )
