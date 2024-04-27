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

