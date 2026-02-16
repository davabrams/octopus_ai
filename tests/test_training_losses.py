"""
Unit tests for training loss functions
"""
import unittest
import numpy as np
import tensorflow as tf
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from training.losses import ConstraintLoss, WeightedSumLoss


class TestConstraintLoss(unittest.TestCase):
    """Test ConstraintLoss custom loss function"""

    def setUp(self):
        """Set up test fixtures"""
        self.constraint_loss = ConstraintLoss(threshold=0.25)

    def test_constraint_loss_initialization(self):
        """Test ConstraintLoss initialization"""
        self.assertEqual(self.constraint_loss.threshold, 0.25)
        self.assertIsInstance(self.constraint_loss, tf.keras.losses.Loss)

    def test_constraint_loss_call_perfect_match(self):
        """Test ConstraintLoss with perfect predictions (no change)"""
        y_true = tf.constant([0.5, 0.7, 0.3], dtype=tf.float32)
        y_pred = tf.constant([0.5, 0.7, 0.3], dtype=tf.float32)

        loss = self.constraint_loss(y_true, y_pred)

        # Loss should be zero when pred == true (no excess change)
        np.testing.assert_allclose(loss.numpy(), 0.0, atol=1e-6)

    def test_constraint_loss_call_with_error(self):
        """Test ConstraintLoss with prediction errors exceeding threshold"""
        # Diff of 0.5 exceeds threshold 0.25, so penalty = (0.5-0.25)^2 = 0.0625
        y_true = tf.constant([0.0], dtype=tf.float32)
        y_pred = tf.constant([0.5], dtype=tf.float32)

        loss = self.constraint_loss(y_true, y_pred)

        expected = (0.5 - 0.25) ** 2  # 0.0625
        np.testing.assert_allclose(loss.numpy(), expected, atol=1e-5)

    def test_constraint_loss_symmetric(self):
        """Test that ConstraintLoss is symmetric (depends on abs diff)"""
        y_true = tf.constant([0.4, 0.6], dtype=tf.float32)
        y_pred = tf.constant([0.8, 0.2], dtype=tf.float32)

        loss1 = self.constraint_loss(y_true, y_pred)
        loss2 = self.constraint_loss(y_pred, y_true)

        np.testing.assert_allclose(loss1.numpy(), loss2.numpy(), atol=1e-5)

    def test_constraint_loss_batch_processing(self):
        """Test ConstraintLoss with different batch sizes"""
        # Diff = 0.1, below threshold 0.25 -> penalty = 0
        y_true_1 = tf.constant([0.5], dtype=tf.float32)
        y_pred_1 = tf.constant([0.6], dtype=tf.float32)
        loss_1 = self.constraint_loss(y_true_1, y_pred_1)
        np.testing.assert_allclose(loss_1.numpy(), 0.0, atol=1e-6)

        # Diff = 0.4, above threshold -> penalty = (0.4-0.25)^2 = 0.0225
        y_true_2 = tf.constant([0.5], dtype=tf.float32)
        y_pred_2 = tf.constant([0.9], dtype=tf.float32)
        loss_2 = self.constraint_loss(y_true_2, y_pred_2)
        self.assertGreater(float(loss_2.numpy()), 0.0)

    def test_constraint_loss_config(self):
        """Test ConstraintLoss serialization config"""
        config = self.constraint_loss.get_config()

        self.assertIsInstance(config, dict)
        self.assertIn('threshold', config)
        self.assertEqual(config['threshold'], 0.25)

    def test_constraint_loss_from_config(self):
        """Test ConstraintLoss deserialization from config"""
        config = {'threshold': 0.3}

        restored_loss = ConstraintLoss.from_config(config)

        self.assertIsInstance(restored_loss, ConstraintLoss)
        self.assertEqual(restored_loss.threshold, 0.3)

    def test_constraint_loss_edge_cases(self):
        """Test ConstraintLoss with edge case values"""
        # Test with zero values (no diff -> no penalty)
        y_true_zero = tf.constant([0.0, 0.0], dtype=tf.float32)
        y_pred_zero = tf.constant([0.0, 0.0], dtype=tf.float32)
        loss_zero = self.constraint_loss(y_true_zero, y_pred_zero)
        np.testing.assert_allclose(loss_zero.numpy(), 0.0, atol=1e-6)

        # Test with max diff (0.0 vs 1.0 -> diff=1.0, penalty=(1.0-0.25)^2)
        y_true_bound = tf.constant([0.0], dtype=tf.float32)
        y_pred_bound = tf.constant([1.0], dtype=tf.float32)
        loss_bound = self.constraint_loss(y_true_bound, y_pred_bound)
        self.assertGreater(float(loss_bound.numpy()), 0.0)


class TestWeightedSumLoss(unittest.TestCase):
    """Test WeightedSumLoss custom loss function"""

    def setUp(self):
        """Set up test fixtures"""
        self.weighted_sum_loss = WeightedSumLoss(
            threshold=tf.convert_to_tensor(0.25),
            weight=tf.convert_to_tensor(0.7),
        )

    def test_weighted_sum_loss_initialization(self):
        """Test WeightedSumLoss initialization"""
        self.assertIsNotNone(self.weighted_sum_loss.f1)
        self.assertIsNotNone(self.weighted_sum_loss.f2)
        self.assertIsInstance(self.weighted_sum_loss, tf.keras.losses.Loss)

    def test_weighted_sum_loss_call(self):
        """Test WeightedSumLoss computation"""
        # y_true shape: (2, ...) where [0] = state data, [1] = gt data
        y_true = tf.constant([[0.5], [0.6]], dtype=tf.float32)
        y_pred = tf.constant([[0.55]], dtype=tf.float32)

        loss = self.weighted_sum_loss(y_true, y_pred)

        # Should return a scalar loss
        self.assertIsNotNone(loss)

    def test_weighted_sum_loss_single_function(self):
        """Test WeightedSumLoss with weight=1.0 (only constraint loss)"""
        loss_fn = WeightedSumLoss(
            threshold=tf.convert_to_tensor(0.25),
            weight=tf.convert_to_tensor(1.0),
        )

        y_true = tf.constant([[0.5], [0.6]], dtype=tf.float32)
        y_pred = tf.constant([[0.55]], dtype=tf.float32)

        loss = loss_fn(y_true, y_pred)
        self.assertIsNotNone(loss)

    def test_weighted_sum_loss_equal_weights(self):
        """Test WeightedSumLoss with equal weights"""
        loss_fn = WeightedSumLoss(
            threshold=tf.convert_to_tensor(0.25),
            weight=tf.convert_to_tensor(0.5),
        )

        y_true = tf.constant([[0.3], [0.4]], dtype=tf.float32)
        y_pred = tf.constant([[0.35]], dtype=tf.float32)

        loss = loss_fn(y_true, y_pred)
        self.assertIsNotNone(loss)

    def test_weighted_sum_loss_config(self):
        """Test WeightedSumLoss configuration"""
        config = self.weighted_sum_loss.get_config()

        self.assertIsInstance(config, dict)
        self.assertIn('threshold', config)

    def test_weighted_sum_loss_validation(self):
        """Test WeightedSumLoss accepts valid inputs"""
        # Should not raise with valid tensor inputs
        loss_fn = WeightedSumLoss(
            threshold=tf.convert_to_tensor(0.25),
            weight=tf.convert_to_tensor(0.5),
        )
        self.assertIsNotNone(loss_fn)

    def test_weighted_sum_loss_zero_weight(self):
        """Test WeightedSumLoss with zero constraint weight (only MAE)"""
        loss_fn = WeightedSumLoss(
            threshold=tf.convert_to_tensor(0.25),
            weight=tf.convert_to_tensor(0.0),
        )

        y_true = tf.constant([[0.5], [0.6]], dtype=tf.float32)
        y_pred = tf.constant([[0.55]], dtype=tf.float32)

        loss = loss_fn(y_true, y_pred)
        self.assertIsNotNone(loss)


class TestLossIntegration(unittest.TestCase):
    """Integration tests for loss functions"""

    def test_constraint_loss_in_model(self):
        """Test ConstraintLoss integration with a simple model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(2,))
        ])

        constraint_loss = ConstraintLoss()
        model.compile(optimizer='adam', loss=constraint_loss, metrics=['mae'])

        X = np.random.random((10, 2)).astype(np.float32)
        y = np.random.random((10, 1)).astype(np.float32)

        history = model.fit(X, y, epochs=1, verbose=0)
        self.assertIn('loss', history.history)
        self.assertGreater(len(history.history['loss']), 0)

    def test_weighted_sum_loss_in_model(self):
        """Test WeightedSumLoss can be instantiated for model use"""
        loss_fn = WeightedSumLoss(
            threshold=tf.convert_to_tensor(0.25),
            weight=tf.convert_to_tensor(0.8),
        )

        # Verify it's a valid loss object
        self.assertIsInstance(loss_fn, tf.keras.losses.Loss)

    def test_loss_function_gradients(self):
        """Test that loss functions produce proper gradients"""
        with tf.GradientTape() as tape:
            y_true = tf.constant([0.0], dtype=tf.float32)
            y_pred = tf.Variable([0.5], dtype=tf.float32)
            tape.watch(y_pred)

            constraint_loss = ConstraintLoss(threshold=0.25)
            loss = constraint_loss(y_true, y_pred)

        gradients = tape.gradient(loss, y_pred)

        self.assertIsNotNone(gradients)
        self.assertEqual(gradients.shape, y_pred.shape)
        # Diff = 0.5 > 0.25 threshold, so gradient should be non-zero
        self.assertGreater(tf.reduce_sum(tf.abs(gradients)).numpy(), 0)


if __name__ == '__main__':
    unittest.main()
