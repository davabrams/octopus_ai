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
        self.constraint_loss = ConstraintLoss(name='test_constraint_loss')
    
    def test_constraint_loss_initialization(self):
        """Test ConstraintLoss initialization"""
        self.assertEqual(self.constraint_loss.name, 'test_constraint_loss')
        self.assertIsInstance(self.constraint_loss, tf.keras.losses.Loss)
    
    def test_constraint_loss_call_perfect_match(self):
        """Test ConstraintLoss with perfect predictions"""
        # Create test data where predictions exactly match targets
        y_true = tf.constant([[0.5], [0.7], [0.3]], dtype=tf.float32)
        y_pred = tf.constant([[0.5], [0.7], [0.3]], dtype=tf.float32)
        
        loss = self.constraint_loss(y_true, y_pred)
        
        # Loss should be close to zero for perfect predictions
        self.assertLess(loss.numpy(), 1e-6)
    
    def test_constraint_loss_call_with_error(self):
        """Test ConstraintLoss with prediction errors"""
        # Create test data with prediction errors
        y_true = tf.constant([[0.5], [0.7], [0.3]], dtype=tf.float32)
        y_pred = tf.constant([[0.6], [0.8], [0.2]], dtype=tf.float32)
        
        loss = self.constraint_loss(y_true, y_pred)
        
        # Loss should be positive when there are errors
        self.assertGreater(loss.numpy(), 0.0)
        self.assertIsInstance(loss.numpy(), (float, np.float32))
    
    def test_constraint_loss_symmetric(self):
        """Test that ConstraintLoss is symmetric"""
        y_true = tf.constant([[0.4], [0.6]], dtype=tf.float32)
        y_pred = tf.constant([[0.6], [0.4]], dtype=tf.float32)
        
        loss1 = self.constraint_loss(y_true, y_pred)
        loss2 = self.constraint_loss(y_pred, y_true)
        
        # Loss should be the same regardless of order (if it's symmetric)
        # Note: This depends on the actual implementation of ConstraintLoss
        self.assertAlmostEqual(loss1.numpy(), loss2.numpy(), places=5)
    
    def test_constraint_loss_batch_processing(self):
        """Test ConstraintLoss with different batch sizes"""
        # Test with batch size 1
        y_true_1 = tf.constant([[0.5]], dtype=tf.float32)
        y_pred_1 = tf.constant([[0.6]], dtype=tf.float32)
        loss_1 = self.constraint_loss(y_true_1, y_pred_1)
        
        # Test with batch size 4
        y_true_4 = tf.constant([[0.5], [0.6], [0.7], [0.8]], dtype=tf.float32)
        y_pred_4 = tf.constant([[0.6], [0.7], [0.8], [0.9]], dtype=tf.float32)
        loss_4 = self.constraint_loss(y_true_4, y_pred_4)
        
        # Both should return valid loss values
        self.assertIsInstance(loss_1.numpy(), (float, np.float32))
        self.assertIsInstance(loss_4.numpy(), (float, np.float32))
        self.assertGreater(loss_1.numpy(), 0.0)
        self.assertGreater(loss_4.numpy(), 0.0)
    
    def test_constraint_loss_config(self):
        """Test ConstraintLoss serialization config"""
        config = self.constraint_loss.get_config()
        
        # Should return a dictionary with configuration
        self.assertIsInstance(config, dict)
        self.assertIn('name', config)
        self.assertEqual(config['name'], 'test_constraint_loss')
    
    def test_constraint_loss_from_config(self):
        """Test ConstraintLoss deserialization from config"""
        config = {'name': 'restored_constraint_loss'}
        
        restored_loss = ConstraintLoss.from_config(config)
        
        self.assertIsInstance(restored_loss, ConstraintLoss)
        self.assertEqual(restored_loss.name, 'restored_constraint_loss')
    
    def test_constraint_loss_edge_cases(self):
        """Test ConstraintLoss with edge case values"""
        # Test with zero values
        y_true_zero = tf.constant([[0.0], [0.0]], dtype=tf.float32)
        y_pred_zero = tf.constant([[0.0], [0.0]], dtype=tf.float32)
        loss_zero = self.constraint_loss(y_true_zero, y_pred_zero)
        self.assertLess(loss_zero.numpy(), 1e-6)
        
        # Test with values at boundary (0.0 and 1.0)
        y_true_bound = tf.constant([[0.0], [1.0]], dtype=tf.float32)
        y_pred_bound = tf.constant([[1.0], [0.0]], dtype=tf.float32)
        loss_bound = self.constraint_loss(y_true_bound, y_pred_bound)
        self.assertGreater(loss_bound.numpy(), 0.0)


class TestWeightedSumLoss(unittest.TestCase):
    """Test WeightedSumLoss custom loss function"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create some mock loss functions for testing
        self.mock_loss1 = Mock()
        self.mock_loss1.return_value = tf.constant(0.1, dtype=tf.float32)
        
        self.mock_loss2 = Mock()
        self.mock_loss2.return_value = tf.constant(0.2, dtype=tf.float32)
        
        self.weights = [0.7, 0.3]
        self.loss_functions = [self.mock_loss1, self.mock_loss2]
        
        self.weighted_sum_loss = WeightedSumLoss(
            loss_functions=self.loss_functions,
            weights=self.weights,
            name='test_weighted_sum'
        )
    
    def test_weighted_sum_loss_initialization(self):
        """Test WeightedSumLoss initialization"""
        self.assertEqual(self.weighted_sum_loss.name, 'test_weighted_sum')
        self.assertEqual(len(self.weighted_sum_loss.loss_functions), 2)
        self.assertEqual(self.weighted_sum_loss.weights, [0.7, 0.3])
    
    def test_weighted_sum_loss_call(self):
        """Test WeightedSumLoss computation"""
        y_true = tf.constant([[0.5], [0.6]], dtype=tf.float32)
        y_pred = tf.constant([[0.4], [0.7]], dtype=tf.float32)
        
        loss = self.weighted_sum_loss(y_true, y_pred)
        
        # Expected: 0.7 * 0.1 + 0.3 * 0.2 = 0.07 + 0.06 = 0.13
        expected_loss = 0.7 * 0.1 + 0.3 * 0.2
        self.assertAlmostEqual(loss.numpy(), expected_loss, places=5)
        
        # Verify that both loss functions were called
        self.mock_loss1.assert_called_once_with(y_true, y_pred)
        self.mock_loss2.assert_called_once_with(y_true, y_pred)
    
    def test_weighted_sum_loss_single_function(self):
        """Test WeightedSumLoss with single loss function"""
        single_loss = Mock()
        single_loss.return_value = tf.constant(0.5, dtype=tf.float32)
        
        weighted_loss = WeightedSumLoss(
            loss_functions=[single_loss],
            weights=[1.0],
            name='single_loss'
        )
        
        y_true = tf.constant([[0.5]], dtype=tf.float32)
        y_pred = tf.constant([[0.4]], dtype=tf.float32)
        
        loss = weighted_loss(y_true, y_pred)
        
        self.assertAlmostEqual(loss.numpy(), 0.5, places=5)
        single_loss.assert_called_once()
    
    def test_weighted_sum_loss_equal_weights(self):
        """Test WeightedSumLoss with equal weights"""
        equal_weighted_loss = WeightedSumLoss(
            loss_functions=self.loss_functions,
            weights=[0.5, 0.5],
            name='equal_weighted'
        )
        
        y_true = tf.constant([[0.3]], dtype=tf.float32)
        y_pred = tf.constant([[0.4]], dtype=tf.float32)
        
        loss = equal_weighted_loss(y_true, y_pred)
        
        # Expected: 0.5 * 0.1 + 0.5 * 0.2 = 0.05 + 0.1 = 0.15
        self.assertAlmostEqual(loss.numpy(), 0.15, places=5)
    
    def test_weighted_sum_loss_config(self):
        """Test WeightedSumLoss configuration"""
        config = self.weighted_sum_loss.get_config()
        
        self.assertIsInstance(config, dict)
        self.assertIn('name', config)
        self.assertIn('weights', config)
        self.assertEqual(config['name'], 'test_weighted_sum')
        self.assertEqual(config['weights'], [0.7, 0.3])
    
    def test_weighted_sum_loss_validation(self):
        """Test WeightedSumLoss input validation"""
        # Test mismatched weights and functions
        with self.assertRaises((ValueError, AssertionError)):
            WeightedSumLoss(
                loss_functions=[self.mock_loss1],
                weights=[0.7, 0.3],  # Too many weights
                name='invalid'
            )
    
    def test_weighted_sum_loss_zero_weight(self):
        """Test WeightedSumLoss with zero weight"""
        zero_weighted_loss = WeightedSumLoss(
            loss_functions=self.loss_functions,
            weights=[1.0, 0.0],  # Second loss has zero weight
            name='zero_weighted'
        )
        
        y_true = tf.constant([[0.5]], dtype=tf.float32)
        y_pred = tf.constant([[0.4]], dtype=tf.float32)
        
        loss = zero_weighted_loss(y_true, y_pred)
        
        # Expected: 1.0 * 0.1 + 0.0 * 0.2 = 0.1
        self.assertAlmostEqual(loss.numpy(), 0.1, places=5)


class TestLossIntegration(unittest.TestCase):
    """Integration tests for loss functions"""
    
    def test_constraint_loss_in_model(self):
        """Test ConstraintLoss integration with a simple model"""
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(2,))
        ])
        
        # Compile with ConstraintLoss
        constraint_loss = ConstraintLoss()
        model.compile(optimizer='adam', loss=constraint_loss, metrics=['mae'])
        
        # Create some dummy data
        X = np.random.random((10, 2)).astype(np.float32)
        y = np.random.random((10, 1)).astype(np.float32)
        
        # Should be able to fit without errors
        try:
            history = model.fit(X, y, epochs=1, verbose=0)
            self.assertIn('loss', history.history)
            self.assertGreater(len(history.history['loss']), 0)
        except Exception as e:
            self.fail(f"Model training with ConstraintLoss failed: {e}")
    
    def test_weighted_sum_loss_in_model(self):
        """Test WeightedSumLoss integration with a model"""
        # Create loss functions
        mse = tf.keras.losses.MeanSquaredError()
        mae = tf.keras.losses.MeanAbsoluteError()
        
        weighted_loss = WeightedSumLoss(
            loss_functions=[mse, mae],
            weights=[0.8, 0.2],
            name='combined_loss'
        )
        
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(3,))
        ])
        
        model.compile(optimizer='adam', loss=weighted_loss)
        
        # Test with dummy data
        X = np.random.random((20, 3)).astype(np.float32)
        y = np.random.random((20, 1)).astype(np.float32)
        
        try:
            history = model.fit(X, y, epochs=1, verbose=0)
            self.assertIn('loss', history.history)
            self.assertGreater(history.history['loss'][0], 0)
        except Exception as e:
            self.fail(f"Model training with WeightedSumLoss failed: {e}")
    
    def test_loss_function_gradients(self):
        """Test that loss functions produce proper gradients"""
        # Test with ConstraintLoss
        with tf.GradientTape() as tape:
            y_true = tf.constant([[0.5], [0.7]], dtype=tf.float32)
            y_pred = tf.Variable([[0.4], [0.8]], dtype=tf.float32)
            tape.watch(y_pred)
            
            constraint_loss = ConstraintLoss()
            loss = constraint_loss(y_true, y_pred)
        
        gradients = tape.gradient(loss, y_pred)
        
        # Gradients should exist and be non-zero
        self.assertIsNotNone(gradients)
        self.assertEqual(gradients.shape, y_pred.shape)
        # At least one gradient should be non-zero (unless perfect prediction)
        self.assertGreater(tf.reduce_sum(tf.abs(gradients)).numpy(), 0)


if __name__ == '__main__':
    unittest.main()