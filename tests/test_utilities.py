"""
Unit tests for utility functions in util.py
"""
import unittest
import numpy as np
import tensorflow as tf
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import Optional, Union

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from util import (
    train_test_split, train_test_split_multiple_state_vectors,
    octo_norm, convert_pytype_to_tf_dataset, erase_all_logs,
)
from simulator.simutil import MLMode


class TestTrainTestSplit(unittest.TestCase):
    """Test train_test_split function.

    Note: train_test_split expects (state_data, gt_data) where both are
    2D numpy arrays with shape (num_fields, num_samples). The split index
    is computed as int(num_samples * test_size), meaning test_size determines
    the *first* portion, and the remainder is the second portion.
    Return order: train_state, train_gt, test_state, test_gt
    """

    def test_train_test_split_basic(self):
        """Test basic train/test split functionality"""
        # Shape: (1, 100) - 1 field, 100 samples
        state_data = np.random.random((1, 100)).astype(np.float32)
        gt_data = np.random.random((1, 100)).astype(np.float32)

        train_state, train_gt, test_state, test_gt = train_test_split(
            state_data, gt_data, test_size=0.2
        )

        # test_size=0.2 -> split_point = 20, train=first 20, test=remaining 80
        self.assertEqual(len(train_state), 20)
        self.assertEqual(len(test_state), 80)
        self.assertEqual(len(train_gt), 20)
        self.assertEqual(len(test_gt), 80)

    def test_train_test_split_different_test_sizes(self):
        """Test different test sizes"""
        state_data = np.random.random((1, 100)).astype(np.float32)
        gt_data = np.random.random((1, 100)).astype(np.float32)

        # Test 30% split
        train_state, train_gt, test_state, test_gt = train_test_split(
            state_data, gt_data, test_size=0.3
        )
        self.assertEqual(len(train_state), 30)
        self.assertEqual(len(test_state), 70)

        # Test 50% split
        train_state, train_gt, test_state, test_gt = train_test_split(
            state_data, gt_data, test_size=0.5
        )
        self.assertEqual(len(train_state), 50)
        self.assertEqual(len(test_state), 50)

    def test_train_test_split_reproducibility(self):
        """Test that splits produce consistent total sizes"""
        state_data = np.random.random((1, 50)).astype(np.float32)
        gt_data = np.random.random((1, 50)).astype(np.float32)

        train_state, train_gt, test_state, test_gt = train_test_split(
            state_data, gt_data, test_size=0.2
        )

        # Total should equal original
        self.assertEqual(len(train_state) + len(test_state), 50)
        self.assertEqual(len(train_gt) + len(test_gt), 50)

    def test_train_test_split_multiple_vectors(self):
        """Test multiple state vectors splitting"""
        state_data = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
        gt_data = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.05]

        train_data, train_gt, test_data, test_gt = \
            train_test_split_multiple_state_vectors(state_data, gt_data, test_size=0.2)

        # Should preserve total count
        self.assertEqual(len(train_data[0]) + len(test_data[0]), 10)
        self.assertEqual(len(train_gt) + len(test_gt), 10)


class TestOctoNorm(unittest.TestCase):
    """Test octopus normalization function - converts [0,1] to [-1,1]"""

    def test_octo_norm_basic(self):
        """Test basic normalization from [0,1] to [-1,1]"""
        vec = np.array([0.0, 0.5, 1.0])
        result = octo_norm(vec)
        expected = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_octo_norm_reverse(self):
        """Test reverse normalization from [-1,1] to [0,1]"""
        vec = np.array([-1.0, 0.0, 1.0])
        result = octo_norm(vec, reverse=True)
        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_octo_norm_single_element(self):
        """Test normalization with single element"""
        vec = np.array([0.25])
        result = octo_norm(vec)
        expected = np.array([-0.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_octo_norm_round_trip(self):
        """Test that forward and reverse operations are inverses"""
        original = np.array([0.2, 0.7, 0.9])
        normalized = octo_norm(original)
        restored = octo_norm(normalized, reverse=True)
        np.testing.assert_array_almost_equal(original, restored)


class TestTensorFlowConversion(unittest.TestCase):
    """Test TensorFlow dataset conversion.

    Note: convert_pytype_to_tf_dataset takes (input_np_array, batch_size)
    and returns a tf.data.Dataset of (x, x) pairs (self-supervised).
    """

    def test_convert_pytype_to_tf_dataset_basic(self):
        """Test basic conversion to TensorFlow dataset"""
        data = np.random.random((50, 2)).astype(np.float32)

        dataset = convert_pytype_to_tf_dataset(data, batch_size=10)

        self.assertIsInstance(dataset, tf.data.Dataset)

        batch_count = 0
        for batch_x, batch_y in dataset:
            batch_count += 1
            self.assertEqual(batch_x.shape[1], 2)

        self.assertEqual(batch_count, 5)

    def test_convert_pytype_to_tf_dataset_different_batch_sizes(self):
        """Test conversion with different batch sizes"""
        data = np.random.random((30, 2)).astype(np.float32)

        dataset = convert_pytype_to_tf_dataset(data, batch_size=5)
        batch_count = sum(1 for _ in dataset)
        self.assertEqual(batch_count, 6)

        dataset = convert_pytype_to_tf_dataset(data, batch_size=1)
        batch_count = sum(1 for _ in dataset)
        self.assertEqual(batch_count, 30)


class TestLogErasure(unittest.TestCase):
    """Test log erasure functionality"""

    def test_erase_all_logs(self):
        """Test log erasure runs without error"""
        # erase_all_logs uses hardcoded paths relative to cwd,
        # just verify it doesn't crash when dirs don't exist
        erase_all_logs()


class TestDefaultLoader(unittest.TestCase):
    """Test DefaultLoader abstract base class"""

    def test_confirm_file_exists_valid_file(self):
        """Test file existence confirmation with valid file"""
        from training.models.base_loader import DefaultLoader

        class TestLoader(DefaultLoader):
            def _load(self, **kwargs):
                pass

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(b"test content")

        try:
            loader = TestLoader(temp_path)
            self.assertEqual(loader.path, temp_path)
        finally:
            os.unlink(temp_path)

    def test_confirm_file_exists_invalid_file(self):
        """Test file existence confirmation with invalid file"""
        from training.models.base_loader import DefaultLoader

        class TestLoader(DefaultLoader):
            def _load(self, **kwargs):
                pass

        with self.assertRaises(FileNotFoundError):
            TestLoader("/nonexistent/path/file.txt")


class TestDataValidation(unittest.TestCase):
    """Test data validation and edge cases"""

    def test_train_test_split_empty_data(self):
        """Test train_test_split with empty data returns empty arrays"""
        state_data = np.array([[]]).reshape(1, 0)
        gt_data = np.array([[]]).reshape(1, 0)

        # Empty data is handled gracefully, returning empty arrays
        train_state, train_gt, test_state, test_gt = train_test_split(
            state_data, gt_data, test_size=0.2)
        self.assertEqual(len(train_state), 0)
        self.assertEqual(len(test_state), 0)

    def test_train_test_split_mismatched_sizes(self):
        """Test train_test_split rejects non-numpy input"""
        with self.assertRaises(TypeError):
            train_test_split([1, 2, 3], [4, 5, 6], test_size=0.2)


if __name__ == '__main__':
    unittest.main()
