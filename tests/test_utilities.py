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
    DefaultLoader
)
from simulator.simutil import MLMode


class TestTrainTestSplit(unittest.TestCase):
    """Test train_test_split functions"""
    
    def test_train_test_split_basic(self):
        """Test basic train/test split functionality"""
        # Create sample data
        X = np.random.random((100, 5))
        y = np.random.random((100, 1))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Check shapes
        self.assertEqual(X_train.shape[0], 80)
        self.assertEqual(X_test.shape[0], 20)
        self.assertEqual(y_train.shape[0], 80)
        self.assertEqual(y_test.shape[0], 20)
        
        # Check that all original data is preserved
        self.assertEqual(X_train.shape[0] + X_test.shape[0], X.shape[0])
        self.assertEqual(y_train.shape[0] + y_test.shape[0], y.shape[0])
    
    def test_train_test_split_different_test_sizes(self):
        """Test different test sizes"""
        X = np.random.random((100, 3))
        y = np.random.random((100, 1))
        
        # Test 30% split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.assertEqual(X_train.shape[0], 70)
        self.assertEqual(X_test.shape[0], 30)
        
        # Test 50% split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        self.assertEqual(X_train.shape[0], 50)
        self.assertEqual(X_test.shape[0], 50)
    
    def test_train_test_split_reproducibility(self):
        """Test that same random state gives same results"""
        X = np.random.random((50, 2))
        y = np.random.random((50, 1))
        
        # First split
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=123)
        
        # Second split with same random state
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=123)
        
        # Results should be identical
        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)
        np.testing.assert_array_equal(y_train1, y_train2)
        np.testing.assert_array_equal(y_test1, y_test2)
    
    def test_train_test_split_multiple_vectors(self):
        """Test multiple state vectors splitting"""
        # Create sample data with multiple state vectors
        data = [
            np.random.random((20, 3)),  # First vector
            np.random.random((15, 3)),  # Second vector
            np.random.random((25, 3))   # Third vector
        ]
        
        train_data, test_data = train_test_split_multiple_state_vectors(data, test_size=0.2, random_state=42)
        
        # Should have same number of vectors
        self.assertEqual(len(train_data), len(data))
        self.assertEqual(len(test_data), len(data))
        
        # Check that split ratios are approximately correct
        for i, (train_vec, test_vec, orig_vec) in enumerate(zip(train_data, test_data, data)):
            total_size = len(train_vec) + len(test_vec)
            self.assertEqual(total_size, len(orig_vec))
            
            # Test size should be approximately 20%
            test_ratio = len(test_vec) / total_size
            self.assertAlmostEqual(test_ratio, 0.2, delta=0.1)  # Allow some variance for small datasets


class TestOctoNorm(unittest.TestCase):
    """Test octopus normalization function - converts [0,1] to [-1,1]"""
    
    def test_octo_norm_basic(self):
        """Test basic normalization from [0,1] to [-1,1]"""
        # Test with values in [0,1] range
        vec = np.array([0.0, 0.5, 1.0])
        result = octo_norm(vec)
        expected = np.array([-1.0, 0.0, 1.0])  # 0->-1, 0.5->0, 1->1
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_octo_norm_reverse(self):
        """Test reverse normalization from [-1,1] to [0,1]"""
        vec = np.array([-1.0, 0.0, 1.0])
        result = octo_norm(vec, reverse=True)
        expected = np.array([0.0, 0.5, 1.0])  # -1->0, 0->0.5, 1->1
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_octo_norm_single_element(self):
        """Test normalization with single element"""
        vec = np.array([0.25])
        result = octo_norm(vec)
        expected = np.array([-0.5])  # 0.25 * 2 - 1 = -0.5
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_octo_norm_round_trip(self):
        """Test that forward and reverse operations are inverses"""
        original = np.array([0.2, 0.7, 0.9])
        normalized = octo_norm(original)
        restored = octo_norm(normalized, reverse=True)
        np.testing.assert_array_almost_equal(original, restored)


class TestTensorFlowConversion(unittest.TestCase):
    """Test TensorFlow dataset conversion"""
    
    def test_convert_pytype_to_tf_dataset_basic(self):
        """Test basic conversion to TensorFlow dataset"""
        # Create sample data
        X = np.random.random((50, 3)).astype(np.float32)
        y = np.random.random((50, 1)).astype(np.float32)
        
        dataset = convert_pytype_to_tf_dataset(X, y, batch_size=10)
        
        # Should return a TensorFlow dataset
        self.assertIsInstance(dataset, tf.data.Dataset)
        
        # Check that we can iterate through batches
        batch_count = 0
        for batch_x, batch_y in dataset:
            batch_count += 1
            self.assertEqual(batch_x.shape[0], 10)  # Batch size
            self.assertEqual(batch_x.shape[1], 3)   # Feature size
            self.assertEqual(batch_y.shape[0], 10)  # Batch size
            self.assertEqual(batch_y.shape[1], 1)   # Target size
            
        # Should have 5 batches (50 samples / 10 batch size)
        self.assertEqual(batch_count, 5)
    
    def test_convert_pytype_to_tf_dataset_different_batch_sizes(self):
        """Test conversion with different batch sizes"""
        X = np.random.random((30, 2)).astype(np.float32)
        y = np.random.random((30, 1)).astype(np.float32)
        
        # Test batch size of 5
        dataset = convert_pytype_to_tf_dataset(X, y, batch_size=5)
        batch_count = sum(1 for _ in dataset)
        self.assertEqual(batch_count, 6)  # 30 / 5 = 6
        
        # Test batch size of 1
        dataset = convert_pytype_to_tf_dataset(X, y, batch_size=1)
        batch_count = sum(1 for _ in dataset)
        self.assertEqual(batch_count, 30)  # 30 / 1 = 30


class TestLogErasure(unittest.TestCase):
    """Test log erasure functionality"""
    
    def setUp(self):
        """Set up test directory structure"""
        self.test_dir = tempfile.mkdtemp()
        self.logs_dir = os.path.join(self.test_dir, 'models', 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Create some test log files
        for i in range(3):
            log_file = os.path.join(self.logs_dir, f'test_log_{i}.log')
            with open(log_file, 'w') as f:
                f.write(f'Test log content {i}')
    
    def tearDown(self):
        """Clean up test directory"""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('util.ROOT_DIR')
    def test_erase_all_logs(self, mock_root_dir):
        """Test log erasure functionality"""
        mock_root_dir.__str__ = lambda: self.test_dir
        mock_root_dir.__add__ = lambda self, other: os.path.join(str(self), other)
        
        # Verify logs exist before erasure
        log_files = os.listdir(self.logs_dir)
        self.assertGreater(len(log_files), 0)
        
        # Erase logs
        with patch('builtins.input', return_value='y'):  # Mock user input
            erase_all_logs()
        
        # Verify logs were erased
        if os.path.exists(self.logs_dir):
            remaining_files = os.listdir(self.logs_dir)
            # Directory might still exist but should be empty or contain only .gitkeep
            non_keep_files = [f for f in remaining_files if f != '.gitkeep']
            self.assertEqual(len(non_keep_files), 0)


class TestDefaultLoader(unittest.TestCase):
    """Test DefaultLoader abstract base class"""
    
    def setUp(self):
        """Set up test loader"""
        class TestLoader(DefaultLoader):
            def __init__(self, file_name_or_ml_mode: Optional[Union[str, MLMode]] = None, **kwargs):
                super().__init__(file_name_or_ml_mode=file_name_or_ml_mode, kwargs=kwargs)
            def _load(self):
                return "test_data"
        
        self.loader = TestLoader()
    
    def test_confirm_file_exists_valid_file(self):
        """Test file existence confirmation with valid file"""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(b"test content")
        
        try:
            # Should not raise exception
            self.loader.path = temp_path
            self.loader._confirm_file_exists()
        finally:
            os.unlink(temp_path)
    
    def test_confirm_file_exists_invalid_file(self):
        """Test file existence confirmation with invalid file"""
        invalid_path = "/nonexistent/path/file.txt"
        self.loader.path = invalid_path
        with self.assertRaises(FileNotFoundError):
            self.loader._confirm_file_exists()

class TestDataValidation(unittest.TestCase):
    """Test data validation and edge cases"""
    
    def test_train_test_split_empty_data(self):
        """Test train_test_split with empty data"""
        X = np.array([]).reshape(0, 3)
        y = np.array([]).reshape(0, 1)
        
        with self.assertRaises((ValueError, IndexError)):
            train_test_split(X, y, test_size=0.2)
    
    def test_train_test_split_mismatched_sizes(self):
        """Test train_test_split with mismatched X and y sizes"""
        X = np.random.random((100, 3))
        y = np.random.random((90, 1))  # Different size
        
        with self.assertRaises((ValueError, IndexError)):
            train_test_split(X, y, test_size=0.2)


if __name__ == '__main__':
    unittest.main()