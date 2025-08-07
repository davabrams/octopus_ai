"""
Unit tests for inference server components
"""
import unittest
import time
import threading
import json
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from inference_server.model_inference import (
    JobStatus, InferenceJob, InferenceQueue, 
    ExecuteSuckerInference, InferenceExecutionBase
)


class TestJobStatus(unittest.TestCase):
    """Test JobStatus enum"""
    
    def test_job_status_values(self):
        """Test JobStatus enum values"""
        self.assertEqual(JobStatus.PENDING, "Pending")
        self.assertEqual(JobStatus.EXECUTING, "Executing")
        self.assertEqual(JobStatus.COMPLETE, "Complete")
        self.assertEqual(JobStatus.FAILED, "Failed")
    
    def test_job_status_string_conversion(self):
        """Test JobStatus string conversion"""
        self.assertEqual(str(JobStatus.PENDING), "Pending")
        self.assertEqual(str(JobStatus.COMPLETE), "Complete")


class TestInferenceJob(unittest.TestCase):
    """Test InferenceJob class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.job_details = {
            "job_id": "123",
            "data": {
                "c.r": 0.5,
                "c_val.r": 0.7
            }
        }
        self.job = InferenceJob(self.job_details)
    
    def test_inference_job_initialization(self):
        """Test InferenceJob initialization"""
        self.assertEqual(self.job.job_id, 123)  # Should be converted to int
        self.assertEqual(self.job.data, self.job_details["data"])
        self.assertEqual(self.job.status, JobStatus.PENDING)
        self.assertIsNotNone(self.job.queue_timestamp)
        self.assertIsNone(self.job.result)
        self.assertIsNone(self.job.inference_start_timestamp)
        self.assertIsNone(self.job.inference_end_timestamp)
    
    def test_inference_job_string_representation(self):
        """Test InferenceJob string representation"""
        expected = "123: Pending"
        self.assertEqual(str(self.job), expected)
        
        # Test with different status
        self.job.status = JobStatus.COMPLETE
        expected = "123: Complete"
        self.assertEqual(str(self.job), expected)
    
    def test_inference_job_as_json(self):
        """Test InferenceJob JSON serialization"""
        self.job.result = 0.85
        self.job.status = JobStatus.COMPLETE
        
        json_data = self.job.as_json()
        
        self.assertIsInstance(json_data, dict)
        self.assertEqual(json_data["job_id"], "123")
        self.assertEqual(json_data["status"], "Complete")
        self.assertEqual(json_data["result"], "0.85")
    
    def test_inference_job_timestamps(self):
        """Test InferenceJob timestamp handling"""
        # Initially, timestamps should be None except queue_timestamp
        self.assertIsNotNone(self.job.queue_timestamp)
        self.assertIsNone(self.job.inference_start_timestamp)
        self.assertIsNone(self.job.inference_end_timestamp)
        
        # Test human readable timestamps
        readable_times = self.job.get_human_readable_timestamps()
        self.assertIsInstance(readable_times, dict)
        self.assertIn("queue_timestamp", readable_times)
        self.assertIsNotNone(readable_times["queue_timestamp"])
    
    @patch('inference_server.model_inference.ExecuteSuckerInference')
    def test_inference_job_execute(self, mock_executor):
        """Test InferenceJob execution"""
        mock_queue = Mock()
        
        # Setup mock executor to set job status
        def mock_execution(parent, data):
            parent.status = JobStatus.COMPLETE
            parent.result = 0.75
        
        mock_executor.side_effect = mock_execution
        
        # Execute the job
        self.job.execute(mock_queue)
        
        # Verify execution behavior
        self.assertEqual(self.job.status, JobStatus.COMPLETE)
        self.assertEqual(self.job.result, 0.75)
        self.assertIsNotNone(self.job.inference_start_timestamp)
        self.assertIsNotNone(self.job.inference_end_timestamp)
        
        # Verify queue method was called
        mock_queue.move_to_complete.assert_called_once_with(self.job.job_id)


class TestExecuteSuckerInference(unittest.TestCase):
    """Test ExecuteSuckerInference class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_parent = Mock()
        self.mock_parent.status = JobStatus.EXECUTING
        
        self.test_data = {
            "c.r": 0.6,
            "c_val.r": 0.8
        }
    
    @patch('inference_server.model_inference.sucker_model')
    def test_execute_sucker_inference_success(self, mock_model):
        """Test successful sucker inference execution"""
        # Mock model prediction
        mock_model.predict.return_value = np.array([[0.95]])
        
        # Execute inference
        executor = ExecuteSuckerInference(parent=self.mock_parent, data=self.test_data)
        
        # Verify model was called with correct input
        mock_model.predict.assert_called_once()
        call_args = mock_model.predict.call_args[0][0]
        expected_input = np.array([[0.6, 0.8]])
        np.testing.assert_array_almost_equal(call_args, expected_input)
        
        # Verify result was set
        self.assertEqual(self.mock_parent.result, 0.95)
        self.assertEqual(self.mock_parent.status, JobStatus.COMPLETE)
    
    @patch('inference_server.model_inference.sucker_model')
    def test_execute_sucker_inference_failure(self, mock_model):
        """Test sucker inference execution with model failure"""
        # Mock model to raise exception
        mock_model.predict.side_effect = Exception("Model prediction failed")
        
        # Execute inference
        executor = ExecuteSuckerInference(parent=self.mock_parent, data=self.test_data)
        
        # Verify failure was handled
        self.assertEqual(self.mock_parent.status, JobStatus.FAILED)


class TestInferenceQueue(unittest.TestCase):
    """Test InferenceQueue class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Patch the watchdog to avoid threading issues in tests
        with patch.object(InferenceQueue, '__init__', lambda x: None):
            self.queue = InferenceQueue()
            self.queue.thread_count = 2
            self.queue.seconds_until_stale = 30.0
            self.queue._q = {}
            self.queue._pending_queue = []
            self.queue._execution_queue = []
            self.queue._completion_queue = []
            self.queue._kill_watchdog = Mock()
            self.queue._watchdog_thread = None
    
    def test_queue_initialization_components(self):
        """Test queue component initialization"""
        self.assertIsInstance(self.queue._q, dict)
        self.assertIsInstance(self.queue._pending_queue, list)
        self.assertIsInstance(self.queue._execution_queue, list)
        self.assertIsInstance(self.queue._completion_queue, list)
    
    def test_add_job_to_queue(self):
        """Test adding job to queue"""
        job_details = {"job_id": "456", "data": {"c.r": 0.3, "c_val.r": 0.4}}
        job = InferenceJob(job_details)
        
        self.queue.add(job)
        
        # Verify job was added
        self.assertIn(456, self.queue._q)
        self.assertEqual(len(self.queue._pending_queue), 1)
        self.assertEqual(self.queue._pending_queue[0][1], 456)  # job_id in tuple
    
    def test_get_job_from_queue(self):
        """Test retrieving job from queue"""
        job_details = {"job_id": "789", "data": {"c.r": 0.2, "c_val.r": 0.3}}
        job = InferenceJob(job_details)
        self.queue.add(job)
        
        retrieved_job = self.queue.get(789)
        
        self.assertEqual(retrieved_job, job)
        self.assertEqual(retrieved_job.job_id, 789)
    
    def test_get_nonexistent_job(self):
        """Test retrieving non-existent job raises IndexError"""
        with self.assertRaises(IndexError):
            self.queue.get(999)
    
    def test_delete_job_from_queue(self):
        """Test deleting job from queue"""
        job_details = {"job_id": "111", "data": {"c.r": 0.1, "c_val.r": 0.2}}
        job = InferenceJob(job_details)
        self.queue.add(job)
        
        # Verify job exists
        self.assertIn(111, self.queue._q)
        
        # Delete job
        self.queue.delete(111)
        
        # Verify job was removed
        self.assertNotIn(111, self.queue._q)
        self.assertEqual(len(self.queue._pending_queue), 0)
    
    def test_delete_nonexistent_job(self):
        """Test deleting non-existent job raises IndexError"""
        with self.assertRaises(IndexError):
            self.queue.delete(888)
    
    def test_all_job_ids(self):
        """Test getting all job IDs"""
        job1 = InferenceJob({"job_id": "100", "data": {}})
        job2 = InferenceJob({"job_id": "200", "data": {}})
        
        self.queue.add(job1)
        self.queue.add(job2)
        
        job_ids = self.queue.all_job_ids()
        self.assertIn(100, job_ids)
        self.assertIn(200, job_ids)
        self.assertEqual(len(job_ids), 2)
    
    def test_all_queues_status(self):
        """Test getting all queue status"""
        queues = self.queue.all_queues()
        
        self.assertIsInstance(queues, dict)
        self.assertIn("pending_queue", queues)
        self.assertIn("execution_queue", queues)
        self.assertIn("completion_queue", queues)
    
    def test_move_to_complete(self):
        """Test moving job from execution to completion queue"""
        # Add job to execution queue manually
        job_id = 123
        timestamp = time.time()
        self.queue._execution_queue.append((timestamp, job_id))
        
        # Move to complete
        self.queue.move_to_complete(job_id)
        
        # Verify job moved
        self.assertEqual(len(self.queue._execution_queue), 0)
        self.assertEqual(len(self.queue._completion_queue), 1)
        self.assertEqual(self.queue._completion_queue[0][1], job_id)
    
    def test_destroy_job(self):
        """Test destroying job from execution queue"""
        # Add job to execution queue
        job_id = 456
        timestamp = time.time()
        self.queue._execution_queue.append((timestamp, job_id))
        
        # Destroy job
        self.queue.destroy_job(job_id)
        
        # Verify job was removed but not moved to completion
        self.assertEqual(len(self.queue._execution_queue), 0)
        self.assertEqual(len(self.queue._completion_queue), 0)
    
    def test_collect_and_clear(self):
        """Test collecting and clearing completed jobs"""
        # Add jobs to completion queue and main queue
        job1 = InferenceJob({"job_id": "301", "data": {}})
        job2 = InferenceJob({"job_id": "302", "data": {}})
        job1.result = 0.85
        job2.result = 0.92
        job1.status = JobStatus.COMPLETE
        job2.status = JobStatus.COMPLETE
        
        self.queue._q[301] = job1
        self.queue._q[302] = job2
        self.queue._completion_queue.append((time.time(), 301))
        self.queue._completion_queue.append((time.time(), 302))
        
        # Collect and clear
        results = self.queue.collect_and_clear()
        
        # Verify results
        self.assertEqual(len(results), 2)
        self.assertEqual(len(self.queue._completion_queue), 0)
        self.assertNotIn(301, self.queue._q)
        self.assertNotIn(302, self.queue._q)
        
        # Verify result format
        for result in results:
            self.assertIn("job_id", result)
            self.assertIn("status", result)
            self.assertIn("result", result)
    
    def test_reset_all_queues(self):
        """Test resetting all queues"""
        # Add some data to queues
        job = InferenceJob({"job_id": "999", "data": {}})
        self.queue.add(job)
        
        # Reset
        self.queue.reset_all_queues()
        
        # Verify all queues are empty
        self.assertEqual(len(self.queue._q), 0)
        self.assertEqual(len(self.queue._pending_queue), 0)
        self.assertEqual(len(self.queue._execution_queue), 0)
        self.assertEqual(len(self.queue._completion_queue), 0)
    
    def test_queue_iteration(self):
        """Test queue iteration functionality"""
        # Add jobs to different queues
        job1 = InferenceJob({"job_id": "501", "data": {}})
        job2 = InferenceJob({"job_id": "502", "data": {}})
        
        self.queue._q[501] = job1
        self.queue._q[502] = job2
        self.queue._pending_queue.append((time.time(), 501))
        self.queue._completion_queue.append((time.time(), 502))
        
        # Test iteration
        jobs = list(self.queue)
        
        self.assertEqual(len(jobs), 2)
        job_ids = [job.job_id for job in jobs]
        self.assertIn(501, job_ids)
        self.assertIn(502, job_ids)
    
    def test_clear_stale_jobs(self):
        """Test clearing stale pending jobs"""
        # Add old job to pending queue
        old_timestamp = time.time() - 100  # 100 seconds ago
        recent_timestamp = time.time() - 10  # 10 seconds ago
        
        old_job = InferenceJob({"job_id": "000", "data": {}})
        recent_job = InferenceJob({"job_id": "100", "data": {}})
        
        self.queue._q["old"] = old_job
        self.queue._q["recent"] = recent_job
        self.queue._pending_queue.append((old_timestamp, "old"))
        self.queue._pending_queue.append((recent_timestamp, "recent"))
        
        # Set short stale time for testing
        self.queue.seconds_until_stale = 50.0
        
        # Clear stale jobs
        self.queue.clear_stale()
        
        # Verify old job was removed, recent job remains
        self.assertNotIn("old", self.queue._q)
        self.assertIn("recent", self.queue._q)
        self.assertEqual(len(self.queue._pending_queue), 1)


class TestInferenceServerIntegration(unittest.TestCase):
    """Integration tests for inference server components"""
    
    def test_job_lifecycle(self):
        """Test complete job lifecycle"""
        # Create queue without watchdog for testing
        with patch.object(InferenceQueue, '__init__', lambda x: None):
            queue = InferenceQueue()
            queue.thread_count = 1
            queue._q = {}
            queue._pending_queue = []
            queue._execution_queue = []
            queue._completion_queue = []
            queue._kill_watchdog = Mock()
            queue._kill_watchdog.is_set.return_value = False
        
        # Create and add job
        job_details = {"job_id": "987", "data": {"c.r": 0.5, "c_val.r": 0.6}}
        job = InferenceJob(job_details)
        queue.add(job)
        
        # Verify job is pending
        self.assertEqual(job.status, JobStatus.PENDING)
        self.assertEqual(len(queue._pending_queue), 1)
        
        # Move to execution (simulate execute_new_job without threading)
        with patch.object(job, 'execute') as mock_execute:
            mock_execute.side_effect = lambda parent_queue: setattr(job, 'status', JobStatus.COMPLETE)
            
            if queue._pending_queue:
                job_ts, job_id = queue._pending_queue[-1]
                queue._execution_queue.append((job_ts, job_id))
                job.status = JobStatus.EXECUTING
                del queue._pending_queue[-1]
        
        # Verify job is executing
        self.assertEqual(job.status, JobStatus.EXECUTING)
        self.assertEqual(len(queue._execution_queue), 1)
        
        # Complete job
        queue.move_to_complete(job.job_id)
        
        # Verify job is completed
        self.assertEqual(len(queue._execution_queue), 0)
        self.assertEqual(len(queue._completion_queue), 1)


if __name__ == '__main__':
    unittest.main()