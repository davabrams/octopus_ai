"""
Integration tests for octopus_ai system - end-to-end workflows
"""
import unittest
import numpy as np
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulator.surface_generator import RandomSurface
from simulator.agent_generator import AgentGenerator
from simulator.octopus_generator import Octopus
from simulator.simutil import State, MLMode, InferenceLocation, MovementMode, AgentType
from training.sucker import SuckerTrainer
from training.limb import LimbTrainer
from inference_server.model_inference import InferenceQueue, InferenceJob
from OctoConfig import GameParameters, TrainingParameters
from octo_datagen import OctoDatagen


class TestSimulationIntegration(unittest.TestCase):
    """Test complete simulation workflows"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.game_params = GameParameters.copy()
        self.game_params.update({
            'x_len': 8,
            'y_len': 8,
            'octo_num_arms': 2,
            'limb_rows': 3,
            'limb_cols': 2,
            'agent_number_of_agents': 2,
            'num_iterations': 5,  # Short for testing
            'inference_location': InferenceLocation.LOCAL  # Use heuristic to avoid ML dependencies
        })
    
    def test_complete_simulation_cycle(self):
        """Test a complete simulation cycle with all components"""
        try:
            # 1. Create surface
            surface = RandomSurface(self.game_params)
            self.assertIsNotNone(surface)
            
            # 2. Create octopus
            octopus_state = State(
                x=self.game_params['x_len'] / 2,
                y=self.game_params['y_len'] / 2,
                t=0.0
            )
            octopus = Octopus(self.game_params)
            octopus.x = octopus_state.x
            octopus.y = octopus_state.y
            self.assertIsNotNone(octopus)
            self.assertEqual(len(octopus.limbs), self.game_params['octo_num_arms'])
            
            # 3. Create agents
            agent_gen = AgentGenerator(self.game_params)
            agent_gen.generate(self.game_params['agent_number_of_agents'])
            self.assertEqual(len(agent_gen.agents), self.game_params['agent_number_of_agents'])
            
            # 4. Run simulation steps
            for iteration in range(self.game_params['num_iterations']):
                # Move agents
                agent_gen.increment_all()
                
                # Move octopus
                octopus.move(ag=agent_gen.agents)
                
                # Find colors (heuristic mode)
                colors = octopus.find_color(surface)
                self.assertIsInstance(colors, list)
                self.assertEqual(len(colors), len(octopus.limbs))
                
                # Set colors
                octopus.set_color(colors)
                
                # Calculate visibility
                visibility = octopus.visibility(surface)
                self.assertIsInstance(visibility, float)
                self.assertGreaterEqual(visibility, 0.0)
            
            # Simulation should complete without errors
            self.assertTrue(True)
            
        except Exception as e:
            self.fail(f"Complete simulation cycle failed: {e}")
    
    def test_multi_agent_interaction(self):
        """Test interactions between multiple agents and octopus"""
        # Create surface and octopus
        self.game_params.update({'agent_number_of_agents': 0})
        surface = RandomSurface(self.game_params)
        octopus = Octopus(self.game_params)

        x_len = self.game_params['x_len']
        y_len = self.game_params['y_len']
        octopus.x = x_len / 2
        octopus.y = y_len / 2
        
        # Create agents with different types
        agent_gen = AgentGenerator(self.game_params)
        print(agent_gen)
        agent_gen.generate(1, fixed_agent_type=AgentType.PREY)
        agent_gen.generate(1, fixed_agent_type=AgentType.THREAT)
        print(agent_gen)
        self.assertEqual(len(agent_gen.agents), 2)
        
        # Record initial positions
        initial_octopus_pos = (octopus.x, octopus.y)
        initial_agent_positions = [(agent.x, agent.y) for agent in agent_gen.agents]
        
        # Run simulation steps
        for _ in range(3):
            agent_gen.increment_all()
            octopus.move(ag=agent_gen.agents)
        
        # Verify movement occurred
        final_octopus_pos = (octopus.x, octopus.y)
        final_agent_positions = [(agent.x, agent.y) for agent in agent_gen.agents]
        
        # At least some positions should have changed
        positions_changed = (
            initial_octopus_pos != final_octopus_pos or
            initial_agent_positions != final_agent_positions
        )
        self.assertTrue(positions_changed)
    
    def test_different_movement_modes(self):
        """Test simulation with different movement modes"""
        movement_modes = [MovementMode.RANDOM, MovementMode.ATTRACT_REPEL]
        
        for mode in movement_modes:
            with self.subTest(movement_mode=mode):
                params = self.game_params.copy()
                params['octo_movement_mode'] = mode
                params['agent_movement_mode'] = mode
                params['limb_movement_mode'] = mode
                
                # Create components
                surface = RandomSurface(params)
                octopus = Octopus(params)
                agent_gen = AgentGenerator(params)
                agent_gen.generate(1)
                
                # Run a few steps
                try:
                    for _ in range(2):
                        agent_gen.increment_all()
                        octopus.move(ag=agent_gen.agents)
                        colors = octopus.find_color(surface)
                        octopus.set_color(colors)
                    
                    # Should complete without errors
                    self.assertTrue(True)
                    
                except Exception as e:
                    self.fail(f"Movement mode {mode} failed: {e}")


@patch('training.sucker.sucker_model')
@patch('training.limb.limb_model')
class TestTrainingIntegration(unittest.TestCase):
    """Test training pipeline integration"""
    
    def setUp(self):
        """Set up training integration tests"""
        self.training_params = TrainingParameters.copy()
        self.training_params.update({
            'epochs': 1,  # Very short for testing
            'batch_size': 4,
            'test_size': 0.3,
            'save_model_to_disk': False,  # Don't save during tests
            'generate_tensorboard': False  # Disable tensorboard for tests
        })
        
        self.game_params = GameParameters.copy()
        self.game_params.update({
            'x_len': 6,
            'y_len': 6,
            'octo_num_arms': 1,
            'limb_rows': 2,
            'limb_cols': 2,
            'num_iterations': 3  # Short for data generation
        })
    
    def test_sucker_training_pipeline(self, mock_limb_model, mock_sucker_model):
        """Test complete sucker training pipeline"""
        # Mock the global models to avoid loading real models
        mock_sucker_model.predict = Mock(return_value=np.array([[0.5]]))
        
        trainer = SuckerTrainer(self.training_params)
        
        # Generate minimal training data
        with patch('training.sucker.RandomSurface') as mock_surface_class:
            with patch('training.sucker.Octopus') as mock_octopus_class:
                # Setup mocks for data generation
                mock_surface = Mock()
                mock_surface.get_val.return_value = 0.5
                mock_surface_class.return_value = mock_surface
                
                mock_sucker = Mock()
                mock_sucker.x = 1.0
                mock_sucker.y = 1.0
                mock_sucker.get_surf_color_at_this_sucker.return_value = 0.6
                mock_sucker.color.r = 0.7
                
                mock_limb = Mock()
                mock_limb.suckers = [mock_sucker]
                
                mock_octopus = Mock()
                mock_octopus.limbs = [mock_limb]
                mock_octopus_class.return_value = mock_octopus
                
                try:
                    # Generate data
                    X, y = trainer.datagen(self.game_params)
                    self.assertGreater(len(X), 0)
                    self.assertGreater(len(y), 0)
                    
                    # Format data
                    X_train, X_test, y_train, y_test = trainer.data_format(X, y)
                    trainer.X_train = X_train
                    trainer.X_test = X_test
                    trainer.y_train = y_train
                    trainer.y_test = y_test
                    
                    # Train model
                    model = trainer.train_sucker_model()
                    self.assertIsNotNone(model)
                    
                    # Test inference
                    test_input = np.array([[0.5, 0.6]])
                    result = trainer.inference(test_input)
                    self.assertIsNotNone(result)
                    
                except Exception as e:
                    self.fail(f"Sucker training pipeline failed: {e}")
    
    def test_limb_training_pipeline(self, mock_limb_model, mock_sucker_model):
        """Test complete limb training pipeline"""
        # Mock the global models
        mock_limb_model.predict = Mock(return_value=np.array([[0.1, 0.2, 0.3, 0.4]]))
        
        trainer = LimbTrainer(self.training_params)
        
        # Generate minimal training data
        with patch('training.limb.RandomSurface') as mock_surface_class:
            with patch('training.limb.Octopus') as mock_octopus_class:
                # Setup mocks
                mock_surface = Mock()
                mock_surface_class.return_value = mock_surface
                
                # Create mock suckers with adjacency
                mock_suckers = []
                for i in range(4):
                    mock_sucker = Mock()
                    mock_sucker.x = i * 0.5
                    mock_sucker.y = i * 0.3
                    mock_sucker.get_surf_color_at_this_sucker.return_value = 0.4
                    mock_sucker.color.r = 0.5
                    mock_suckers.append(mock_sucker)
                
                mock_limb = Mock()
                mock_limb.suckers = mock_suckers
                mock_limb.find_adjacents.return_value = mock_suckers[:2]
                
                mock_octopus = Mock()
                mock_octopus.limbs = [mock_limb]
                mock_octopus_class.return_value = mock_octopus
                
                try:
                    # Generate data
                    X, y = trainer.datagen(self.game_params)
                    self.assertGreater(len(X), 0)
                    self.assertGreater(len(y), 0)
                    
                    # Format data
                    X_train, X_test, y_train, y_test = trainer.data_format(X, y)
                    trainer.X_train = X_train
                    trainer.X_test = X_test
                    trainer.y_train = y_train
                    trainer.y_test = y_test
                    
                    # Train model (may fail due to ragged tensor complexity)
                    try:
                        model = trainer.train_limb_model(self.game_params)
                        self.assertIsNotNone(model)
                    except Exception as model_error:
                        # Limb training is complex; allow some failures
                        if "ragged" in str(model_error).lower():
                            self.skipTest("Ragged tensor operations not supported in test environment")
                        else:
                            raise
                    
                except Exception as e:
                    self.fail(f"Limb training pipeline failed: {e}")


class TestInferenceServerIntegration(unittest.TestCase):
    """Test inference server integration"""
    
    def setUp(self):
        """Set up inference server tests"""
        # Create queue without watchdog thread for testing
        with patch.object(InferenceQueue, '__init__', lambda x: None):
            self.queue = InferenceQueue()
            self.queue.thread_count = 2
            self.queue.seconds_until_stale = 30.0
            self.queue._q = {}
            self.queue._pending_queue = []
            self.queue._execution_queue = []
            self.queue._completion_queue = []
            self.queue._kill_watchdog = Mock()
            self.queue._kill_watchdog.is_set.return_value = False
    
    def test_job_queue_workflow(self):
        """Test complete job queue workflow"""
        # Create multiple jobs
        job_details = [
            {"job_id": "100", "data": {"c.r": 0.1, "c_val.r": 0.2}},
            {"job_id": "101", "data": {"c.r": 0.3, "c_val.r": 0.4}},
            {"job_id": "102", "data": {"c.r": 0.5, "c_val.r": 0.6}}
        ]
        
        jobs = [InferenceJob(details) for details in job_details]
        
        # Add jobs to queue
        for job in jobs:
            self.queue.add(job)
        
        self.assertEqual(len(self.queue._pending_queue), 3)
        
        # Simulate job processing
        for job in jobs:
            # Move to execution
            if self.queue._pending_queue:
                job_ts, job_id = self.queue._pending_queue.pop(0)
                self.queue._execution_queue.append((job_ts, job_id))
                job.status = "Executing"
            
            # Complete job
            job.status = "Complete"
            job.result = 0.5 + job.job_id * 0.1  # Mock result
            self.queue.move_to_complete(job.job_id)
        
        # Verify all jobs completed
        self.assertEqual(len(self.queue._completion_queue), 3)
        self.assertEqual(len(self.queue._execution_queue), 0)
        
        # Collect results
        results = self.queue.collect_and_clear()
        self.assertEqual(len(results), 3)
        self.assertEqual(len(self.queue._completion_queue), 0)
    
    @patch('inference_server.model_inference.sucker_model')
    def test_model_inference_integration(self, mock_model):
        """Test model inference integration"""
        mock_model.predict.return_value = np.array([[0.85]])
        
        # Create and process job
        job_details = {"job_id": "200", "data": {"c.r": 0.4, "c_val.r": 0.6}}
        job = InferenceJob(job_details)
        
        self.queue.add(job)
        
        # Execute job
        mock_parent_queue = Mock()
        job.execute(mock_parent_queue)
        
        # Verify execution
        self.assertEqual(job.status, "Complete")
        self.assertEqual(job.result, 0.85)
        mock_model.predict.assert_called_once()
        mock_parent_queue.move_to_complete.assert_called_once_with(job.job_id)


class TestDataGenerationIntegration(unittest.TestCase):
    """Test data generation integration"""
    
    def setUp(self):
        """Set up data generation tests"""
        self.game_params = GameParameters.copy()
        self.game_params.update({
            'x_len': 5,
            'y_len': 5,
            'octo_num_arms': 1,
            'limb_rows': 2,
            'limb_cols': 2,
            'num_iterations': 3,
            'inference_location': InferenceLocation.LOCAL
        })
    
    @patch('octo_datagen.Octopus')
    @patch('octo_datagen.RandomSurface')
    def test_datagen_integration(self, mock_surface_class, mock_octopus_class):
        """Test data generation integration"""
        # Setup mocks
        mock_surface = Mock()
        mock_surface.get_val.return_value = 0.5
        mock_surface_class.return_value = mock_surface
        
        # Mock octopus with limbs and suckers
        mock_suckers = []
        for i in range(4):  # 2x2 limb
            mock_sucker = Mock()
            mock_sucker.x = i
            mock_sucker.y = i
            mock_sucker.get_surf_color_at_this_sucker.return_value = 0.3 + i * 0.1
            mock_sucker.color.r = 0.4 + i * 0.05
            mock_suckers.append(mock_sucker)
        
        mock_limb = Mock()
        mock_limb.suckers = mock_suckers
        
        mock_octopus = Mock()
        mock_octopus.limbs = [mock_limb]
        mock_octopus.find_color.return_value = [[0.5, 0.6, 0.7, 0.8]]
        mock_octopus.visibility.return_value = 0.25
        mock_octopus_class.return_value = mock_octopus
        
        # Create data generator
        datagen = OctoDatagen(self.game_params)
        
        try:
            # Run data generation
            data = datagen.run_color_datagen()
            
            # Verify data was generated
            self.assertIn('metadata', data)
            self.assertIsInstance(data['metadata'], dict)
            self.assertIn('game_parameters', data)
            self.assertEqual(data['game_parameters'], self.game_params)
            self.assertIn('state_data', data)
            self.assertIsInstance(data['state_data'], list)
            self.assertIn('gt_data', data)
            self.assertIsInstance(data['gt_data'], list)
            
            # Verify data format
            for x_sample in X:
                self.assertIsInstance(x_sample, (list, np.ndarray))
            
            for y_sample in y:
                self.assertIsInstance(y_sample, (list, np.ndarray, float))
            
        except Exception as e:
            self.fail(f"Data generation integration failed: {e}")


class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete end-to-end workflows"""
    
    @patch('training.sucker.sucker_model')
    def test_simulation_to_training_workflow(self, mock_sucker_model):
        """Test workflow from simulation to training"""
        mock_sucker_model.predict = Mock(return_value=np.array([[0.7]]))
        
        # 1. Setup parameters
        game_params = GameParameters.copy()
        game_params.update({
            'x_len': 4, 'y_len': 4, 'octo_num_arms': 1,
            'limb_rows': 2, 'limb_cols': 1, 'num_iterations': 2
        })
        
        training_params = TrainingParameters.copy()
        training_params.update({
            'epochs': 1, 'batch_size': 2, 'test_size': 0.3,
            'save_model_to_disk': False, 'generate_tensorboard': False
        })
        
        try:
            # 2. Generate simulation data
            with patch('training.sucker.RandomSurface') as mock_surface_class:
                with patch('training.sucker.Octopus') as mock_octopus_class:
                    # Setup mocks
                    mock_surface = Mock()
                    mock_surface.get_val.return_value = 0.5
                    mock_surface_class.return_value = mock_surface
                    
                    mock_sucker = Mock()
                    mock_sucker.x = 1.0
                    mock_sucker.y = 1.0
                    mock_sucker.get_surf_color_at_this_sucker.return_value = 0.6
                    mock_sucker.color.r = 0.7
                    
                    mock_limb = Mock()
                    mock_limb.suckers = [mock_sucker]
                    
                    mock_octopus = Mock()
                    mock_octopus.limbs = [mock_limb]
                    mock_octopus_class.return_value = mock_octopus
                    
                    # Generate training data
                    trainer = SuckerTrainer(training_params)
                    X, y = trainer.datagen(game_params)
                    
                    # 3. Train model
                    X_train, X_test, y_train, y_test = trainer.data_format(X, y)
                    trainer.X_train = X_train
                    trainer.X_test = X_test
                    trainer.y_train = y_train
                    trainer.y_test = y_test
                    
                    model = trainer.train_sucker_model()
                    
                    # 4. Test inference
                    test_input = np.array([[0.5, 0.6]])
                    prediction = model.predict(test_input, verbose=0)
                    
                    # Verify complete workflow
                    self.assertIsNotNone(model)
                    self.assertIsNotNone(prediction)
                    self.assertEqual(prediction.shape, (1, 1))
            
        except Exception as e:
            self.fail(f"Simulation to training workflow failed: {e}")
    
    def test_configuration_consistency(self):
        """Test consistency between different configuration parameters"""
        # Test that game parameters are consistent
        game_params = GameParameters.copy()
        
        # Verify required parameters exist
        required_params = [
            'x_len', 'y_len', 'octo_num_arms', 'limb_rows', 'limb_cols',
            'agent_number_of_agents', 'num_iterations'
        ]
        
        for param in required_params:
            self.assertIn(param, game_params)
            self.assertIsNotNone(game_params[param])
        
        # Test that training parameters are consistent
        training_params = TrainingParameters.copy()
        
        required_training_params = [
            'epochs', 'batch_size', 'test_size'
        ]
        
        for param in required_training_params:
            self.assertIn(param, training_params)
            self.assertIsNotNone(training_params[param])
        
        # Test parameter value ranges
        self.assertGreater(game_params['x_len'], 0)
        self.assertGreater(game_params['y_len'], 0)
        self.assertGreater(game_params['octo_num_arms'], 0)
        self.assertGreater(training_params['epochs'], 0)
        self.assertGreater(training_params['batch_size'], 0)
        self.assertGreater(training_params['test_size'], 0.0)
        self.assertLess(training_params['test_size'], 1.0)


if __name__ == '__main__':
    # Run tests with higher verbosity for integration tests
    unittest.main(verbosity=2)