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


class TestSimulationIntegration(unittest.TestCase):
    """Test complete simulation workflows"""

    def setUp(self):
        """Set up integration test environment"""
        # Clear class-level shared agents list
        AgentGenerator.agents = []
        self.game_params = GameParameters.copy()
        self.game_params.update({
            'x_len': 8,
            'y_len': 8,
            'octo_num_arms': 2,
            'limb_rows': 3,
            'limb_cols': 2,
            'agent_number_of_agents': 2,
            'num_iterations': 5,
            'inference_location': InferenceLocation.LOCAL,
        })

    def test_complete_simulation_cycle(self):
        """Test a complete simulation cycle with all components"""
        surface = RandomSurface(self.game_params)
        self.assertIsNotNone(surface)

        octopus = Octopus(self.game_params)
        self.assertIsNotNone(octopus)
        self.assertEqual(len(octopus.limbs), self.game_params['octo_num_arms'])

        agent_gen = AgentGenerator(self.game_params)
        agent_gen.generate(self.game_params['agent_number_of_agents'])
        self.assertEqual(len(agent_gen.agents), self.game_params['agent_number_of_agents'])

        for iteration in range(self.game_params['num_iterations']):
            agent_gen.increment_all(octopus)
            octopus.move()

            # set_color runs find_color + applies colors internally
            octopus.set_color(surface)

            visibility = octopus.visibility(surface)
            self.assertIsInstance(visibility, float)
            self.assertGreaterEqual(visibility, 0.0)

    def test_multi_agent_interaction(self):
        """Test interactions between multiple agents and octopus"""
        self.game_params.update({'agent_number_of_agents': 0})
        surface = RandomSurface(self.game_params)
        octopus = Octopus(self.game_params)

        agent_gen = AgentGenerator(self.game_params)
        agent_gen.generate(1, fixed_agent_type=AgentType.PREY)
        agent_gen.generate(1, fixed_agent_type=AgentType.THREAT)
        self.assertEqual(len(agent_gen.agents), 2)

        initial_octopus_pos = (octopus.x, octopus.y)
        initial_agent_positions = [(agent.x, agent.y) for agent in agent_gen.agents]

        for _ in range(3):
            agent_gen.increment_all(octopus)
            octopus.move()

        final_octopus_pos = (octopus.x, octopus.y)
        final_agent_positions = [(agent.x, agent.y) for agent in agent_gen.agents]

        positions_changed = (
            initial_octopus_pos != final_octopus_pos or
            initial_agent_positions != final_agent_positions
        )
        self.assertTrue(positions_changed)

    def test_different_movement_modes(self):
        """Test simulation with different movement modes"""
        # ATTRACT_REPEL is not yet implemented (raises NotImplementedError
        # in Limb._move_attract_repel), so only test RANDOM for now
        movement_modes = [MovementMode.RANDOM]

        for mode in movement_modes:
            with self.subTest(movement_mode=mode):
                AgentGenerator.agents = []  # Clear shared list
                params = self.game_params.copy()
                params['octo_movement_mode'] = mode
                params['agent_movement_mode'] = mode
                params['limb_movement_mode'] = mode

                surface = RandomSurface(params)
                octopus = Octopus(params)
                agent_gen = AgentGenerator(params)
                agent_gen.generate(1)

                for _ in range(2):
                    agent_gen.increment_all(octopus)
                    octopus.move()
                    octopus.set_color(surface)

    def test_configuration_consistency(self):
        """Test consistency between different configuration parameters"""
        game_params = GameParameters.copy()

        required_params = [
            'x_len', 'y_len', 'octo_num_arms', 'limb_rows', 'limb_cols',
            'agent_number_of_agents', 'num_iterations'
        ]

        for param in required_params:
            self.assertIn(param, game_params)
            self.assertIsNotNone(game_params[param])

        training_params = TrainingParameters.copy()

        required_training_params = [
            'epochs', 'batch_size', 'test_size'
        ]

        for param in required_training_params:
            self.assertIn(param, training_params)
            self.assertIsNotNone(training_params[param])

        self.assertGreater(game_params['x_len'], 0)
        self.assertGreater(game_params['y_len'], 0)
        self.assertGreater(game_params['octo_num_arms'], 0)
        self.assertGreater(training_params['epochs'], 0)
        self.assertGreater(training_params['batch_size'], 0)
        self.assertGreater(training_params['test_size'], 0.0)
        self.assertLess(training_params['test_size'], 1.0)


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
            'inference_location': InferenceLocation.LOCAL,
        })

    def test_datagen_integration(self):
        """Test data generation integration with real components"""
        from octo_datagen import OctoDatagen

        datagen = OctoDatagen(self.game_params)
        data = datagen.run_color_datagen()

        self.assertIn('metadata', data)
        self.assertIsInstance(data['metadata'], dict)
        self.assertIn('game_parameters', data)
        self.assertEqual(data['game_parameters'], self.game_params)
        self.assertIn('state_data', data)
        self.assertIsInstance(data['state_data'], list)
        self.assertIn('gt_data', data)
        self.assertIsInstance(data['gt_data'], list)
        self.assertGreater(len(data['state_data']), 0)
        self.assertGreater(len(data['gt_data']), 0)


class TestInferenceServerIntegration(unittest.TestCase):
    """Test inference server integration"""

    def setUp(self):
        """Set up inference server tests"""
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
        job_details = [
            {"job_id": "100", "data": {"c.r": 0.1, "c_val.r": 0.2}},
            {"job_id": "101", "data": {"c.r": 0.3, "c_val.r": 0.4}},
            {"job_id": "102", "data": {"c.r": 0.5, "c_val.r": 0.6}}
        ]

        jobs = [InferenceJob(details) for details in job_details]

        for job in jobs:
            self.queue.add(job)

        self.assertEqual(len(self.queue._pending_queue), 3)

        for job in jobs:
            if self.queue._pending_queue:
                job_ts, job_id = self.queue._pending_queue.pop(0)
                self.queue._execution_queue.append((job_ts, job_id))
                job.status = "Executing"

            job.status = "Complete"
            job.result = 0.5
            self.queue.move_to_complete(job.job_id)

        self.assertEqual(len(self.queue._completion_queue), 3)
        self.assertEqual(len(self.queue._execution_queue), 0)

        results = self.queue.collect_and_clear()
        self.assertEqual(len(results), 3)
        self.assertEqual(len(self.queue._completion_queue), 0)

    @patch('inference_server.model_inference.sucker_model')
    def test_model_inference_integration(self, mock_model):
        """Test model inference integration"""
        mock_model.predict.return_value = np.array([[0.85]])

        job_details = {"job_id": "200", "data": {"c.r": 0.4, "c_val.r": 0.6}}
        job = InferenceJob(job_details)

        self.queue.add(job)

        mock_parent_queue = Mock()
        job.execute(mock_parent_queue)

        self.assertEqual(job.status, "Complete")
        self.assertEqual(job.result, 0.85)
        mock_model.predict.assert_called_once()
        mock_parent_queue.move_to_complete.assert_called_once_with(job.job_id)


class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete end-to-end workflows"""

    def test_simulation_to_training_workflow(self):
        """Test workflow from simulation to training"""
        from simulator.simutil import Color

        game_params = GameParameters.copy()
        game_params.update({
            'x_len': 4, 'y_len': 4, 'octo_num_arms': 1,
            'limb_rows': 2, 'limb_cols': 1, 'num_iterations': 2,
            'epochs': 1, 'batch_size': 2,
        })

        trainer = SuckerTrainer(game_params)

        # Create synthetic data directly
        data = {
            'state_data': [float(i) / 10 for i in range(10)],
            'gt_data': [Color(float(i) / 10 + 0.05, 0.5, 0.5) for i in range(10)],
        }

        train_dataset, test_dataset = trainer.data_format(data)
        model = trainer.train(train_dataset, GENERATE_TENSORBOARD=False)

        test_input = np.array([[0.5, 0.6]])
        prediction = model.predict(test_input, verbose=0)

        self.assertIsNotNone(model)
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.shape, (1, 1))


if __name__ == '__main__':
    unittest.main(verbosity=2)
