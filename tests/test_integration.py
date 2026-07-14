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
from OctoConfig import DEFAULT
from helpers import make_config, make_flat


class TestSimulationIntegration(unittest.TestCase):
    """Test complete simulation workflows"""

    def setUp(self):
        """Set up integration test environment"""
        # Clear class-level shared agents list
        AgentGenerator.agents = []
        self._base = dict(
            x_len=8,
            y_len=8,
            octo_num_arms=2,
            limb_rows=3,
            limb_cols=2,
            agent_number_of_agents=2,
            num_iterations=5,
            inference_location=InferenceLocation.LOCAL,
            # These tests call move() with no agent, valid only in RANDOM.
            octo_movement_mode=MovementMode.RANDOM,
            limb_movement_mode=MovementMode.RANDOM,
            agent_movement_mode=MovementMode.RANDOM,
        )
        self.game_params = make_config(**self._base)

    def test_complete_simulation_cycle(self):
        """Test a complete simulation cycle with all components"""
        surface = RandomSurface(self.game_params)
        self.assertIsNotNone(surface)

        octopus = Octopus(self.game_params)
        self.assertIsNotNone(octopus)
        self.assertEqual(len(octopus.limbs),
                         self.game_params.octopus.num_arms)

        agent_gen = AgentGenerator(self.game_params)
        agent_gen.generate(self.game_params.agents.count)
        self.assertEqual(len(agent_gen.agents),
                         self.game_params.agents.count)

        for iteration in range(self.game_params.run.num_iterations):
            agent_gen.increment_all(octopus)
            octopus.move()

            # set_color runs find_color + applies colors internally
            octopus.set_color(surface)

            visibility = octopus.visibility(surface)
            self.assertIsInstance(visibility, float)
            self.assertGreaterEqual(visibility, 0.0)

    def test_multi_agent_interaction(self):
        """Test interactions between multiple agents and octopus.

        agent_number_of_agents=0 says "don't auto-spawn; this test makes its
        own agents below". It is inert either way - AgentGenerator never
        reads the count, callers pass it to generate() - but it is stated as
        config rather than by mutating the shared fixture mid-test.
        """
        params = make_config(**{**self._base, 'agent_number_of_agents': 0})
        surface = RandomSurface(params)
        octopus = Octopus(params)

        agent_gen = AgentGenerator(params)
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
        """Test simulation with every movement mode.

        RANDOM ignores agents; the spring modes require an agent object, so
        pass the generator through. All three must run a step and colour
        without blowing up.
        """
        movement_modes = [
            MovementMode.RANDOM,
            MovementMode.LUMPED_SPRING,
            MovementMode.SPRING_CHAIN,
        ]

        for mode in movement_modes:
            with self.subTest(movement_mode=mode):
                # agents themselves stay RANDOM; the spring modes describe
                # how the octopus moves, not the agents
                params = make_config(**{
                    **self._base,
                    'octo_movement_mode': mode,
                    'limb_movement_mode': mode,
                    'agent_movement_mode': MovementMode.RANDOM,
                })

                surface = RandomSurface(params)
                octopus = Octopus(params)
                agent_gen = AgentGenerator(params)
                agent_gen.generate(1)

                for _ in range(2):
                    agent_gen.increment_all(octopus)
                    if mode == MovementMode.RANDOM:
                        octopus.move()
                    else:
                        octopus.move(agent_gen)
                    octopus.set_color(surface)

    def test_default_profile_values_are_sane(self):
        """Sanity-check the shipped DEFAULT profile.

        This used to also assert that required KEYS were present and
        non-None. That half is gone: the fields are dataclass attributes
        now, so a missing key is an AttributeError at import rather than
        something a test has to go looking for. What is still worth
        checking is that the shipped VALUES are sensible.
        """
        self.assertGreater(DEFAULT.world.x_len, 0)
        self.assertGreater(DEFAULT.world.y_len, 0)
        self.assertGreater(DEFAULT.octopus.num_arms, 0)
        self.assertGreater(DEFAULT.octopus.limb.rows, 0)
        self.assertGreater(DEFAULT.octopus.limb.cols, 0)
        self.assertGreater(DEFAULT.agents.count, 0)

        self.assertGreater(DEFAULT.training.epochs, 0)
        self.assertGreater(DEFAULT.training.batch_size, 0)
        self.assertGreater(DEFAULT.training.test_size, 0.0)
        self.assertLess(DEFAULT.training.test_size, 1.0)

        # a sucker must be able to sit at least as far out as its minimum
        self.assertLessEqual(DEFAULT.octopus.limb.min_sucker_distance,
                             DEFAULT.octopus.limb.max_sucker_distance)


class TestDataGenerationIntegration(unittest.TestCase):
    """Test data generation integration"""

    def setUp(self):
        """Set up data generation tests"""
        self._base = dict(
            x_len=5,
            y_len=5,
            octo_num_arms=1,
            limb_rows=2,
            limb_cols=2,
            num_iterations=3,
            inference_location=InferenceLocation.LOCAL,
        )
        self.game_params = make_config(**self._base)

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

    def test_datagen_accepts_legacy_flat_dict(self):
        """OctoDatagen still takes a flat dict.

        The trainers hand it a Config now, but from_game_parameters is
        deliberately tolerant and the wire protocol still speaks flat, so
        this path has to keep working. The snapshot in the payload is
        whatever the caller passed, not a normalized form.
        """
        from octo_datagen import OctoDatagen

        flat = make_flat(**self._base)
        datagen = OctoDatagen(flat)
        data = datagen.run_color_datagen()

        self.assertEqual(data['game_parameters'], flat)
        self.assertGreater(len(data['state_data']), 0)


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

        game_params = make_config(
            x_len=4, y_len=4, octo_num_arms=1,
            limb_rows=2, limb_cols=1, num_iterations=2,
            epochs=1, batch_size=2,
        )

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
