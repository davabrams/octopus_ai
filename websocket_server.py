#!/usr/bin/env python3
"""
WebSocket server for Octopus AI visualization
Integrates with your existing octopus AI simulation code
"""

import asyncio
import json
import logging
import math
import random
import threading
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

import numpy as np
import websockets

from OctoConfig import GameConfig, TrainingConfig, GameParameters, TrainingParameters
from simulator.agent_generator import AgentGenerator
from simulator.octopus_generator import Octopus
from simulator.surface_generator import RandomSurface


class OctopusSimulationServer:
    def __init__(self, port=8765):
        self.port = port
        self.clients: set = set()
        self.config: GameConfig = GameParameters
        self.training_config: TrainingConfig = TrainingParameters

        # Simulation state
        self.is_running = False
        self.iteration = 0
        self.simulation_thread = None
        self.simulation_lock = threading.Lock()

        # Initialize simulation components
        self.surface = None
        self.octopus = None
        self.agent_generator = None
        self.visibility_score = 0.0
        self.fps = 0.0

        # Performance tracking
        self.last_frame_time = time.time()

        self.setup_simulation()

    def setup_simulation(self):
        """Initialize the simulation with current config"""
        try:
            self.surface = RandomSurface(self.config)
            self.octopus = Octopus(self.config)
            self.agent_generator = AgentGenerator(self.config)
            self.agent_generator.generate(
                num_agents=self.config.agent_number_of_agents
            )
        except Exception as e:
            print(f"Error setting up simulation: {e}")
            # Create mock data if simulation setup fails
            self.setup_mock_simulation()

    def setup_mock_simulation(self):
        """Create mock simulation data for testing without full simulation"""

        class MockSurface:
            def __init__(self, x_len, y_len):
                self.grid = np.random.randint(2, size=(y_len, x_len))

            def get_color_at(self, x, y):
                try:
                    return float(self.grid[int(y)][int(x)])
                except Exception:
                    return 0.0

        class MockOctopus:
            def __init__(self, x, y, num_arms, limb_rows, limb_cols):
                self.x = x
                self.y = y
                self.num_arms = num_arms
                self.limbs = []
                self.suckers = []

                for i in range(num_arms):
                    angle = (i / num_arms) * 2 * math.pi
                    limb = []
                    for j in range(limb_rows):
                        distance = 1 + (j / limb_rows) * 4
                        limb_x = x + math.cos(angle) * distance
                        limb_y = y + math.sin(angle) * distance
                        limb.append({"x": limb_x, "y": limb_y})
                        for k in range(limb_cols):
                            offset = (k - limb_cols / 2 + 0.5) * 0.3
                            sucker_x = limb_x + math.cos(
                                angle + math.pi / 2
                            ) * offset
                            sucker_y = limb_y + math.sin(
                                angle + math.pi / 2
                            ) * offset
                            self.suckers.append(
                                {
                                    "x": sucker_x,
                                    "y": sucker_y,
                                    "color": 0.5,
                                    "target_color": 0.0,
                                }
                            )
                    self.limbs.append(limb)

            def move(self):
                self.x += (random.random() - 0.5) * 0.1
                self.y += (random.random() - 0.5) * 0.1

            def set_color(self, surface, model=None):
                for sucker in self.suckers:
                    target = surface.get_color_at(sucker["x"], sucker["y"])
                    sucker["target_color"] = target
                    diff = target - sucker["color"]
                    max_change = 0.25
                    change = max(-max_change, min(max_change, diff))
                    sucker["color"] += change

        class MockAgentGenerator:
            def __init__(self, config):
                self.agents = []
                self.config = config

            def generate(self, num_agents, x_bounds, y_bounds):
                self.agents = []
                for _ in range(num_agents):
                    self.agents.append(
                        {
                            "x": random.uniform(x_bounds[0], x_bounds[1]),
                            "y": random.uniform(y_bounds[0], y_bounds[1]),
                            "type": (
                                "predator"
                                if random.random() > 0.5
                                else "prey"
                            ),
                            "velocity": random.random() * 0.2,
                            "angle": random.random() * 2 * math.pi,
                        }
                    )

            def increment_all(self):
                for agent in self.agents:
                    agent["x"] += (
                        math.cos(agent["angle"]) * agent["velocity"]
                    )
                    agent["y"] += (
                        math.sin(agent["angle"]) * agent["velocity"]
                    )
                    agent["angle"] += (random.random() - 0.5) * 0.1
                    agent["x"] = max(
                        0, min(self.config.x_len, agent["x"])
                    )
                    agent["y"] = max(
                        0, min(self.config.y_len, agent["y"])
                    )

        self.surface = MockSurface(self.config.x_len, self.config.y_len)
        self.octopus = MockOctopus(
            self.config.x_len / 2,
            self.config.y_len / 2,
            self.config.octo_num_arms,
            self.config.limb_rows,
            self.config.limb_cols,
        )
        self.agent_generator = MockAgentGenerator(self.config)
        self.agent_generator.generate(
            self.config.agent_number_of_agents,
            (0, self.config.x_len),
            (0, self.config.y_len),
        )

    def calculate_visibility_score(self):
        """Calculate the visibility/camouflage score"""
        try:
            total_error = 0
            count = 0

            for sucker in self.octopus.suckers:
                target_color = self.surface.get_color_at(
                    sucker["x"], sucker["y"]
                )
                error = abs(sucker["color"] - target_color)
                total_error += error
                count += 1

            if count > 0:
                self.visibility_score = total_error / count
            else:
                self.visibility_score = 0.0

        except Exception as e:
            print(f"Error calculating visibility score: {e}")
            self.visibility_score = 0.0

    def get_simulation_state(self):
        """Get current simulation state as JSON-serializable dict"""
        try:
            current_time = time.time()
            if hasattr(self, "last_frame_time"):
                frame_time = current_time - self.last_frame_time
                self.fps = 1.0 / frame_time if frame_time > 0 else 0
            self.last_frame_time = current_time

            state = {
                "background": (
                    self.surface.grid.tolist()
                    if hasattr(self.surface, "grid")
                    else []
                ),
                "octopus": {
                    "head": {
                        "x": float(self.octopus.x),
                        "y": float(self.octopus.y),
                    },
                    "limbs": self.octopus.limbs,
                    "suckers": self.octopus.suckers,
                },
                "agents": (
                    self.agent_generator.agents
                    if hasattr(self.agent_generator, "agents")
                    else []
                ),
                "metadata": {
                    "iteration": self.iteration,
                    "visibility_score": float(self.visibility_score),
                    "fps": float(self.fps),
                },
            }
            return state
        except Exception as e:
            print(f"Error getting simulation state: {e}")
            return {"error": str(e)}

    def update_config(self, new_config):
        """Update simulation configuration"""
        with self.simulation_lock:
            for key, value in new_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            self.setup_simulation()
            self.iteration = 0

    def simulation_step(self):
        """Perform one simulation step"""
        try:
            with self.simulation_lock:
                if hasattr(self.agent_generator, "increment_all"):
                    self.agent_generator.increment_all()

                if hasattr(self.octopus, "move"):
                    self.octopus.move()

                if hasattr(self.octopus, "set_color"):
                    self.octopus.set_color(self.surface)

                self.calculate_visibility_score()

                self.iteration += 1

                max_iterations = self.config.num_iterations
                if max_iterations > 0 and self.iteration >= max_iterations:
                    self.is_running = False

        except Exception as e:
            print(f"Error in simulation step: {e}")

    async def simulation_loop(self):
        """Main simulation loop"""
        while self.is_running:
            self.simulation_step()

            if self.clients:
                state = self.get_simulation_state()
                message = {"type": "simulation_state", "data": state}

                disconnected = set()
                for client in self.clients:
                    try:
                        await client.send(json.dumps(message))
                    except websockets.exceptions.ConnectionClosed:
                        disconnected.add(client)

                self.clients -= disconnected

            await asyncio.sleep(0.1)  # 10 FPS

    def start_simulation(self):
        """Start the simulation"""
        if not self.is_running:
            self.is_running = True

    def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False

    def reset_simulation(self):
        """Reset the simulation"""
        with self.simulation_lock:
            self.iteration = 0
            self.setup_simulation()

    async def handle_client(self, websocket, path):
        """Handle a new WebSocket client connection"""
        print(f"Client connected: {websocket.remote_address}")
        self.clients.add(websocket)

        try:
            initial_state = self.get_simulation_state()
            await websocket.send(
                json.dumps({"type": "simulation_state", "data": initial_state})
            )

            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_message(websocket, data)
                except json.JSONDecodeError:
                    print(f"Invalid JSON received: {message}")
                except Exception as e:
                    print(f"Error handling message: {e}")

        except websockets.exceptions.ConnectionClosed:
            print(f"Client disconnected: {websocket.remote_address}")
        finally:
            self.clients.discard(websocket)

    async def handle_message(self, websocket, data):
        """Handle incoming WebSocket messages"""
        message_type = data.get("type")
        message_data = data.get("data", {})

        if message_type == "config_update":
            self.update_config(message_data)
            await websocket.send(
                json.dumps(
                    {"type": "config_response", "data": {"status": "updated"}}
                )
            )

        elif message_type == "simulation_control":
            action = message_data.get("action")

            if action == "play":
                self.start_simulation()
            elif action == "pause":
                self.stop_simulation()
            elif action == "reset":
                self.reset_simulation()

            await websocket.send(
                json.dumps(
                    {
                        "type": "simulation_control_response",
                        "data": {"action": action, "status": "completed"},
                    }
                )
            )

        else:
            print(f"Unknown message type: {message_type}")

    async def start_server(self):
        """Start the WebSocket server"""
        print(f"Starting Octopus AI WebSocket server on port {self.port}")

        asyncio.create_task(self.simulation_loop())

        async with websockets.serve(
            self.handle_client, "localhost", self.port
        ):
            print(f"Server running on ws://localhost:{self.port}")
            await asyncio.Future()


def main():
    """Main entry point"""
    logging.basicConfig(level=logging.INFO)

    server = OctopusSimulationServer(port=8765)

    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    main()
