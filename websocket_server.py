#!/usr/bin/env python3
"""
WebSocket server for Octopus AI visualization.

Streams live simulation state as JSON to browser frontends
(octopus-visualizer.html / octopus-ai-visualizer.tsx) at ~10 FPS on
ws://localhost:8765, and accepts config-update and play/pause/reset
control messages. The same port also serves the visualizer HTML page over
plain HTTP, so opening http://localhost:8765/ in a browser loads the UI,
which then upgrades to a WebSocket on the same port. No separate static
file server is needed.

Wire format (message type "simulation_state"):
{
  "background": [[0|1, ...], ...],              # y_len x x_len grid
  "octopus": {
    "head":    {"x": float, "y": float},
    "limbs":   [[{"x": float, "y": float}, ...], ...],   # centerline points
    "suckers": [{"x": float, "y": float,
                 "color": float, "target_color": float}, ...],
  },
  "agents": [{"x": float, "y": float, "type": "prey"|"predator",
              "velocity": float, "angle": float}, ...],
  "metadata": {"iteration": int, "visibility_score": float, "fps": float},
}
"""

import asyncio
import http
import json
import logging
import pathlib
import threading
import time

import websockets
from websockets.datastructures import Headers
from websockets.http11 import Response

from OctoConfig import DEFAULT, config_from_flat, config_to_flat
from simulator.agent_generator import AgentGenerator
from simulator.octopus_generator import Octopus
from simulator.simutil import AgentType, MLMode
from simulator.surface_generator import RandomSurface
from simulator.force_logger import ForceLogger
from training.losses import (
    ClampedTargetLoss,
    ConstraintLoss,
    DeltaColorLayer,
)
from training.models.model_loader import ModelLoader


class OctopusSimulationServer:
    def __init__(self, port=8765):
        self.port = port
        self.clients: set = set()
        # The browser speaks a FLAT config (it sends {"x_len": 20, ...}), so
        # keep a flat mirror for the wire protocol and a typed Config for
        # everything internal. update_config() edits the mirror and rebuilds
        # the Config from it.
        self.profile = DEFAULT
        self.config: dict = config_to_flat(self.profile)
        self.cfg = self.profile

        # Simulation state
        self.is_running = False
        self.iteration = 0
        self.simulation_lock = threading.Lock()

        # Simulation components
        self.surface = None
        self.octopus = None
        self.agent_generator = None
        self.model = None
        self.inference_mode = MLMode.NO_MODEL
        self.visibility_score = 0.0
        self.fps = 0.0

        # Performance tracking
        self.last_frame_time = time.time()

        # Optional force logging (created lazily on first play so the run
        # label reflects when it actually started). All DB access stays on
        # the simulation-loop thread, matching SQLite's threading rules.
        self.force_logger = None

        self.setup_simulation()

    def setup_simulation(self):
        """Initialize the simulation with the current config."""
        # Rebuild the typed config from the flat mirror the browser edits.
        self.cfg = config_from_flat(self.config)

        self.surface = RandomSurface(self.cfg)
        self.octopus = Octopus(self.cfg)
        self.agent_generator = AgentGenerator(self.cfg)
        self.agent_generator.generate(num_agents=self.cfg.agents.count)

        # Optional ML inference: fall back to the heuristic if no model
        # can be loaded for the configured mode.
        self.inference_mode = self.cfg.inference.mode
        self.model = None
        if self.inference_mode is not MLMode.NO_MODEL:
            try:
                self.model = ModelLoader(
                    self.cfg.inference_model_path,
                    custom_objects={
                        "ConstraintLoss": ConstraintLoss,
                        "ClampedTargetLoss": ClampedTargetLoss,
                        "DeltaColorLayer": DeltaColorLayer,
                    },
                ).get_object()
            except Exception as e:
                logging.warning(
                    "Could not load model for %s (%s); "
                    "falling back to heuristic",
                    self.inference_mode, e,
                )
                self.inference_mode = MLMode.NO_MODEL

        # Initial camouflage pass so suckers start matched-ish
        self.octopus.set_color(self.surface)
        self.calculate_visibility_score()

    def calculate_visibility_score(self):
        """Mean squared color error across all suckers (lower = hidden)."""
        try:
            self.visibility_score = float(
                self.octopus.visibility(self.surface)
            )
        except Exception as e:
            logging.error("Error calculating visibility score: %s", e)
            self.visibility_score = 0.0

    def get_simulation_state(self):
        """Get current simulation state as a JSON-serializable dict."""
        try:
            current_time = time.time()
            frame_time = current_time - self.last_frame_time
            self.fps = 1.0 / frame_time if frame_time > 0 else 0.0
            self.last_frame_time = current_time

            limbs = []
            suckers = []
            for limb in self.octopus.limbs:
                limbs.append(
                    [{"x": float(pt.x), "y": float(pt.y)}
                     for pt in limb.center_line]
                )
                for s in limb.suckers:
                    target = s.get_surf_color_at_this_sucker(self.surface)
                    suckers.append({
                        "x": float(s.x),
                        "y": float(s.y),
                        "color": float(s.c.r),
                        "target_color": float(target.r),
                    })

            agents = [
                {
                    "x": float(agent.x),
                    "y": float(agent.y),
                    "type": (
                        "prey"
                        if agent.agent_type == AgentType.PREY
                        else "predator"
                    ),
                    "velocity": float(agent.vx),
                    "angle": float(agent.t),
                }
                for agent in self.agent_generator.agents
            ]

            return {
                "background": self.surface.grid.tolist(),
                "octopus": {
                    "head": {
                        "x": float(self.octopus.x),
                        "y": float(self.octopus.y),
                    },
                    "limbs": limbs,
                    "suckers": suckers,
                },
                "agents": agents,
                "metadata": {
                    "iteration": self.iteration,
                    "visibility_score": float(self.visibility_score),
                    "fps": float(self.fps),
                    "prey_captured": int(
                        self.agent_generator.prey_captured),
                },
            }
        except Exception as e:
            logging.error("Error getting simulation state: %s", e)
            return {"error": str(e)}

    def update_config(self, new_config):
        """Update simulation configuration (dict keys) and rebuild."""
        with self.simulation_lock:
            for key, value in new_config.items():
                if key not in self.config:
                    logging.warning("Ignoring unknown config key: %s", key)
                    continue
                current = self.config[key]
                # Coerce JSON numbers/bools to the existing value's type so
                # e.g. x_len stays an int
                try:
                    if isinstance(current, bool):
                        value = bool(value)
                    elif isinstance(current, int):
                        value = int(value)
                    elif isinstance(current, float):
                        value = float(value)
                except (TypeError, ValueError):
                    logging.warning(
                        "Ignoring config value for %s: %r", key, value
                    )
                    continue
                self.config[key] = value
            self.setup_simulation()
            self.iteration = 0

    def simulation_step(self):
        """Perform one simulation step."""
        try:
            with self.simulation_lock:
                self.agent_generator.increment_all(self.octopus)
                self.octopus.move(self.agent_generator)
                self.agent_generator.remove_captured_prey(self.octopus)
                self.octopus.set_color(
                    self.surface, self.inference_mode, self.model
                )
                self.calculate_visibility_score()

                self.iteration += 1

                if self.cfg.output.log_forces:
                    if self.force_logger is None:
                        self.force_logger = ForceLogger(
                            run_label="websocket_server", config=self.cfg)
                    self.force_logger.log_frame(self.iteration, self.octopus)

                max_iterations = self.cfg.run.num_iterations
                if max_iterations > 0 and self.iteration >= max_iterations:
                    self.is_running = False

        except Exception as e:
            logging.error("Error in simulation step: %s", e)

    async def simulation_loop(self):
        """Main simulation loop (runs regardless; steps only when playing)."""
        while True:
            if self.is_running:
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
        """Start the simulation."""
        self.is_running = True

    def stop_simulation(self):
        """Stop the simulation."""
        self.is_running = False

    def reset_simulation(self):
        """Reset the simulation."""
        with self.simulation_lock:
            self.iteration = 0
            if self.force_logger is not None:
                self.force_logger.close()
                self.force_logger = None
            self.setup_simulation()

    async def handle_client(self, websocket):
        """Handle a new WebSocket client connection.

        websockets >= 13 passes only the connection object (no path arg).
        """
        logging.info("Client connected: %s", websocket.remote_address)
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
                    logging.warning("Invalid JSON received: %s", message)
                except Exception as e:
                    logging.error("Error handling message: %s", e)

        except websockets.exceptions.ConnectionClosed:
            logging.info(
                "Client disconnected: %s", websocket.remote_address
            )
        finally:
            self.clients.discard(websocket)

    async def handle_message(self, websocket, data):
        """Handle incoming WebSocket messages."""
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
            logging.warning("Unknown message type: %s", message_type)

    # Path (relative to this file) of the browser UI served over HTTP.
    HTML_PAGE = "octopus-visualizer.html"

    def _read_html_page(self):
        """Return (body_bytes, content_type) for the visualizer page."""
        page_path = pathlib.Path(__file__).parent / self.HTML_PAGE
        try:
            body = page_path.read_bytes()
            return body, "text/html; charset=utf-8"
        except FileNotFoundError:
            msg = (
                f"Visualizer page not found: {self.HTML_PAGE}. "
                "Expected it next to websocket_server.py."
            ).encode()
            return msg, "text/plain; charset=utf-8"

    def process_request(self, connection, request):
        """Serve the UI over HTTP; let WebSocket upgrades pass through.

        websockets calls this for every incoming request before the
        handshake. Returning a Response short-circuits it (plain HTTP);
        returning None lets the WebSocket upgrade proceed. We detect the
        upgrade via the Connection/Upgrade headers so a normal browser
        navigation to http://localhost:8765/ gets the page instead of the
        "you need a WebSocket client" error.
        """
        headers = request.headers
        connection_hdr = headers.get("Upgrade", "")
        if connection_hdr.lower() == "websocket":
            return None  # proceed with the WebSocket handshake

        if request.path in ("/", "/index.html", "/" + self.HTML_PAGE):
            body, content_type = self._read_html_page()
            status = (
                http.HTTPStatus.OK
                if content_type.startswith("text/html")
                else http.HTTPStatus.NOT_FOUND
            )
            resp_headers = Headers({
                "Content-Type": content_type,
                "Content-Length": str(len(body)),
                "Cache-Control": "no-store",
            })
            return Response(status.value, status.phrase, resp_headers, body)

        body = b"Not found. Open http://localhost:%d/ for the visualizer." \
            % self.port
        resp_headers = Headers({
            "Content-Type": "text/plain; charset=utf-8",
            "Content-Length": str(len(body)),
        })
        return Response(
            http.HTTPStatus.NOT_FOUND.value,
            http.HTTPStatus.NOT_FOUND.phrase,
            resp_headers,
            body,
        )

    async def start_server(self):
        """Start the WebSocket server."""
        print(f"Starting Octopus AI WebSocket server on port {self.port}")

        asyncio.create_task(self.simulation_loop())

        async with websockets.serve(
            self.handle_client,
            "localhost",
            self.port,
            process_request=self.process_request,
        ):
            print(
                f"Open the visualizer at http://localhost:{self.port}/  "
                f"(WebSocket on ws://localhost:{self.port})"
            )
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
