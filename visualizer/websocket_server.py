#!/usr/bin/env python3
"""WebSocket server v2 for the Octopus AI record & replay analyzer.

Two modes, one sim driver. The browser (`visualizer/analyzer.html`) either
runs a fresh headless simulation and watches it record, or scrubs a saved run
frame-by-frame (and iLQR iteration-by-iteration within a frame). The server
never runs the simulation on its own event loop: a `simulate` hands off to
`HeadlessRunner.run` on a worker thread (`asyncio.to_thread`) so socket I/O
stays live during a multi-minute iLQR run, and playback queries go through a
read-only `RunStore` (one `.duckdb` file per run, so completed runs read fine
while another is being written).

This replaces the v1 live-streaming protocol (D7): the old
`config_update`/`simulation_control` messages now return `error code="gone"`.

v2 WIRE PROTOCOL
================
Envelope both directions: ``{"type": str, "req_id": str?, "data": {...}}``.
Every request gets exactly one terminal reply (its response or an ``error``),
echoing ``req_id``. ``simulate_progress``/``simulate_complete`` are also
broadcast to all clients. Floats are rounded to 4 decimals (D10). Enums
serialize as their ``.name``.

On connect::

    {"type":"server_info","data":{"protocol":2,
      "active_run": null | {"run_id":..,"frame":17,"num_frames":120},
      "default_config": { config_to_flat(VIZ_ILQR) }}}

Simulate::

    -> {"type":"simulate","req_id":"a1","data":{
         "num_frames":120, "config":{...flat overrides...},
         "record_ilqr_history":true, "label":"baseline-120"}}
    <- {"type":"simulate_started","req_id":"a1","data":{"run_id":..,
         "num_frames":120,"config":{...merged...},"ignored_keys":[...],
         "db_path":"logs/runs/<id>.duckdb"}}
    <- {"type":"simulate_progress","data":{"run_id":..,"frame":17,
         "num_frames":120,"visibility_score":..,"prey_captured":..,
         "elapsed_s":..,"frame_ms":..}}                         (broadcast)
    <- {"type":"simulate_complete","req_id":"a1","data":{"run_id":..,
         "status":"complete|cancelled|failed","frames_recorded":121,
         "elapsed_s":..,"error":null,"final_state":{...}}}      (broadcast)

    -> {"type":"simulate_cancel","req_id":"a2","data":{}}
    <- {"type":"simulate_cancel_ack","req_id":"a2","data":{"run_id":..}}

Playback::

    -> {"type":"list_runs"}          <- {"type":"runs_list","data":{"runs":[...]}}
    -> {"type":"load_run","data":{"run_id":..}}
                                     <- {"type":"run_meta","data":{...}}
    -> {"type":"get_frame","data":{"run_id":..,"frame":42,"include_ilqr":true}}
                                     <- {"type":"frame_data","data":{...}}
    -> {"type":"get_frames","data":{"run_id":..,"start":0,"count":20}}
                                     <- {"type":"frames_data","data":{...}}

Errors (terminal reply for any failed request)::

    {"type":"error","req_id":..,"data":{"code":..,"message":..,"detail":{}}}
    codes: busy|not_running|unknown_run|run_in_progress|frame_out_of_range|
           bad_request|sim_failed|gone|unknown_type
"""
import asyncio
import base64
import http
import json
import logging
import os
import pathlib
import sys
import time as _time
import threading
from dataclasses import replace

# This lives in visualizer/ but imports top-level project modules; put the repo
# root on sys.path so `python visualizer/websocket_server.py` works.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enum import Enum

import websockets
from websockets.datastructures import Headers
from websockets.http11 import Response

from octopus_ai.config import (
    VIZ_ILQR,
    config_from_flat,
    config_to_flat,
    json_default,
    print_config,
)
from simulator.headless_runner import HeadlessRunner
from simulator.run_store import FrameOutOfRangeError, RunNotFoundError, RunStore
from simulator.sim_recorder import DEFAULT_RUNS_DIR, new_run_id


class BadRequestError(ValueError):
    """A malformed request field; maps to error code 'bad_request'."""


def merge_flat_overrides(flat: dict, overrides: dict):
    """Apply browser overrides onto a flat config, coercing by baseline type.

    Returns (merged, ignored_keys). Coercion mirrors the baseline value's type
    INCLUDING Enum (member lookup by name, D14): a `{"octo_movement_mode":
    "ILQR"}` override becomes MovementMode.ILQR, and an unknown member raises
    BadRequestError rather than silently producing a Config holding the string.
    Unknown keys are ignored (collected, not fatal).
    """
    merged = dict(flat)
    ignored = []
    for key, value in overrides.items():
        if key not in flat:
            ignored.append(key)
            continue
        current = flat[key]
        try:
            if isinstance(current, bool):
                value = bool(value)
            elif isinstance(current, Enum):
                value = type(current)[value]  # by name; KeyError if unknown
            elif isinstance(current, int):
                value = int(value)
            elif isinstance(current, float):
                value = float(value)
            # str / None: pass through unchanged
        except (TypeError, ValueError, KeyError) as e:
            raise BadRequestError(
                f"bad value for {key!r}: {value!r}") from e
        merged[key] = value
    return merged, ignored


class ActiveRun:
    """In-flight simulate: cancel signal + progress mailbox + live frame."""

    def __init__(self, run_id: str, num_frames: int):
        self.run_id = run_id
        self.num_frames = num_frames
        self.frame = 0
        self.cancel = threading.Event()
        self.progress_q: asyncio.Queue = asyncio.Queue()
        self.done = False


class OctopusSimulationServer:
    # Factory seam: tests inject a fake runner without monkeypatching modules.
    runner_factory = HeadlessRunner
    HTML_PAGE = "analyzer.html"

    def __init__(self, port=8765, runs_dir: str | None = None):
        self.port = port
        self.clients: set = set()
        self.profile = VIZ_ILQR
        self.runs_dir = runs_dir or DEFAULT_RUNS_DIR
        self.run_store = RunStore(self.runs_dir)
        self._active: ActiveRun | None = None
        self._dispatch = {
            "simulate": self._h_simulate,
            "simulate_cancel": self._h_simulate_cancel,
            "list_runs": self._h_list_runs,
            "load_run": self._h_load_run,
            "get_frame": self._h_get_frame,
            "get_frames": self._h_get_frames,
            "rename_run": self._h_rename_run,
            "upload_image": self._h_upload_image,
        }
        self.uploads_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs", "uploads")

    # ---- send helpers ----------------------------------------------------
    async def _send(self, ws, obj):
        await ws.send(json.dumps(obj, default=json_default))

    async def _broadcast(self, obj):
        msg = json.dumps(obj, default=json_default)
        dead = set()
        for ws in self.clients:
            try:
                await ws.send(msg)
            except websockets.exceptions.ConnectionClosed:
                dead.add(ws)
        self.clients -= dead

    async def _error(self, ws, req_id, code, message, detail=None):
        await self._send(ws, {"type": "error", "req_id": req_id,
                              "data": {"code": code, "message": message,
                                       "detail": detail or {}}})

    def _server_info(self):
        active = None
        if self._active is not None:
            active = {"run_id": self._active.run_id,
                      "frame": self._active.frame,
                      "num_frames": self._active.num_frames}
        return {"type": "server_info",
                "data": {"protocol": 2, "active_run": active,
                         "default_config": config_to_flat(self.profile)}}

    # ---- connection ------------------------------------------------------
    async def handle_client(self, websocket):
        logging.info("Client connected: %s", websocket.remote_address)
        self.clients.add(websocket)
        try:
            await self._send(websocket, self._server_info())
            async for message in websocket:
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    logging.warning("Invalid JSON received: %s", message)
                    continue
                try:
                    await self.handle_message(websocket, data)
                except Exception as e:  # never let one message kill the socket
                    logging.exception("Error handling message: %s", e)
        except websockets.exceptions.ConnectionClosed:
            logging.info("Client disconnected: %s", websocket.remote_address)
        finally:
            self.clients.discard(websocket)

    async def handle_message(self, websocket, data):
        mtype = data.get("type")
        req_id = data.get("req_id")
        mdata = data.get("data", {}) or {}
        if mtype in ("config_update", "simulation_control"):
            await self._error(websocket, req_id, "gone",
                              f"{mtype} was removed in protocol 2; use the "
                              "simulate/playback messages instead")
            return
        handler = self._dispatch.get(mtype)
        if handler is None:
            await self._error(websocket, req_id, "unknown_type",
                              f"unknown message type: {mtype!r}")
            return
        await handler(websocket, req_id, mdata)

    # ---- simulate --------------------------------------------------------
    async def _h_simulate(self, ws, req_id, data):
        if self._active is not None:
            await self._error(ws, req_id, "busy",
                              "a simulation is already running")
            return

        num_frames = data.get("num_frames", self.profile.run.num_iterations)
        if (not isinstance(num_frames, int) or isinstance(num_frames, bool)
                or not (1 <= num_frames <= 10000)):
            await self._error(ws, req_id, "bad_request",
                              "num_frames must be an int in [1, 10000]")
            return

        try:
            merged, ignored = merge_flat_overrides(
                config_to_flat(self.profile), data.get("config", {}) or {})
        except BadRequestError as e:
            await self._error(ws, req_id, "bad_request", str(e))
            return

        cfg = config_from_flat(merged)
        cfg = replace(
            cfg,
            run=replace(cfg.run, num_iterations=num_frames),
            output=replace(cfg.output, record_run=True,
                           record_ilqr_history=data.get(
                               "record_ilqr_history", True)))

        run_id = new_run_id()
        label = str(data.get("label", ""))
        setup = data.get("setup") or {}
        db_path = os.path.join(self.runs_dir, f"{run_id}.duckdb")
        runner = self.runner_factory(cfg, run_id=run_id, label=label,
                                     db_path=db_path, setup=setup)
        active = ActiveRun(run_id, num_frames)
        self._active = active

        # Echo the FINAL flat config (record_run forced on, num_iterations set)
        # so the client sees exactly what will be recorded.
        await self._send(ws, {
            "type": "simulate_started", "req_id": req_id,
            "data": {"run_id": run_id, "num_frames": num_frames,
                     "config": config_to_flat(cfg), "ignored_keys": ignored,
                     "db_path": db_path}})

        loop = asyncio.get_running_loop()

        def progress_cb(info):
            loop.call_soon_threadsafe(active.progress_q.put_nowait, info)

        drain = asyncio.create_task(self._drain_progress(active))
        status, error, summary = "complete", None, None
        try:
            summary = await asyncio.to_thread(
                runner.run, progress_cb, active.cancel.is_set)
            status = summary.status
        except Exception as e:
            status = "failed"
            error = str(e)
            logging.exception("simulate run failed: %s", e)
        finally:
            active.done = True
            await drain
            self._active = None

        frames_recorded = summary.frames_recorded if summary else 0
        elapsed = summary.elapsed_s if summary else 0.0
        final_state = {}
        if status != "failed" and frames_recorded > 0:
            try:
                final_state = self.run_store.get_frame(
                    run_id, frames_recorded - 1)["state"]
            except Exception as e:  # read-back is best-effort
                logging.warning("could not read final_state: %s", e)

        await self._broadcast({
            "type": "simulate_complete", "req_id": req_id,
            "data": {"run_id": run_id, "status": status,
                     "frames_recorded": frames_recorded,
                     "elapsed_s": round(elapsed, 4), "error": error,
                     "final_state": final_state}})

    async def _drain_progress(self, active: ActiveRun):
        """Broadcast progress, coalescing (latest wins) so a slow client can
        never back-pressure the sim thread."""
        while not (active.done and active.progress_q.empty()):
            try:
                info = await asyncio.wait_for(active.progress_q.get(),
                                              timeout=0.02)
            except asyncio.TimeoutError:
                continue
            while not active.progress_q.empty():  # coalesce
                info = active.progress_q.get_nowait()
            active.frame = int(info.get("frame", active.frame))
            await self._broadcast({
                "type": "simulate_progress",
                "data": {"run_id": active.run_id,
                         "frame": info["frame"],
                         "num_frames": info["num_frames"],
                         "visibility_score": round(
                             info["visibility_score"], 4),
                         "prey_captured": info["prey_captured"],
                         "elapsed_s": round(info["elapsed_s"], 4),
                         "frame_ms": round(info["frame_ms"], 4)}})

    async def _h_simulate_cancel(self, ws, req_id, data):
        if self._active is None:
            await self._error(ws, req_id, "not_running",
                              "no simulation is running")
            return
        self._active.cancel.set()
        await self._send(ws, {"type": "simulate_cancel_ack", "req_id": req_id,
                              "data": {"run_id": self._active.run_id}})

    # ---- playback --------------------------------------------------------
    async def _h_list_runs(self, ws, req_id, data):
        active_id = self._active.run_id if self._active else None
        runs = await asyncio.to_thread(self.run_store.list_runs, active_id)
        if self._active is not None:  # synthesize the active row (file locked)
            runs.insert(0, {
                "run_id": self._active.run_id, "label": "",
                "started_at": "", "status": "running",
                "frames_recorded": self._active.frame + 1,
                "has_ilqr_history": self.profile.output.record_ilqr_history,
                "config_summary": {
                    "x_len": self.profile.world.x_len,
                    "y_len": self.profile.world.y_len,
                    "octo_num_arms": self.profile.octopus.num_arms,
                    "octo_movement_mode":
                        self.profile.octopus.movement_mode.name,
                    "inference_mode": self.profile.inference.mode.name}})
        await self._send(ws, {"type": "runs_list", "req_id": req_id,
                              "data": {"runs": runs}})

    async def _h_load_run(self, ws, req_id, data):
        run_id = data.get("run_id")
        if self._active is not None and run_id == self._active.run_id:
            await self._error(ws, req_id, "run_in_progress",
                              "run is still being recorded")
            return
        try:
            meta = await asyncio.to_thread(self.run_store.run_meta, run_id)
        except RunNotFoundError:
            await self._error(ws, req_id, "unknown_run",
                              f"no such run: {run_id}")
            return
        await self._send(ws, {"type": "run_meta", "req_id": req_id,
                              "data": meta})

    async def _h_get_frame(self, ws, req_id, data):
        run_id = data.get("run_id")
        frame = data.get("frame")
        include_ilqr = bool(data.get("include_ilqr", False))
        if self._active is not None and run_id == self._active.run_id:
            await self._error(ws, req_id, "run_in_progress",
                              "run is still being recorded")
            return
        if not isinstance(frame, int) or isinstance(frame, bool):
            await self._error(ws, req_id, "bad_request",
                              "frame must be an int")
            return
        try:
            result = await asyncio.to_thread(
                self.run_store.get_frame, run_id, frame, include_ilqr)
        except RunNotFoundError:
            await self._error(ws, req_id, "unknown_run",
                              f"no such run: {run_id}")
            return
        except FrameOutOfRangeError as e:
            await self._error(ws, req_id, "frame_out_of_range",
                              str(e), {"max_frame": e.max_frame})
            return
        await self._send(ws, {"type": "frame_data", "req_id": req_id,
                              "data": result})

    async def _h_get_frames(self, ws, req_id, data):
        run_id = data.get("run_id")
        start = data.get("start", 0)
        count = data.get("count", 20)
        if self._active is not None and run_id == self._active.run_id:
            await self._error(ws, req_id, "run_in_progress",
                              "run is still being recorded")
            return
        if (not isinstance(start, int) or isinstance(start, bool)
                or not isinstance(count, int) or isinstance(count, bool)
                or count < 1):
            await self._error(ws, req_id, "bad_request",
                              "start/count must be ints, count >= 1")
            return
        try:
            result = await asyncio.to_thread(
                self.run_store.get_frames, run_id, start, count)
        except RunNotFoundError:
            await self._error(ws, req_id, "unknown_run",
                              f"no such run: {run_id}")
            return
        except FrameOutOfRangeError as e:
            await self._error(ws, req_id, "frame_out_of_range",
                              str(e), {"max_frame": e.max_frame})
            return
        await self._send(ws, {"type": "frames_data", "req_id": req_id,
                              "data": result})

    async def _h_upload_image(self, ws, req_id, data):
        content_b64 = data.get("content_b64")
        filename = data.get("filename", "upload.jpg")
        if not content_b64:
            await self._error(ws, req_id, "bad_request",
                              "content_b64 is required")
            return
        try:
            raw = base64.b64decode(content_b64)
        except Exception:
            await self._error(ws, req_id, "bad_request", "invalid base64")
            return
        os.makedirs(self.uploads_dir, exist_ok=True)
        safe_name = f"{int(_time.time())}_{os.path.basename(filename)}"
        path = os.path.join(self.uploads_dir, safe_name)
        with open(path, "wb") as f:
            f.write(raw)
        await self._send(ws, {"type": "image_uploaded", "req_id": req_id,
                              "data": {"path": path}})

    async def _h_rename_run(self, ws, req_id, data):
        run_id = data.get("run_id")
        label = data.get("label", "")
        if not isinstance(label, str):
            await self._error(ws, req_id, "bad_request",
                              "label must be a string")
            return
        try:
            await asyncio.to_thread(
                self.run_store.rename_run, run_id, label)
        except RunNotFoundError:
            await self._error(ws, req_id, "unknown_run",
                              f"no such run: {run_id}")
            return
        await self._send(ws, {"type": "run_renamed", "req_id": req_id,
                              "data": {"run_id": run_id, "label": label}})

    # ---- HTTP page serving (unchanged in P4; flip is P5) -----------------
    def _read_html_page(self):
        page_path = pathlib.Path(__file__).parent / self.HTML_PAGE
        try:
            return page_path.read_bytes(), "text/html; charset=utf-8"
        except FileNotFoundError:
            msg = (f"Visualizer page not found: {self.HTML_PAGE}.").encode()
            return msg, "text/plain; charset=utf-8"

    def process_request(self, connection, request):
        headers = request.headers
        if headers.get("Upgrade", "").lower() == "websocket":
            return None  # proceed with the WebSocket handshake
        if request.path in ("/", "/index.html", "/" + self.HTML_PAGE,
                            "/octopus-visualizer.html"):
            body, content_type = self._read_html_page()
            status = (http.HTTPStatus.OK if content_type.startswith("text/html")
                      else http.HTTPStatus.NOT_FOUND)
            resp_headers = Headers({
                "Content-Type": content_type,
                "Content-Length": str(len(body)),
                "Cache-Control": "no-store"})
            return Response(status.value, status.phrase, resp_headers, body)
        body = (b"Not found. Open http://localhost:%d/ for the analyzer."
                % self.port)
        resp_headers = Headers({"Content-Type": "text/plain; charset=utf-8",
                                "Content-Length": str(len(body))})
        return Response(http.HTTPStatus.NOT_FOUND.value,
                        http.HTTPStatus.NOT_FOUND.phrase, resp_headers, body)

    async def start_server(self):
        print(f"Starting Octopus AI analyzer server on port {self.port}")
        print_config(self.profile, "websocket_server CONFIG")
        async with websockets.serve(self.handle_client, "localhost",
                                    self.port,
                                    process_request=self.process_request):
            print(f"Open the analyzer at http://localhost:{self.port}/  "
                  f"(WebSocket on ws://localhost:{self.port})")
            await asyncio.Future()


def main():
    logging.basicConfig(level=logging.INFO)
    server = OctopusSimulationServer(port=8765)
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    main()
