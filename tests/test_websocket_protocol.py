"""Socket-free tests for the v2 websocket protocol (record & replay, Phase 4).

Each test calls `server.handle_message(fake_ws, msg)` directly (async wrapped in
asyncio.run) with a FakeWebSocket that records what was sent. A fake runner is
injected via `server.runner_factory`, so no real simulation runs and no port is
bound.
"""

import asyncio
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "visualizer"))

from octopus_ai.config import VIZ_ILQR, config_to_flat
from simulator.headless_runner import RunSummary
from simulator.simutil import MovementMode
from visualizer.websocket_server import (
    ActiveRun,
    OctopusSimulationServer,
    merge_flat_overrides,
)


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send(self, message):
        self.sent.append(json.loads(message))


class FakeRunner:
    """Configurable stand-in for HeadlessRunner (no DuckDB, no sim)."""

    progress_frames = 3
    status = "complete"
    frames_recorded = 4
    raise_exc = None
    respect_stop = False

    def __init__(self, cfg, run_id=None, label="", db_path=None):
        self.cfg = cfg
        self.run_id = run_id
        self.db_path = db_path

    def run(self, progress_cb=None, should_stop=None):
        if self.raise_exc:
            raise self.raise_exc
        for f in range(1, self.progress_frames + 1):
            if self.respect_stop and should_stop is not None and should_stop():
                return RunSummary("cancelled", f, 0.1, {})
            if progress_cb is not None:
                progress_cb(
                    {
                        "frame": f,
                        "num_frames": self.progress_frames,
                        "visibility_score": 0.1,
                        "prey_captured": 0,
                        "frame_ms": 10.0,
                        "elapsed_s": 0.1 * f,
                    }
                )
        return RunSummary(self.status, self.frames_recorded, 0.3, {})


def run_async(coro):
    return asyncio.run(coro)


def make_server(**runner_attrs):
    srv = OctopusSimulationServer(runs_dir="/tmp/does_not_exist_runs")

    class R(FakeRunner):
        pass

    for k, v in runner_attrs.items():
        setattr(R, k, v)
    srv.runner_factory = R
    return srv


def types_of(ws):
    return [m["type"] for m in ws.sent]


class TestMergeFlatOverrides(unittest.TestCase):
    def setUp(self):
        self.flat = config_to_flat(VIZ_ILQR)

    def test_enum_by_name(self):
        merged, ignored = merge_flat_overrides(
            self.flat, {"octo_movement_mode": "RANDOM"}
        )
        self.assertEqual(merged["octo_movement_mode"], MovementMode.RANDOM)
        self.assertEqual(ignored, [])

    def test_unknown_member_raises(self):
        from visualizer.websocket_server import BadRequestError

        with self.assertRaises(BadRequestError):
            merge_flat_overrides(self.flat, {"octo_movement_mode": "NOPE"})

    def test_type_coercion_and_ignore(self):
        merged, ignored = merge_flat_overrides(
            self.flat, {"x_len": "42", "bogus_key": 1}
        )
        self.assertEqual(merged["x_len"], 42)
        self.assertIsInstance(merged["x_len"], int)
        self.assertEqual(ignored, ["bogus_key"])


class TestServerInfo(unittest.TestCase):
    def test_server_info_shape_and_json(self):
        srv = make_server()
        info = srv._server_info()
        self.assertEqual(info["data"]["protocol"], 2)
        self.assertIsNone(info["data"]["active_run"])
        # default_config must be JSON-serializable with enums as names.
        text = json.dumps(
            info,
            default=__import__(
                "octopus_ai.config", fromlist=["json_default"]
            ).json_default,
        )
        self.assertIn("ILQR", text)  # octo_movement_mode enum -> name


class TestSimulateHappyPath(unittest.TestCase):
    def test_started_progress_complete(self):
        srv = make_server()
        ws = FakeWS()
        srv.clients.add(ws)
        run_async(
            srv.handle_message(
                ws, {"type": "simulate", "req_id": "a1", "data": {"num_frames": 3}}
            )
        )
        t = types_of(ws)
        self.assertEqual(t[0], "simulate_started")
        self.assertIn("simulate_progress", t)
        self.assertEqual(t[-1], "simulate_complete")
        self.assertIsNone(srv._active)
        started = ws.sent[0]["data"]
        self.assertTrue(started["config"]["record_run"])
        self.assertEqual(
            started["req_id"] if "req_id" in started else ws.sent[0]["req_id"], "a1"
        )
        complete = ws.sent[-1]["data"]
        self.assertEqual(complete["status"], "complete")
        self.assertEqual(complete["frames_recorded"], 4)


class TestSimulateBusy(unittest.TestCase):
    def test_busy_rejects_second(self):
        srv = make_server()
        srv._active = ActiveRun("existing", 10)  # pretend one is running
        ws = FakeWS()
        run_async(
            srv.handle_message(ws, {"type": "simulate", "req_id": "b1", "data": {}})
        )
        self.assertEqual(ws.sent[-1]["type"], "error")
        self.assertEqual(ws.sent[-1]["data"]["code"], "busy")


class TestSimulateBadFrames(unittest.TestCase):
    def test_bad_num_frames(self):
        srv = make_server()
        for bad in (0, -1, "abc", 20000):
            ws = FakeWS()
            run_async(
                srv.handle_message(
                    ws, {"type": "simulate", "req_id": "x", "data": {"num_frames": bad}}
                )
            )
            self.assertEqual(ws.sent[-1]["type"], "error", f"for {bad!r}")
            self.assertEqual(ws.sent[-1]["data"]["code"], "bad_request")


class TestSimulateCancel(unittest.TestCase):
    def test_cancel_path(self):
        srv = make_server(respect_stop=True, progress_frames=50)
        ws = FakeWS()
        srv.clients.add(ws)

        async def scenario():
            # Kick off simulate, and cancel it from a concurrent task once the
            # active run appears.
            async def canceller():
                while srv._active is None:
                    await asyncio.sleep(0)
                await srv.handle_message(
                    ws, {"type": "simulate_cancel", "req_id": "c1", "data": {}}
                )

            await asyncio.gather(
                srv.handle_message(
                    ws, {"type": "simulate", "req_id": "a1", "data": {"num_frames": 50}}
                ),
                canceller(),
            )

        run_async(scenario())
        self.assertIn("simulate_cancel_ack", types_of(ws))
        complete = ws.sent[-1]
        self.assertEqual(complete["type"], "simulate_complete")
        self.assertEqual(complete["data"]["status"], "cancelled")

    def test_cancel_when_idle(self):
        srv = make_server()
        ws = FakeWS()
        run_async(
            srv.handle_message(
                ws, {"type": "simulate_cancel", "req_id": "c1", "data": {}}
            )
        )
        self.assertEqual(ws.sent[-1]["data"]["code"], "not_running")


class TestSimulateFailure(unittest.TestCase):
    def test_run_exception_reports_failed(self):
        srv = make_server(raise_exc=RuntimeError("boom"))
        ws = FakeWS()
        srv.clients.add(ws)
        run_async(
            srv.handle_message(
                ws, {"type": "simulate", "req_id": "a1", "data": {"num_frames": 3}}
            )
        )
        self.assertIsNone(srv._active)  # cleared even on failure
        complete = ws.sent[-1]
        self.assertEqual(complete["type"], "simulate_complete")
        self.assertEqual(complete["data"]["status"], "failed")
        self.assertEqual(complete["data"]["error"], "boom")


class TestProgressCoalescing(unittest.TestCase):
    def test_coalesces_backlog(self):
        srv = make_server()

        async def scenario():
            active = ActiveRun("r", 100)
            broadcasts = []

            async def capture(obj):
                if obj["type"] == "simulate_progress":
                    broadcasts.append(obj)

            srv._broadcast = capture
            # Pre-fill the queue with 100 updates, then mark done.
            for f in range(1, 101):
                active.progress_q.put_nowait(
                    {
                        "frame": f,
                        "num_frames": 100,
                        "visibility_score": 0.1,
                        "prey_captured": 0,
                        "frame_ms": 1.0,
                        "elapsed_s": 0.01 * f,
                    }
                )
            active.done = True
            await srv._drain_progress(active)
            return broadcasts

        broadcasts = run_async(scenario())
        # Coalescing keeps only the latest of a backlog: far fewer than 100,
        # and the last one carries the final frame.
        self.assertLess(len(broadcasts), 100)
        self.assertEqual(broadcasts[-1]["data"]["frame"], 100)


class TestV1Tombstones(unittest.TestCase):
    def test_v1_types_gone(self):
        srv = make_server()
        for mtype in ("config_update", "simulation_control"):
            ws = FakeWS()
            run_async(
                srv.handle_message(ws, {"type": mtype, "req_id": "x", "data": {}})
            )
            self.assertEqual(ws.sent[-1]["data"]["code"], "gone")

    def test_unknown_type(self):
        srv = make_server()
        ws = FakeWS()
        run_async(srv.handle_message(ws, {"type": "wat", "req_id": "x", "data": {}}))
        self.assertEqual(ws.sent[-1]["data"]["code"], "unknown_type")


class TestPlaybackErrors(unittest.TestCase):
    def test_unknown_run(self):
        srv = make_server()
        ws = FakeWS()
        run_async(
            srv.handle_message(
                ws, {"type": "load_run", "req_id": "x", "data": {"run_id": "nope"}}
            )
        )
        self.assertEqual(ws.sent[-1]["data"]["code"], "unknown_run")

    def test_run_in_progress(self):
        srv = make_server()
        srv._active = ActiveRun("live", 10)
        ws = FakeWS()
        run_async(
            srv.handle_message(
                ws,
                {
                    "type": "get_frame",
                    "req_id": "x",
                    "data": {"run_id": "live", "frame": 0},
                },
            )
        )
        self.assertEqual(ws.sent[-1]["data"]["code"], "run_in_progress")

    def test_req_id_echoed(self):
        srv = make_server()
        ws = FakeWS()
        run_async(
            srv.handle_message(
                ws, {"type": "load_run", "req_id": "zz", "data": {"run_id": "nope"}}
            )
        )
        self.assertEqual(ws.sent[-1]["req_id"], "zz")


class TestListRunsWithActive(unittest.TestCase):
    def test_active_run_included_without_opening_file(self):
        srv = make_server()
        srv._active = ActiveRun("live_run", 120)
        srv._active.frame = 5
        ws = FakeWS()
        run_async(
            srv.handle_message(ws, {"type": "list_runs", "req_id": "l1", "data": {}})
        )
        reply = ws.sent[-1]
        self.assertEqual(reply["type"], "runs_list")
        ids = [r["run_id"] for r in reply["data"]["runs"]]
        self.assertIn("live_run", ids)
        active_row = reply["data"]["runs"][0]
        self.assertEqual(active_row["status"], "running")
        self.assertEqual(active_row["frames_recorded"], 6)  # frame+1


if __name__ == "__main__":
    unittest.main()
