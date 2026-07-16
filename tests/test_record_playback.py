"""End-to-end record -> playback integration (record & replay, Phase 4).

Drives the REAL HeadlessRunner + SimRecorder through the server's simulate
handler into a tmp runs dir, then exercises list_runs/load_run/get_frame over
the same server. Socket-free (FakeWebSocket); no port bound.
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "visualizer"))

from helpers import make_config

from simulator.headless_runner import HeadlessRunner
from visualizer.websocket_server import OctopusSimulationServer


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send(self, message):
        self.sent.append(json.loads(message))


def _last(ws, mtype):
    for m in reversed(ws.sent):
        if m["type"] == mtype:
            return m
    raise AssertionError(f"no {mtype} in {[m['type'] for m in ws.sent]}")


class TestRecordPlayback(unittest.TestCase):
    def setUp(self):
        self.runs_dir = tempfile.mkdtemp(prefix="record_playback_")
        self.srv = OctopusSimulationServer(runs_dir=self.runs_dir)

        # A small, fast scenario: patch the runner factory to build a tiny
        # config regardless of the server's VIZ_ILQR profile, so the test
        # stays quick (no 8-arm iLQR compile).
        small = make_config(
            x_len=8, y_len=8, limb_rows=3, octo_num_arms=2, agent_number_of_agents=2
        )

        class SmallRunner(HeadlessRunner):
            def __init__(self, cfg, run_id=None, label="", db_path=None,
                         setup=None):
                cfg = replace(
                    small,
                    run=replace(small.run, num_iterations=cfg.run.num_iterations),
                    output=replace(
                        small.output, record_run=True, record_ilqr_history=False
                    ),
                )
                super().__init__(cfg, run_id=run_id, label=label,
                                 db_path=db_path, setup=setup)

        self.srv.runner_factory = SmallRunner

    def _run(self, coro):
        return asyncio.run(coro)

    def test_simulate_then_playback_round_trip(self):
        ws = FakeWS()
        self.srv.clients.add(ws)

        # 1) Simulate 2 frames end to end.
        self._run(
            self.srv.handle_message(
                ws,
                {
                    "type": "simulate",
                    "req_id": "a1",
                    "data": {"num_frames": 2, "label": "itest"},
                },
            )
        )
        complete = _last(ws, "simulate_complete")["data"]
        self.assertEqual(complete["status"], "complete")
        self.assertEqual(complete["frames_recorded"], 3)  # frame 0 + 2
        run_id = complete["run_id"]
        self.assertTrue(os.path.exists(os.path.join(self.runs_dir, f"{run_id}.duckdb")))
        self.assertIsNone(self.srv._active)

        # 2) list_runs shows it.
        ws2 = FakeWS()
        self._run(
            self.srv.handle_message(
                ws2, {"type": "list_runs", "req_id": "l1", "data": {}}
            )
        )
        runs = _last(ws2, "runs_list")["data"]["runs"]
        self.assertIn(run_id, [r["run_id"] for r in runs])

        # 3) load_run returns full meta.
        ws3 = FakeWS()
        self._run(
            self.srv.handle_message(
                ws3, {"type": "load_run", "req_id": "c1", "data": {"run_id": run_id}}
            )
        )
        meta = _last(ws3, "run_meta")["data"]
        self.assertEqual(meta["frames_recorded"], 3)
        self.assertEqual(len(meta["background"]), 8)  # y_len rows

        # 4) get_frame(last).state equals simulate_complete.final_state.
        #    Both come from RunStore.get_frame (D15) -> trivially exact; this
        #    guards the read-back path, not float fidelity.
        ws4 = FakeWS()
        self._run(
            self.srv.handle_message(
                ws4,
                {
                    "type": "get_frame",
                    "req_id": "d1",
                    "data": {"run_id": run_id, "frame": 2},
                },
            )
        )
        frame = _last(ws4, "frame_data")["data"]
        self.assertEqual(frame["state"], complete["final_state"])

    def test_get_frame_out_of_range(self):
        ws = FakeWS()
        self.srv.clients.add(ws)
        self._run(
            self.srv.handle_message(
                ws, {"type": "simulate", "req_id": "a1", "data": {"num_frames": 2}}
            )
        )
        run_id = _last(ws, "simulate_complete")["data"]["run_id"]
        ws2 = FakeWS()
        self._run(
            self.srv.handle_message(
                ws2,
                {
                    "type": "get_frame",
                    "req_id": "d9",
                    "data": {"run_id": run_id, "frame": 99},
                },
            )
        )
        err = ws2.sent[-1]
        self.assertEqual(err["data"]["code"], "frame_out_of_range")
        self.assertEqual(err["data"]["detail"]["max_frame"], 2)


if __name__ == "__main__":
    unittest.main()
