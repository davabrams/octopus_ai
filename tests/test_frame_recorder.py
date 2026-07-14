"""
Unit tests for the frame recorder / video stitcher.
"""
import os
import shutil
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulator.frame_recorder import FrameRecorder


class TestFrameRecorder(unittest.TestCase):
    def setUp(self):
        # Write into a throwaway tmp dir, NOT the project's logs/: these
        # tests produce real PNGs and a real MP4, and a failure mid-test
        # would otherwise leave artifacts sitting in the repo.
        self.tmp = tempfile.mkdtemp(prefix="octo_frames_test_")
        self.rec = FrameRecorder(fps=5, run_stamp="unittest_run",
                                 base_dir=self.tmp)
        self.fig, self.ax = plt.subplots(figsize=(3, 3))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        plt.close(self.fig)

    def test_save_frame_creates_numbered_png(self):
        self.ax.plot([0, 1], [0, 1])
        p1 = self.rec.save_frame(self.fig)
        p2 = self.rec.save_frame(self.fig)
        self.assertTrue(p1.endswith("frame_00001.png"))
        self.assertTrue(p2.endswith("frame_00002.png"))
        self.assertTrue(os.path.exists(p1))
        self.assertTrue(os.path.exists(p2))
        self.assertEqual(self.rec.frame_count, 2)

    def test_stitch_video_with_no_frames_returns_empty(self):
        self.assertEqual(self.rec.stitch_video(), "")

    def test_stitch_video_produces_mp4(self):
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not installed")
        for _ in range(4):
            self.ax.clear()
            self.ax.plot([0, 1], [0, 1])
            self.rec.save_frame(self.fig)
        video = self.rec.stitch_video(keep_frames=True)
        self.assertTrue(video.endswith(".mp4"))
        self.assertTrue(os.path.exists(video))
        self.assertGreater(os.path.getsize(video), 0)

    def test_stitch_keep_frames_false_removes_dir(self):
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not installed")
        for _ in range(3):
            self.ax.clear()
            self.ax.plot([0, 1], [1, 0])
            self.rec.save_frame(self.fig)
        frame_dir = self.rec.frame_dir
        self.rec.stitch_video(keep_frames=False)
        self.assertFalse(os.path.isdir(frame_dir))


if __name__ == '__main__':
    unittest.main()
