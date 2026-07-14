"""Frame capture + video stitching for the octopus visualizer.

When GameParameters['save_images'] is on, octo_viz saves one PNG per frame
into a per-run folder, then stitches them into an MP4 at the end of the
run. Encoding shells out to the system ffmpeg binary (no extra Python
dependency); if ffmpeg is missing, the PNGs are left in place and a clear
message is printed instead of crashing.

Layout (both under logs/, which is gitignored):
    logs/frames/<run_stamp>/frame_00001.png ...
    logs/videos/octo_run_<run_stamp>.mp4
"""
import os
import shutil
import subprocess
import time


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES_DIR = os.path.join(ROOT_DIR, "logs", "frames")
VIDEOS_DIR = os.path.join(ROOT_DIR, "logs", "videos")


class FrameRecorder:
    """Saves per-frame PNGs to a run folder and stitches them to MP4."""

    def __init__(self, fps: int = 5, run_stamp: str = None,
                 base_dir: str = None):
        """fps: playback rate of the stitched video.

        run_stamp: folder/file name for this run; defaults to a timestamp.

        base_dir: where frames/ and videos/ live. Defaults to the project's
            logs/ directory. Tests MUST pass a tmp path here - otherwise
            they write real videos into the repo, and a failure mid-test
            leaves them behind.
        """
        self.fps = fps
        self.run_stamp = run_stamp or time.strftime("%Y%m%d-%H%M%S")
        if base_dir is None:
            self.frames_root = FRAMES_DIR
            self.videos_root = VIDEOS_DIR
        else:
            self.frames_root = os.path.join(base_dir, "frames")
            self.videos_root = os.path.join(base_dir, "videos")
        self.frame_dir = os.path.join(self.frames_root, self.run_stamp)
        os.makedirs(self.frame_dir, exist_ok=True)
        self.frame_count = 0

    def save_frame(self, fig) -> str:
        """Save the current figure as the next numbered frame."""
        self.frame_count += 1
        path = os.path.join(self.frame_dir,
                            f"frame_{self.frame_count:05d}.png")
        fig.savefig(path, dpi=100)
        return path

    def stitch_video(self, keep_frames: bool = True) -> str:
        """Stitch saved frames into an MP4 via ffmpeg.

        Returns the video path on success, or "" if nothing was written
        (no frames, or ffmpeg unavailable). Never raises for a missing
        encoder - prints guidance instead.
        """
        if self.frame_count == 0:
            print("No frames saved; skipping video.")
            return ""

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            print(
                "ffmpeg not found on PATH; frames left in "
                f"{self.frame_dir}. Install ffmpeg (e.g. 'brew install "
                "ffmpeg') to enable video stitching."
            )
            return ""

        os.makedirs(self.videos_root, exist_ok=True)
        out_path = os.path.join(
            self.videos_root, f"octo_run_{self.run_stamp}.mp4")
        pattern = os.path.join(self.frame_dir, "frame_%05d.png")

        cmd = [
            ffmpeg, "-y",
            "-framerate", str(self.fps),
            "-i", pattern,
            # pad to even dimensions (H.264 requires it) and set a broadly
            # compatible pixel format
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            out_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print("ffmpeg failed to stitch the video:")
            print(e.stderr[-500:] if e.stderr else str(e))
            return ""

        if not keep_frames:
            shutil.rmtree(self.frame_dir, ignore_errors=True)

        return out_path
