"""Lightweight performance tracking for simulation runs.

A small, dependency-free profiler for the per-frame loop: time named steps,
count calls, and report where the wall clock goes, plus process/Python memory.
Designed to be threaded through an entry point's loop and dumped once at the
end.

Usage:

    perf = PerfTracker(enabled=cfg.output.track_performance)
    while running:
        with perf.track("move"):
            octo.move(ag)
        with perf.track("find_color"):
            octo.find_color(surf, mode, model)
        perf.end_frame()
    perf.print_summary()

When ``enabled`` is False every method is a cheap no-op, so it is safe to
leave the instrumentation in the hot loop permanently.
"""
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager


def _peak_rss_bytes():
    """Peak resident set size of this process, in bytes (or None).

    Uses resource.getrusage, which is stdlib on Unix. ru_maxrss is bytes on
    macOS but kilobytes on Linux - normalize to bytes.
    """
    try:
        import resource
    except ImportError:  # not available on Windows
        return None
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return maxrss if sys.platform == "darwin" else maxrss * 1024


def _mb(n):
    return "n/a" if n is None else f"{n / (1024 * 1024):.1f} MB"


class PerfTracker:
    """Accumulates per-step timings and memory stats over a run.

    enabled:       master switch; when False all methods no-op.
    track_memory:  also start tracemalloc to report Python-heap current/peak
                   (adds allocation-tracking overhead; process peak RSS is
                   always reported regardless, for free).
    """

    def __init__(self, enabled: bool = True, track_memory: bool = True,
                 label: str = "run"):
        self.enabled = enabled
        self.track_memory = enabled and track_memory
        self.label = label
        self._total = defaultdict(float)   # name -> summed seconds
        self._calls = defaultdict(int)     # name -> call count
        self._order = []                   # names in first-seen order
        self.frames = 0
        self._t0 = time.perf_counter() if enabled else 0.0
        self._rss_start = _peak_rss_bytes() if enabled else None

        if self.track_memory:
            import tracemalloc
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            self._own_tracemalloc = True
        else:
            self._own_tracemalloc = False

    @contextmanager
    def track(self, name: str):
        """Time the wrapped block, accumulating under ``name``."""
        if not self.enabled:
            yield
            return
        if name not in self._calls:
            self._order.append(name)
        start = time.perf_counter()
        try:
            yield
        finally:
            self._total[name] += time.perf_counter() - start
            self._calls[name] += 1

    def end_frame(self):
        """Mark the end of one frame (for per-frame averages)."""
        if self.enabled:
            self.frames += 1

    def summary(self) -> str:
        """Render the timing + memory report as a string."""
        if not self.enabled:
            return "(performance tracking disabled)"

        wall = time.perf_counter() - self._t0
        tracked = sum(self._total.values()) or 1e-12

        title = f" PERF: {self.label} "
        lines = [title.center(64, "=")]
        lines.append(f"frames: {self.frames}    wall: {wall:.2f}s"
                     + (f"    {wall / self.frames * 1e3:.1f} ms/frame"
                        if self.frames else ""))
        lines.append("")
        lines.append(f"{'step':<22}{'calls':>7}{'total_s':>10}"
                     f"{'mean_ms':>10}{'%wall':>8}")
        lines.append("-" * 57)
        for name in sorted(self._order, key=lambda n: -self._total[n]):
            total = self._total[name]
            calls = self._calls[name]
            mean_ms = (total / calls * 1e3) if calls else 0.0
            lines.append(f"{name:<22}{calls:>7}{total:>10.3f}"
                         f"{mean_ms:>10.2f}{total / wall * 100:>7.1f}%")
        untracked = wall - tracked
        lines.append(f"{'(untracked)':<22}{'':>7}{untracked:>10.3f}"
                     f"{'':>10}{untracked / wall * 100:>7.1f}%")

        lines.append("")
        peak = _peak_rss_bytes()
        lines.append(f"process peak RSS: {_mb(peak)}"
                     + (f"  (start {_mb(self._rss_start)})"
                        if self._rss_start else ""))
        if self.track_memory:
            import tracemalloc
            cur, pk = tracemalloc.get_traced_memory()
            lines.append(f"python heap: current {_mb(cur)}, peak {_mb(pk)}")

        lines.append("=" * 64)
        return "\n".join(lines)

    def print_summary(self):
        """Print the report to stdout (and stop tracemalloc if we started it)."""
        if not self.enabled:
            return
        print(self.summary())
        if self._own_tracemalloc:
            import tracemalloc
            tracemalloc.stop()
            self._own_tracemalloc = False


# A shared no-op tracker for code paths that want to call the API
# unconditionally without threading a config flag through.
DISABLED = PerfTracker(enabled=False)


def _self_check():
    """Tiny smoke test; run `python octopus_ai/perf.py`."""
    p = PerfTracker(label="self-check")
    for _ in range(3):
        with p.track("busy"):
            sum(i * i for i in range(100000))
        p.end_frame()
    p.print_summary()


if __name__ == "__main__":
    _self_check()
    sys.stdout.flush()
    os._exit(0)  # skip slow TF atexit teardown for this standalone check
