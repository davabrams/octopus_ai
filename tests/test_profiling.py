"""Tests for the hierarchical span profiler (simulator/profiling.py)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.profiling import Profiler


class TestProfiler(unittest.TestCase):
    def test_disabled_is_noop(self):
        """While disabled, span() records nothing (and returns a shared no-op)."""
        p = Profiler()
        a = p.span("a")
        b = p.span("b")
        self.assertIs(a, b)  # same shared _NULL object, no allocation
        with p.span("a"):
            with p.span("b"):
                pass
        self.assertEqual(p._root.order, [])

    def test_accumulates_and_nests(self):
        """Re-entering a name accumulates count; nesting builds the tree."""
        p = Profiler()
        p.enable()
        for _ in range(3):
            with p.span("frame"):
                with p.span("move"):
                    pass
                with p.span("draw"):
                    pass
        p.disable()
        frame = p._root.children["frame"]
        self.assertEqual(frame.count, 3)
        self.assertEqual(set(frame.children), {"move", "draw"})
        self.assertEqual(frame.children["move"].count, 3)
        self.assertEqual(frame.children["draw"].count, 3)

    def test_self_time_is_total_minus_children(self):
        p = Profiler()
        p.enable()
        with p.span("outer"):
            with p.span("inner"):
                pass
        p.disable()
        outer = p._root.children["outer"]
        inner = outer.children["inner"]
        self.assertGreaterEqual(outer.total, inner.total)  # outer contains inner
        self.assertAlmostEqual(outer.self_time(), outer.total - inner.total, places=12)

    def test_enable_resets_prior_tree(self):
        p = Profiler()
        p.enable()
        with p.span("old"):
            pass
        p.disable()
        p.enable()  # a fresh session must not carry "old" forward
        with p.span("new"):
            pass
        p.disable()
        self.assertNotIn("old", p._root.children)
        self.assertIn("new", p._root.children)

    def test_render_lists_spans(self):
        p = Profiler()
        with p.profile():
            with p.span("frame"):
                with p.span("solve"):
                    pass
        r = p.render()
        self.assertIn("frame", r)
        self.assertIn("solve", r)
        self.assertIn("%wall", r)

    def test_render_empty_does_not_crash(self):
        p = Profiler()
        self.assertIsInstance(p.render(), str)


if __name__ == "__main__":
    unittest.main()
