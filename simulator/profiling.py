"""Lightweight hierarchical span profiler for the simulator.

A domain-level "where did the time go" tree - NOT a function-level profile like
cProfile. You wrap logical phases in ``with span("name"):`` blocks; nesting the
blocks builds a tree, and re-entering the same name (e.g. one "move" per frame)
accumulates its total time + call count. ``PROFILER.render()`` prints the tree
with each node's share of the wall clock, self time, count, and average.

Near-zero overhead when disabled: ``span()`` returns a shared no-op object, so
the instrumented code pays only a flag check + two empty method calls. Enable it
around a run and read the report:

    from simulator.profiling import PROFILER
    with PROFILER.profile():
        runner.run()
    print(PROFILER.render())

Single-threaded ONLY: the span stack is shared mutable state, so never open a
span from inside a ThreadPool worker (e.g. the parallel colour inference). Wrap
the whole parallel call in ONE span on the calling thread instead - its wall
time still shows up, just not broken down per worker.
"""
import time


class _Node:
    __slots__ = ("children", "count", "name", "order", "total")

    def __init__(self, name):
        self.name = name
        self.total = 0.0        # wall seconds accumulated across all entries
        self.count = 0          # number of times this span closed
        self.children = {}      # name -> _Node
        self.order = []         # children in first-seen order

    def child(self, name):
        node = self.children.get(name)
        if node is None:
            node = _Node(name)
            self.children[name] = node
            self.order.append(node)
        return node

    def self_time(self):
        return self.total - sum(c.total for c in self.order)


class _Span:
    """Real timing context: push a node on enter, accumulate on exit."""
    __slots__ = ("name", "node", "prof", "t0")

    def __init__(self, prof, name):
        self.prof = prof
        self.name = name

    def __enter__(self):
        self.node = self.prof._stack[-1].child(self.name)
        self.prof._stack.append(self.node)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.node.total += time.perf_counter() - self.t0
        self.node.count += 1
        self.prof._stack.pop()
        return False


class _Null:
    """No-op context used when profiling is disabled."""
    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NULL = _Null()


class _Session:
    """`with PROFILER.profile():` — reset+enable on enter, disable on exit."""
    __slots__ = ("prof",)

    def __init__(self, prof):
        self.prof = prof

    def __enter__(self):
        self.prof.enable()
        return self.prof

    def __exit__(self, *exc):
        self.prof.disable()
        return False


class Profiler:
    def __init__(self):
        self.enabled = False
        self._root = _Node("total")
        self._stack = [self._root]
        self._wall = 0.0
        self._t0 = None

    def span(self, name):
        """Context manager timing a named span (no-op while disabled)."""
        if not self.enabled:
            return _NULL
        return _Span(self, name)

    def reset(self):
        self._root = _Node("total")
        self._stack = [self._root]
        self._wall = 0.0
        self._t0 = None

    def enable(self):
        self.reset()
        self.enabled = True
        self._t0 = time.perf_counter()

    def disable(self):
        if self._t0 is not None:
            self._wall = time.perf_counter() - self._t0
        self.enabled = False

    def profile(self):
        return _Session(self)

    def render(self, min_pct=0.1):
        """A formatted tree, children sorted by total time (desc). Nodes under
        ``min_pct`` of the wall clock are folded into a per-parent "(other)"."""
        top_sum = sum(c.total for c in self._root.order)
        total = self._wall if self._wall > 0 else (top_sum or 1e-12)
        top_calls = max((c.count for c in self._root.order), default=0)
        out = [
            f"where the sim spent its time — {total * 1000:.1f} ms wall, "
            f"{top_calls} top-level spans",
            f"{'%wall':>6}  {'total':>10}  {'self':>10}  {'count':>7}  {'avg':>9}  name",
            "-" * 72,
        ]

        def walk(node, depth):
            kids = sorted(node.order, key=lambda n: -n.total)
            shown_total = 0.0
            for c in kids:
                pct = 100.0 * c.total / total
                if pct < min_pct:
                    continue
                shown_total += c.total
                avg = c.total / c.count if c.count else 0.0
                out.append(
                    f"{pct:5.1f}%  {c.total * 1000:8.2f}ms  "
                    f"{c.self_time() * 1000:8.2f}ms  {c.count:7d}  "
                    f"{avg * 1000:7.3f}ms  {'  ' * depth}{c.name}"
                )
                walk(c, depth + 1)
            # Time in this node not attributed to any (shown) child.
            hidden = node.total - shown_total if node is not self._root else total - shown_total
            if depth == 0 and hidden > 0:
                out.append(
                    f"{100.0 * hidden / total:5.1f}%  {hidden * 1000:8.2f}ms  "
                    f"{'':>8}    {'':>7}  {'':>9}  (unprofiled)"
                )

        walk(self._root, 0)
        return "\n".join(out)


# Module singleton + convenience alias so call sites can do `from simulator.
# profiling import span; with span("x"): ...`.
PROFILER = Profiler()


def span(name):
    return PROFILER.span(name)
