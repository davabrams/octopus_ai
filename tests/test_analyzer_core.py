"""Node-based tests for the analyzer's plain-JS core (record & replay, Phase 5).

The analyzer is a single self-contained HTML file with no JS build toolchain,
so its pure logic lives in a `<script id="analyzer-core">` block attached to
`window.AnalyzerCore`. These tests regex-extract that block and run assertions
against it via a `node -e` subprocess (skipped if node is missing). Plus a
couple of static checks on the HTML itself.
"""

import os
import re
import shutil
import subprocess
import unittest

HERE = os.path.dirname(__file__)
ANALYZER = os.path.join(HERE, "..", "visualizer", "analyzer.html")

_NODE = shutil.which("node")


def _core_block():
    with open(ANALYZER, encoding="utf-8") as f:
        html = f.read()
    m = re.search(r'<script id="analyzer-core">(.*?)</script>', html, re.S)
    assert m, "analyzer-core script block not found"
    return m.group(1)


def _run_node(assertions_js):
    """Run the core block + assertions under node; raise on non-zero exit."""
    src = _core_block() + "\nconst A = globalThis.AnalyzerCore;\n" + assertions_js
    proc = subprocess.run([_NODE, "-e", src], capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(
            f"node assertions failed:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
        )
    return proc.stdout


@unittest.skipIf(_NODE is None, "node not installed")
class TestCoreLogic(unittest.TestCase):
    def test_to255_and_astriple(self):
        _run_node("""
          const assert = require('assert');
          assert.strictEqual(A.to255(0), 0);
          assert.strictEqual(A.to255(1), 255);
          assert.strictEqual(A.to255(1.5), 255);   // clamps
          assert.strictEqual(A.to255(-1), 0);
          assert.deepStrictEqual(A.asTriple([0.1,0.2,0.3]), [0.1,0.2,0.3]);
          assert.deepStrictEqual(A.asTriple(0.5), [0.5,0.5,0.5]);  // scalar
        """)

    def test_lru_eviction(self):
        _run_node("""
          const assert = require('assert');
          const lru = A.makeLru(2);
          lru.set('a',1); lru.set('b',2);
          lru.get('a');            // refresh a
          lru.set('c',3);          // evicts b (LRU)
          assert.ok(lru.has('a'));
          assert.ok(!lru.has('b'));
          assert.ok(lru.has('c'));
          assert.strictEqual(lru.size, 2);
        """)

    def test_prefetch_plan_windows(self):
        _run_node("""
          const assert = require('assert');
          // around 10, ahead 3 behind 1, clamped to [0,12]
          const p = A.prefetchPlan(10, 12, 3, 1, null);
          assert.deepStrictEqual(p.sort((a,b)=>a-b), [9,10,11,12]);
          // nearest-first ordering
          const q = A.prefetchPlan(10, 100, 3, 3, null);
          assert.strictEqual(q[0], 10);
          // clamps at 0
          const r = A.prefetchPlan(0, 100, 2, 5, null);
          assert.ok(Math.min(...r) >= 0);
          // skips already-cached frames
          const s = A.prefetchPlan(5, 100, 2, 2, (f)=>f===5);
          assert.ok(!s.includes(5));
        """)

    def test_chains_with_base(self):
        _run_node("""
          const assert = require('assert');
          // 2 steps, 2 free nodes each; base prepended per step, nodes intact
          const traj = [ [[1,1],[2,2]], [[3,3],[4,4]] ];
          const base = {x:0, y:0};
          const chains = A.chainsWithBase(traj, base);
          assert.strictEqual(chains.length, 2);
          assert.deepStrictEqual(chains[0], [[0,0],[1,1],[2,2]]);
          assert.deepStrictEqual(chains[1], [[0,0],[3,3],[4,4]]);
          // null trajectory -> empty
          assert.deepStrictEqual(A.chainsWithBase(null, base), []);
        """)

    def test_nearest_sucker(self):
        _run_node("""
          const assert = require('assert');
          const suckers = [{x:0,y:0},{x:5,y:5},{x:10,y:0}];
          assert.strictEqual(A.nearestSucker(suckers, 4.6, 4.6), 1);
          assert.strictEqual(A.nearestSucker(suckers, 9.5, 0.2), 2);
          // maxDist filters out far picks
          assert.strictEqual(A.nearestSucker(suckers, 100, 100, 1.0), -1);
        """)

    def test_color_error_stats(self):
        _run_node("""
          const assert = require('assert');
          const st = A.colorErrorStats([0,0,0],[1,0,0],[0,0,0]);
          assert.strictEqual(st.errBefore, 0);
          assert.strictEqual(st.errAfter, 1);   // (1-0)^2
          const st2 = A.colorErrorStats([1,1,1],[1,1,1],[0,0,0]);
          assert.strictEqual(st2.errAfter, 3);  // max, one per channel
        """)

    def test_node_costs_and_directions(self):
        _run_node("""
          const assert = require('assert');
          const near = (a,b)=>assert.ok(Math.abs(a-b)<1e-9, a+" != "+b);
          // Arrows use a y-DOWN convention (canvas): +dy reads as "down".
          assert.strictEqual(A.arrowFor(1,0), "→");   // right
          assert.strictEqual(A.arrowFor(0,1), "↓");   // down
          assert.strictEqual(A.arrowFor(0,-1), "↑");  // up
          assert.strictEqual(A.arrowFor(0,0), "·");   // zero
          assert.strictEqual(A.angleDeg(1,0), 0);
          assert.strictEqual(A.angleDeg(0,1), 90);

          // Straight horizontal chain at rest length -> spring/bending vanish.
          const chain = [[0,0],[1,0],[2,0],[3,0]];
          const cfg = { restLength:1, wSpring:2, wBend:1, wReachRun:0.1,
                        wReachTerminal:6, wExplore:0.5, wEffort:3, wRepel:8,
                        repelRadius:2.5, repelTipFraction:1, target:[5,0],
                        targetKind:"explore", threat:null };
          // Interior node: spring/bending ~0; effort from a downward velocity;
          // whole-arm attraction now reaches this node too (running weight,
          // normalized by n_free=3), pointing at the target.
          const c = A.nodeCosts(chain, 1, cfg, false, [0,0.5]);
          near(c.spring.cost, 0); near(c.bending.cost, 0);
          near(c.effort.cost, 0.75);                       // wEffort*|v|^2 = 3*0.25
          near(c.explore.cost, (0.1/3)*16);                // (wReachRun/nFree)*d^2
          assert.strictEqual(A.arrowFor(c.explore.dir[0], c.explore.dir[1]), "→");

          // Tip at terminal: EXPLORE attraction uses w_explore (0.5), not
          // w_reach_terminal (6) - the weight-label fix - normalized by n_free,
          // and points at the target.
          const t = A.nodeCosts(chain, 3, cfg, true, null);
          near(t.explore.cost, (0.5/3)*4);                 // (wExplore/nFree)*d^2
          assert.strictEqual(A.arrowFor(t.explore.dir[0], t.explore.dir[1]), "→");
          assert.ok(!('effort' in t));                     // terminal has no control

          // Repel points AWAY from a threat just below the tip (ungraded here).
          const cfg2 = Object.assign({}, cfg, { threat:[3,1] });
          const r = A.nodeCosts(chain, 3, cfg2, true, null);
          near(r.repel.cost, 18.0);                        // 8 * 1 * (2.5-1)^2
          assert.strictEqual(A.arrowFor(r.repel.dir[0], r.repel.dir[1]), "↑");

          // Repel shows even when the node is BEYOND the keep-out radius: value
          // 0, so it's visibly present rather than dropped ("every cost shows").
          const cfgFar = Object.assign({}, cfg, { threat:[100,100] });
          const rf = A.nodeCosts(chain, 2, cfgFar, false, [0,0]);
          assert.ok('repel' in rf && rf.repel.cost === 0);

          // Graded repel: at EQUAL range the body-adjacent node avoids harder
          // than the tip (repelTipFraction=0.3). threat [2,1] is equidistant
          // from node 1 and node 3.
          const cfgG = Object.assign({}, cfg,
            { threat:[2,1], repelTipFraction:0.3 });
          const rBody = A.nodeCosts(chain, 1, cfgG, true, null).repel;
          const rTip  = A.nodeCosts(chain, 3, cfgG, true, null).repel;
          assert.ok(rBody.cost > rTip.cost);
          near(rTip.cost / rBody.cost, 0.3);

          // Stretched spring pulls the node back toward its previous neighbour.
          const stretched = [[0,0],[1.5,0],[2.5,0],[3.5,0]];
          const s = A.nodeCosts(stretched, 1, cfg, false, null);
          near(s.spring.cost, 0.5);                        // 2 * 0.5^2
          assert.strictEqual(A.arrowFor(s.spring.dir[0], s.spring.dir[1]), "←");
        """)

    def test_playback_advance(self):
        _run_node("""
          const assert = require('assert');
          assert.deepStrictEqual(A.playbackAdvance(3, 10, 1, false), {idx:4, playing:true});
          assert.deepStrictEqual(A.playbackAdvance(10, 10, 1, false), {idx:10, playing:false});
          assert.deepStrictEqual(A.playbackAdvance(10, 10, 1, true), {idx:0, playing:true});
          assert.strictEqual(A.clampIdx(99, 10), 9);
          assert.strictEqual(A.clampIdx(-5, 10), 0);
        """)


class TestStaticChecks(unittest.TestCase):
    def test_half_cell_shift_present(self):
        """The load-bearing entity shift (commit 089a7ff) must survive."""
        with open(ANALYZER, encoding="utf-8") as f:
            html = f.read()
        self.assertIn("translate(cs / 2", html)

    def test_no_deprecated_reactdom_render(self):
        with open(ANALYZER, encoding="utf-8") as f:
            html = f.read()
        self.assertNotIn("ReactDOM.render(", html)
        self.assertIn("createRoot", html)

    def test_no_fabricated_data_generator(self):
        """An analyzer must never fabricate data (old page's Math.random fill)."""
        with open(ANALYZER, encoding="utf-8") as f:
            html = f.read()
        # No random-surface / random-agent generator in the analyzer.
        self.assertNotIn("Math.random() > 0.5 ? 'predator'", html)

    def test_jsx_uses_classic_runtime_not_autorun_babel(self):
        """Regression guard: the JSX must be transformed with the CLASSIC JSX
        runtime and NOT auto-run as text/babel.

        @babel/standalone auto-processes text/babel (and text/jsx) with the
        AUTOMATIC runtime, which injects `import {jsx} from "react/jsx-runtime"`
        — an ES-module import a classic <script> can't run, so the page mounts
        nothing (blank). We keep the source as text/plain and transform it once
        with runtime:"classic"."""
        with open(ANALYZER, encoding="utf-8") as f:
            html = f.read()
        self.assertNotIn('type="text/babel"', html)
        self.assertIn('id="app-src"', html)
        self.assertRegex(html, r'runtime:\s*"classic"')


if __name__ == "__main__":
    unittest.main()
