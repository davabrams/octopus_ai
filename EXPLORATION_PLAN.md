# Exploration Behavior — Design / To-Do

> **STATUS: implemented (per-arm, sucker-seeking variant).** The build follows
> this plan's memory + reward-hierarchy design, with one deliberate change the
> user directed: exploration is **per-arm, not body-level**. Areas are marked
> explored by the **suckers** (`Octopus.visit_counts`, incremented at each
> sucker's cell each frame, `_mark_explored`). When an arm senses no prey, its
> **tip reaches the least-explored cell within that arm's reach**
> (`Limb._ilqr_explore_target`) at a gentle weight `w_explore` (a per-solve
> terminal-reach weight in the arm's params tensor — one compiled controller,
> no retrace). Arms coordinate implicitly through the shared map. Prey preempts
> exploration; the threat repel is unchanged and always dominates. Config:
> `octo_ilqr_explore_enabled` (default off), `octo_ilqr_w_explore` (0.5),
> `octo_ilqr_explore_decay` (1.0). Tests: `tests/test_exploration.py`. The
> §5 "single body-level target" below is superseded by the per-arm approach.


Give the octopus a drive to **explore its environment**: track how much each grid
cell has been visited and steer toward the cell (or region of cells) it has
explored the least. Exploration is a **weak, lowest-priority** drive — it must
never trade away catching prey or avoiding threats.

This is a design document, not an implementation. It is written against the
current iLQR motor stack (`simulator/octopus_generator.py` +
`simulator/ilqr/`), the frozen-dataclass config (`octopus_ai/config_schema.py`),
and the record & replay pipeline (`RECORD_REPLAY_PLAN.md`). Line/symbol
references are as of the record & replay landing.

---

## 1. Requirements (verbatim intent)

1. The octopus **explores** the world rather than sitting idle when nothing is
   nearby.
2. It **keeps a per-cell memory** of how much each cell has been explored.
3. It **seeks the least-explored cell or group of cells.**
4. The **reward for exploration is much smaller** than the reward for catching
   prey or avoiding threats. Exploration must lose every contest with those.

### Non-goals

- No global path planning / A*. The octopus is a reactive controller; exploration
  is one more soft pull on the same per-frame iLQR solve, not a planner.
- No new movement mode. This rides on `MovementMode.ILQR`.
- No persistence of the exploration map across runs (it is per-run memory).

---

## 2. Where this fits in the current motor model

Today, in `MovementMode.ILQR`:

- Each `Limb._move_ilqr` (`octopus_generator.py`) plans a short trajectory with
  its own compiled `ArmController`. It picks a **reach target** via
  `_ilqr_target(tip, agents)` — the nearest prey within `agent_range_radius`
  (`octopus.sensing_radius`), or `None` (then it *holds* at its own tip). A
  threat within `repel_radius` adds a repulsion term.
- The **body has no target of its own**: `Octopus._drift_body_by_tension` moves
  the head by the summed base tension of the arms. An arm straining toward a
  far prey stretches its base segment, and that tension drags the body after it.
  So *the octopus pursues by reaching* — the body is a passive mass integrating
  arm tension.

**Consequence for exploration:** the exploration drive must express itself as a
**reach target for the arms** (like prey), so the existing body-follows-tension
dynamics carry the octopus toward under-explored space with no new body code.
The one difference is the *weight* and the *priority*: exploration is a much
weaker pull, chosen only when there is no prey to chase.

Key files/symbols this touches:

| File | What changes |
|---|---|
| `octopus_generator.py` | `Octopus`: the visit-count memory + target selector; `Limb._move_ilqr` / `_ilqr_target`: accept a body-level explore target and a reach weight |
| `simulator/ilqr/residuals.py` | the explore reach is just `reach_residual` at a low weight — the cost library is where the term lives |
| `simulator/ilqr/arm.py` | `ArmController.solve`: reach weight becomes a **per-solve param** (params tensor), so one compiled controller serves prey-strength and explore-strength reaches without retracing |
| `config_schema.py` / `config.py` | `ExplorationConfig` + flat keys + a profile flag |
| record & replay | `target_kind` gains `'explore'`; optional visit-map recording + analyzer heatmap overlay |

---

## 3. The reward hierarchy (the load-bearing requirement)

Exploration must be dominated by prey and threat **two independent ways**, so it
can never win by accident:

1. **Priority gate (selection).** `_ilqr_target` resolves in strict order:
   1. **prey** in sensing range → reach it (`target_kind='prey'`, full weight);
   2. else if exploration enabled → the least-explored target
      (`target_kind='explore'`, *weak* weight);
   3. else → hold at the tip (`target_kind='hold'`).
   Prey therefore always pre-empts exploration; exploration only fills the idle
   gap that today produces the inert "neutral star" pose.

2. **Weight (magnitude).** The explore reach uses a weight
   `w_explore ≪ w_reach_terminal`. Current iLQR weights (`ILQRConfig`):
   `w_reach_terminal = 6.0` (prey), `w_repel = 8.0` (threat). Exploration should
   sit far below both, e.g. **`w_explore ≈ 0.5`** (an explore pull ~1/12 of a
   prey pull, ~1/16 of a threat push). Even in the degenerate case where the
   gate were bypassed, the solver would still favor prey/threat by a large
   margin.

3. **Threat is never traded off.** Threat avoidance is a *separate additive*
   repulsion residual in the arm cost, not a competing target. It is present
   regardless of whether the arm is reaching for prey or exploring, so
   "avoiding threats" is structurally independent of (and always on top of) the
   exploration pull. An exploring octopus that wanders near a threat still
   recoils.

**Invariant to test:** with a prey in range, `target_kind` is never `'explore'`;
with an active threat, the arm's net motion is away from the threat regardless of
the explore target.

---

## 4. Exploration memory (per-cell tracking)

Add to `Octopus.__init__` (a mutable array — initialize in `__init__`, never as a
class attribute, per the July-2026 shared-state gotcha):

```python
self.visit_counts = np.zeros((cfg.world.y_len, cfg.world.x_len), dtype=np.float32)
```

Indexed `[y][x]` to match `RandomSurface.grid`.

**Update, once per frame** (after the body + arms move, before/at the recorder
seam so it is captured): mark the octopus's **physical footprint** as visited.
Two candidate footprints (config choice `ExplorationConfig.footprint`):

- `SUCKERS` (recommended): increment the cell under each sucker
  (`round(s.x), round(s.y)`, clamped) — ties "explored" to the octopus's actual
  reach/coverage, and the 256 suckers give a soft brush.
- `HEAD`: increment only the head cell (+ optional radius) — cheaper, coarser.

Optional **recency decay** (`ExplorationConfig.decay`, default `1.0` = off):
`visit_counts *= decay` each frame before the increment. `decay < 1` turns "least
*visited*" into "least *recently* visited," which prevents the octopus getting
permanently stuck avoiding a corner it fully explored once. Start with decay off;
it is a one-line knob if wandering stalls.

Cost: one `(y_len, x_len)` float array (900 floats at 30×30) + ~256 integer
increments/frame. Negligible.

---

## 5. Target selection (least-explored cell / region)

Computed **once per frame on the `Octopus`** (a single body-level target shared
by all arms, mirroring how the body pursues one prey — coherent directed motion
rather than eight arms wandering independently), and passed down to the arms.

Algorithm (`Octopus._exploration_target() -> (x, y) | None`):

1. **Region, not pixel (requirement 3's "group of cells").** Optionally blur the
   count map with a Gaussian (`ExplorationConfig.blur_sigma`, default ~1.5) so
   the argmin lands on the center of an under-explored *region*, not a single
   noisy cell:
   `M = gaussian_filter(visit_counts, blur_sigma)` (SciPy is not a dep; a small
   separable box blur or a NumPy convolution suffices — spec it as "a cheap
   blur," no new dependency).

2. **Locality bias.** Score each cell by exploration need *plus* a distance
   penalty so the octopus heads for the *nearest* under-explored region rather
   than teleport-seeking across the map:
   ```
   score[y,x] = M[y,x] + locality_weight * dist((x,y), (octo.x, octo.y))
   target_cell = argmin(score)          # cell center in world coords
   ```
   `locality_weight` (`ExplorationConfig.locality_weight`, default ~0.3) trades
   "go somewhere genuinely unexplored" against "don't cross the whole world."

3. **Hysteresis / re-target.** Recomputing argmin every frame dithers. Keep the
   current explore target until either the octopus arrives within
   `arrive_radius` of it **or** `retarget_interval` frames pass — then pick a new
   one. Store `self._explore_target` + `self._explore_age` on the `Octopus`.

4. **Fully-explored / tie fallback.** If all cells are ~equal (early frames, or a
   fully-swept map with decay off), argmin is arbitrary-but-stable; that is fine.
   Optionally break ties toward the current heading so motion stays smooth.

Returns world coords of the chosen cell center, or `None` if exploration is
disabled.

---

## 6. Feeding the target into the solve (no retrace)

The arm's `params` tensor (`arm.py`) currently carries
`[base_xy, target_xy, threat_xy, threat_w]`. Extend it with a **reach weight
slot**: `[base_xy, target_xy, threat_xy, threat_w, reach_w]`. The terminal/running
reach residual multiplies the tip error by `sqrt(reach_w)` (like the existing
`threat_w` gate for repulsion), so:

- **prey** → `reach_w = w_reach_terminal` (strong);
- **explore** → `reach_w = w_explore` (weak);
- **hold** → target = tip, `reach_w` irrelevant.

Because `reach_w` rides in the params tensor (not a Python/`@tf.function`
signature change), the **one compiled `ArmController` serves all three modes with
zero retrace** — the same design rule that keeps `record_history` free
(RECORD_REPLAY_PLAN D5).

`Limb._move_ilqr` gets the body-level explore target + weight via the existing,
currently-unused `coordinated_influence` parameter of `Limb.move` (the natural
hook — the `Octopus` computes one exploration target and hands it to every arm).
`_ilqr_target` becomes:

```
prey = nearest prey within range
if prey is not None:      return prey,           w_reach_terminal, 'prey'
if explore_target:        return explore_target, w_explore,        'explore'
return (tip.x, tip.y),    w_reach_terminal(unused), 'hold'
```

---

## 7. Configuration

New `ExplorationConfig` (frozen dataclass) nested under `OctopusConfig` (sibling
of `limb`/`sucker`), plus flat keys and a profile flag:

```python
@dataclass(frozen=True)
class ExplorationConfig:
    enabled: bool = False          # off by default (fresh checkout unchanged)
    w_explore: float = 0.5         # reach weight; MUST stay << w_reach_terminal (6.0)
    footprint: Footprint = SUCKERS # what counts as "visited" each frame
    decay: float = 1.0             # <1 => least-recently-visited (recency)
    blur_sigma: float = 1.5        # region vs single-cell seeking
    locality_weight: float = 0.3   # nearest-under-explored bias
    retarget_interval: int = 15    # frames before re-choosing a target
    arrive_radius: float = 1.5     # "reached" threshold (cells)
```

- Flat keys (`config_to_flat`/`config_from_flat`): `explore_enabled`,
  `explore_w`, `explore_decay`, `explore_blur_sigma`, `explore_locality_weight`,
  `explore_retarget_interval`, `explore_arrive_radius` (+ bump
  `EXPECTED_FLAT_KEYS`). `footprint` enum serializes by `.name`.
- **Guardrail:** a config-validation test asserts `w_explore < w_reach_terminal`
  and `w_explore < w_repel`, encoding requirement 4 so a future tuner can't
  silently invert the hierarchy.
- Enable it in a profile (e.g. an `EXPLORE` profile derived from `VIZ_ILQR`, and
  optionally on in `RECORD` so recorded runs show it).

---

## 8. Record & replay integration

- `target_kind` gains **`'explore'`** across `Limb.last_ilqr_meta`,
  `sim_recorder.limb_solves.target_kind`, and the `run_store`/analyzer wire — one
  vocabulary everywhere (RECORD_REPLAY_PLAN D17). The analyzer's iLQR strip then
  labels frames where the octopus is exploring vs hunting.
- **Optional: record the visit map.** A `visit_map` blob per run (final state,
  or every N frames) lets the analyzer draw an **exploration heatmap overlay**
  (a `C` color-mode alongside after/before/target/error). Per-frame is heavy
  (900 floats × 121 ≈ 0.4 MB — cheap actually); a final-state snapshot is enough
  for a coverage picture. Decide during Phase 2 below.
- No schema break is forced: `visit_map` is additive; `'explore'` is just a new
  string value in the existing `target_kind` column.

---

## 9. Phases (to-do)

- [ ] **P1 — Config.** Add `ExplorationConfig` + `Footprint` enum, flat keys,
      `EXPECTED_FLAT_KEYS` bump, the `w_explore < w_reach_terminal/w_repel`
      guardrail test, and an `EXPLORE` profile. All off by default.
- [ ] **P2 — Memory + selector (pure, unit-testable).**
      `Octopus.visit_counts`, the per-frame footprint update, and
      `Octopus._exploration_target()` (blur + locality + hysteresis). Tests:
      least-visited region is chosen; locality bias picks the nearer of two
      equally-unexplored regions; visited cells increment; decay ages counts;
      disabled ⇒ `None` and zero overhead.
- [ ] **P3 — Motor wiring.** `reach_w` slot in the arm params tensor;
      `_ilqr_target` priority gate returning `(target, weight, kind)`;
      `Limb.move`/`_move_ilqr` consume the body-level explore target via
      `coordinated_influence`. Tests: **prey in range ⇒ never `'explore'`**;
      **active threat ⇒ arm still recoils while exploring**; no-agents idle
      octopus **increases cell coverage over N frames** (the behavioral proof);
      solver tracing count unchanged (no retrace).
- [ ] **P4 — Record & replay.** `'explore'` `target_kind` through recorder →
      run_store → analyzer; optional `visit_map` recording + a heatmap overlay
      color-mode in `analyzer.html`. Update the v2 protocol docstring.
- [ ] **P5 — Docs.** `ARCHITECTURE.md` §4.5 (exploration as the idle-gap drive,
      and the two-way reward hierarchy), `CLAUDE.md` key concepts + a gotcha
      ("exploration is gated below prey and weighted far below threat — keep it
      that way"), `config` profile table.

**Every phase boundary keeps `python run_tests.py` and `bazel test //...`
green** (the record & replay plan's rule).

---

## 10. Open questions / decisions to make when implementing

- **Footprint granularity:** suckers (soft, physical) vs head (cheap). Start
  `SUCKERS`.
- **Decay default:** off (pure least-visited) vs a mild `0.99` (recency, avoids
  corner-lock). Start off; revisit if the octopus stalls.
- **Shared vs per-arm explore target:** shared body-level target (§5,
  recommended — coherent motion) vs each arm choosing its own least-explored
  direction (spreads arms, muddier body drift). Start shared.
- **Coverage metric for the behavioral test:** fraction of cells with
  `visit_count > 0` after N idle frames should strictly exceed the same metric
  for exploration-disabled (which stays near the octopus's start footprint).
- **Interaction with camouflage:** exploration moves the octopus over varied
  surface, which *raises* transient visibility (more color change needed). This
  is fine and expected; note it so it is not mistaken for a camouflage
  regression in recorded runs.
