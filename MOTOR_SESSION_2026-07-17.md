# Motor-tier session log — 2026-07-17

A working log of a long session reshaping the iLQR motor tier (arm movement,
sensing, body dynamics) and the analyzer. Records **what changed, why, and the
experiments** that drove each decision. Branch: `bazel-improvements`.

Starting point for this log: commit `dbad7b2`
(*node-autonomous per-node attract/repel*). Everything below built on that.

---

## 1. Current design (where we are)

### Node-autonomous sensing (the core model)
Every free centreline **node** senses independently within its window and gets
its own cost terms — there is **no limb-level policy** (`target_kind` is gone).
Attract and flee can act on *different nodes of the same arm at once*.

Per-node drive terms (0 where the node senses nothing):
- **attract → prey**: node pulled to its nearest sensed prey (`w_reach_terminal=6`).
- **attract → explore**: if no prey, node nudged **one rest-length** toward *its
  own* nearest least-visited cell (`w_explore=0.5`). Visit map is whole-body/
  shared; the attraction is per-node so the arm spreads (no ball-up).
- **flee (repel)**: if a threat is in range, node **retracts toward the body
  centre** ("scrunch up") — force points node→body, weight ∝ threat proximity,
  graded body>tip (`repel_tip_fraction=0.3`). NOT push-away-from-threat.

Structure/regularization terms (every free node):
- **spring** (`w_spring=2`) + **spring_stiffen** (`w_spring_stiffen=30`, cubic
  restoring force via a `dev²` residual → quartic cost). Both share a **deadband**
  `spring_slack=0.25`: a segment may deviate ±0.25 from rest for free.
- **bending** (`w_bend=1`) with a **15° deadband** (`bend_deadzone_deg`): a node
  bends up to 15° for free; only the excess curvature magnitude is penalized.
- **effort** (`w_effort=0.5`) + **effort_stiffen** (`w_effort_stiffen=5`, cubic
  force via `|u|·u` residual) — penalize per-frame node velocity; the stiffen
  term is a hard wall against teleportation.

### Body dynamics
- The body drifts by the summed arm base tension and rotates by summed torque
  (`_drift_body_by_tension`), applied **before** the limbs solve (pre-rotate).
- **No rigid carry**: when the body rotates/translates, the base moves but the
  free nodes stay in world coords, so the solve must smooth the arm back —
  paying **effort**. Turning/translating the limb is no longer free. The spring
  deadband is what keeps this stable (absorbs the small rotation-lag strain).

### Agents (`cfg.agents.movement_mode`)
- `RANDOM`: ignore octopus.
- `PURSUIT_FLEE` (RECORD/VIZ_ILQR default): threats pursue the nearest sucker,
  prey flee, inside the sense window, **ignoring camouflage**. Grid-clamped.
- `LUMPED_SPRING`/`SPRING_CHAIN`: same but **visibility-gated** (camouflage hides).

### Analyzer
- Cost panel is **per-node**: attract (toward prey/explore), repel (toward body),
  spring + springStiffen, bending, effort + effortStiffen — each with magnitude
  + a force-direction arrow. Header shows what the node **senses** (prey vs
  explore vs threat). All terms shown even at 0 ("every cost is visible").
- Recorder **schema v2**: `limb_solves` stores per-node attract/repel targets +
  weights as `FLOAT[]`.
- **"outlines"** checkbox (was "sucker outlines") toggles sucker *and* agent
  outlines. **Agent hover tooltip**: kind (THREAT/PREY), live state (PURSUING/
  FLEEING vs wandering-out-of-range, with distance), speed/max, sense radius.
- Exploration overlay: recency ramp (recent = dark orange, old = faded), driven
  by `explore_decay=0.95`.

---

## 2. Chronological changes (this session, on top of `dbad7b2`)

1. **Per-node exploration** — replaced the shared limb explore target with a
   per-node search: each node aims one rest-length toward its own nearest
   least-visited cell. Fixed the "all suckers ball up on one cell" bug.
2. **Super-linear spring** (`spring_stiffen`) — held segment spacing without
   collapse; also surfaced in the analyzer panel (was invisible before).
3. **Super-linear effort** (`effort_stiffen`) — stopped node "teleportation"
   (max per-frame jump 6.6 → ~2 units).
4. **Flee = scrunch toward body** — reframed repel from push-away-from-threat to
   retract-toward-body (matches octopus defensive behaviour).
5. **Deadbands** — spring ±0.25 and bending ±15° free zones (nodes move freely,
   paying only effort, within tolerance).
6. **Agent tooltip + outlines** — analyzer UI.
7. **Removed the rigid arm carry** — rotation (and translation) now cost effort
   instead of the arm being rotated for free after iLQR.

---

## 3. Experiments run (the evidence behind decisions)

Diagnostics live in the session scratchpad; results summarized here.

- **Spring normalization for whole-arm reach (earlier)**: swept the residual
  scale; `/sqrt(n)` (cost = w·mean) reaches robustly at n=5 and n=15; no
  normalization balls the arm up.
- **Spring deadband vs slack size**: `spring_slack=0.25` is ±83% of the 0.3 rest
  length — very floppy; segments settle compressed (mean ~0.77 rest). Tighten
  if too loose.
- **Effort-stiffen weight sweep**: max per-frame jump 3.53 (w=0) → 1.70 (w=5) →
  1.59 (w=20). Diminishing returns; **w=5** chosen. Residual floor ~1.6 comes
  from discrete explore-target flips, not weak damping.
- **Flee-toward-body**: tip→body distance collapses 4.57 → 0.13 with a threat on
  the body (full coil); gentle (2.64 → 2.37) at standoff distance.
- **Rigid-carry / rotation investigation** (the big one):
  - Confirmed the user's diagnosis: body rotation computed *after* iLQR; a rigid
    carry rotated the free nodes for free (no effort) before the next solve.
  - Removing the carry alone → **spurious spin even with no agents** (~0.07/cap
    0.1). The carry was load-bearing for *stability*, not just "following".
  - Viscous angular damping **can't** fix it (constant off-axis torque → constant
    rotation; only a restoring force stops it — inappropriate). Backed it out.
  - Key finding — the **spring deadband breaks the runaway**: with the deadband,
    removing the carry is stable (no-prey ~0.000 for tens of frames); with a
    *tight* spring (slack 0) it spins at 0.10. The deadband absorbs the small
    rotation-lag strain so it makes no feedback torque.
  - Reachable stimulus → **settles** (late |dtheta| ~0.001); fully idle → slow
    drift/eventual spin (accepted — "a resting octopus slowly reorienting").
  - **Pre-existing rotational jitter**: during exploration the body rotates near
    the cap almost every frame (mean |dtheta| ~0.09 of 0.10, ~91% of frames) —
    *identical* with and without the carry, so not a regression. It's the
    explore→torque coupling over-driving rotation. Left as an open item.

---

## 4. Config knobs added this session (`ILQRConfig`, flat `octo_ilqr_*`)

| knob | default | meaning |
|---|---|---|
| `w_explore_threat_avoid` | 10.0 | penalize explore cells near a threat |
| `explore_threat_radius` | 5.0 | radius of that penalty |
| `explore_node_radius` | 3.0 | per-node explore search radius |
| `explore_decay` | 0.95 | recency decay of the visit map |
| `w_effort` | 0.5 | per-node velocity penalty |
| `w_effort_stiffen` | 5.0 | super-linear velocity (anti-teleport) |
| `w_spring_stiffen` | 30.0 | super-linear spring (anti stretch/collapse) |
| `spring_slack` | 0.25 | spring deadband (free zone) |
| `bend_deadzone_deg` | 15.0 | free bend per node |
| `repel_tip_fraction` | 0.3 | tip flees this fraction of the body-adjacent node |

`--explore` flag added to `simulator/headless_runner.py` (RECORD has exploration
off by default).

---

## 5. Open items (next time)

1. **Folded-node local minima**: a node folded ~180° stays folded, paying steady
   bending, because `effort_stiffen` walls off the large move needed to un-fold
   (Gauss-Newton is local). Options: lower `w_effort_stiffen`, raise `w_bend`, or
   **fold detection + straight re-init** when a node's bend exceeds ~120°
   (recommended, robust).
2. **Rotational jitter**: body rotates near the angular cap during exploration
   (pre-existing). Likely fix: lower `body_torque_gain` or damp the
   explore-driven torque.
3. **Spring slack is large** (±83% of rest) — tighten `spring_slack` if the arm
   looks too floppy/bunched.
4. **Prey are hard to catch** (they flee well): consider a shorter sense window
   for prey than threats to make the hunt winnable.
5. **Colour mismatch** (camouflage): the SUCKER model trained on grayscale
   surfaces + a constraint-dominated loss (0.95 constraint / 0.05 MAE) → colours
   lag. Retrain on colour surfaces / raise MAE weight / try `NO_MODEL` to confirm.

## 6. Latest recordings (for the analyzer)

- `20260717_140150_a8ffd4` — no rigid carry (rotation costs effort).
- `20260717_131501_896554` — deadbands (floppy).
- `20260717_124329_46fd4d` — flee = scrunch toward body.
