# Body Rotation + Base Ring — Design / To-Do

Stop the eight independent iLQR limbs from collapsing onto each other, **without
coupling the solvers**. Fix the geometry instead: root each limb at its own point
on a ring around the body, and give the body an **orientation** that rotates from
the arms' net torque so the whole fan can turn (the "top" limb isn't always on
top).

Design document, not an implementation. Written against the current iLQR motor
stack (`simulator/octopus_generator.py` + `simulator/ilqr/`) and the record &
replay pipeline. Symbol/line references are as of commit `386ee23`.

---

## 1. Requirements (verbatim intent)

1. Limbs must **remain independent** — each keeps solving its own iLQR; no
   limb-to-limb cost coupling.
2. Add a mechanism so limbs **don't overlap / converge to the same location.**
3. Each limb's **first nodes stay equally distributed around a circle** around
   the body (they must not collocate).
4. The **body may rotate** to let the fan of limbs shift — so a given limb isn't
   permanently stuck facing one direction.
5. Alongside the existing **linear** vector that drifts the body's *position*,
   add an **angular** vector that drives the body's *orientation*.

### Non-goals

- No pairwise inter-limb repulsion cost (rejected: it couples the independent
  solvers and is O(N²·nodes); the base ring + rotation solves the collapse
  geometrically). Noted as a possible future addition if arms still cross far
  from the body.
- No new movement mode; this rides on `MovementMode.ILQR`.

---

## 2. Root cause (why they collapse today)

`Limb._move_ilqr` pins **every** limb's base (centerline node 0) to the *exact
same point* — the body center:

```python
base = self.center_line[0]
base.x = x_octo          # body center, identical for all 8 limbs
base.y = y_octo
```

So all arms share one root. Nothing angular holds them apart, and when several
arms reach the same prey the terminal-reach cost (`w_reach_terminal = 6`) drags
all their nodes toward that one point — the fan collapses into a single arm (see
the reported image). The body also has **no orientation**: `Octopus` stores only
`self.x, self.y`, and `_drift_body_by_tension` only ever integrates a *linear*
sum of tensions. Each limb's angle is fixed relative to a non-rotating body
(`init_angle = 2π·i/N`), so limb *i* always points the same way.

Two independent facts to fix: **(a) one shared base point**, and **(b) no body
orientation**. They are coupled — a base ring is what gives arm tension a
*tangential* component, which is what makes body rotation well-defined (torque is
identically zero when every base sits on the center).

---

## 3. Design

### 3.1 Body orientation state `θ`

`Octopus` gains `self.theta: float` (radians, wrapped `[0, 2π)`), initialized to
`0.0`. This is the "angular position" the body integrates, the rotational twin of
`(self.x, self.y)`.

### 3.2 Base ring (keeps limbs independent AND separated) — requirement 1–3

Each limb owns a **fixed angular offset** `φ_i = init_angle = 2π·i/N` (already
computed at construction; just store it: `Limb.base_angle`). Instead of pinning
node 0 at the center, pin it at that limb's point on a ring of radius
`ring_radius = R` around the body, rotated by the body's orientation:

```
ψ_i  = θ_body + φ_i
base_i = (x_octo + R·cos ψ_i,  y_octo + R·sin ψ_i)
```

- Arms fan out from **distinct roots** → they can't all collapse to one point,
  and their first nodes are exactly "equally distributed around the unit circle"
  (requirement 3), on a ring that rotates with the body.
- Each limb **still solves its own iLQR** from its own base, seeing its own
  prey/threat (requirement 1). The only shared state is the body pose
  `(x, y, θ)`; there is no limb-to-limb term.
- `_move_ilqr` change: compute `base_i` from `(x_octo, y_octo, θ, R, φ_i)` and
  set `center_line[0]` to it (today it sets the center). Everything downstream
  (`x0`, the base→node0 spring, the tip reach) is already expressed relative to
  `base_xy`, so no cost math changes — only where the base sits.

### 3.3 Angular dynamics (requirement 4–5)

Extend `Octopus._drift_body_by_tension` — today it sums each arm's base reaction
`F_i` (`limb.last_tension`) into a linear `total` and eases the body along it,
capped at `max_body_velocity`. Add the **torque** each arm applies at its ring
point:

```
r_i = base_i - (x_octo, y_octo) = R·(cos ψ_i, sin ψ_i)      # moment arm
τ_i = r_i × F_i = r_i.x·F_i.y - r_i.y·F_i.x                 # 2D cross (scalar)
τ_net = Σ τ_i
Δθ  = clamp(body_torque_gain · τ_net,  ±max_body_angular_velocity)
θ  = (θ + Δθ) mod 2π
```

- The **linear** vector still drives position (unchanged); the **angular** scalar
  `τ_net` drives orientation — the two "vectors" the requirement asks for, both
  emergent from the same per-limb base reactions, no central negotiation.
- Because bases sit off-center (R > 0), an arm straining sideways now has a
  tangential component → nonzero torque → the fan rotates so that arm swings
  toward its target. As `θ` turns, all base points rotate, so the top limb
  stops being permanently on top (requirement 4).
- Same **one-step lag** as the existing linear path (arms report `last_tension`,
  the body integrates it next frame, arms re-pin and re-solve, converges over the
  loop). No new solver, no retrace.
- Store `self.last_body_torque` / `self.last_body_dtheta` for the record layer
  and force introspection (mirrors `last_body_force`/`last_body_drift`).

### 3.4 What deliberately does NOT change

- The `ArmController` / solver / `residuals.py` cost terms — this is a geometry +
  body-dynamics change, not a new cost. (Contrast the exploration plan, which
  *does* add a residual.) Arms remain independent compiled controllers.
- Per-limb targeting: each arm still senses its own nearest prey/threat from its
  tip; as `θ` rotates the tips move, so targeting follows for free.

### Key decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Fix geometry (base ring), not a repulsion cost | Keeps the 8 solvers independent (req 1); O(N) not O(N²); directly yields the equal-angular-distribution (req 3) |
| D2 | Body gains `θ`; orientation integrates net arm **torque** | The "angular position vector" (req 5); lets the fan rotate (req 4) |
| D3 | Base ring is the prerequisite for rotation | Torque is identically zero if all bases sit on the center — the ring gives the moment arm |
| D4 | Reuse the one-step-lag tension loop in `_drift_body_by_tension` | Same convergence pattern that already reconciles 8 controllers into one body trajectory; no new coupling |
| D5 | `ring_radius = 0` reproduces today's behavior exactly | Safe default/rollback: bases collapse to center, `θ` inert. Ship with `R > 0` to fix the bug, but keep the escape hatch |
| D6 | Cost terms / solver untouched | It's a pinning + body-integration change; `arm.py` reads the new base via the existing `base_xy` param, no retrace |

---

## 4. Configuration

New knobs (frozen dataclasses), all with defaults that fix the collapse:

- `OctopusConfig.ring_radius: float = 1.0` — base-ring radius (body anatomy).
  `0.0` ⇒ legacy single-point base.
- `OctopusConfig.max_body_angular_velocity: float = 0.1` — per-frame rotation
  cap (rad), the angular twin of `max_body_velocity`.
- `ILQRConfig.body_torque_gain: float = 3.0` — torque→Δθ gain, alongside the
  existing `body_stiffness` (torque and linear tension share the same
  base-reaction source, so they live together).

Flat keys (`ring_radius`, `octo_max_body_angular_velocity`,
`octo_ilqr_body_torque_gain`), `EXPECTED_FLAT_KEYS` bump, and a validation test
that `ring_radius >= 0`.

---

## 5. Record & replay + analyzer ripple

Mostly free, because the recorder already stores **node 0 of every limb**
(`limb_nodes`), so the fanned-out base ring is captured with no schema change and
the analyzer — which draws limbs straight from node positions — renders the fan
automatically. The only genuine addition is the body's new orientation:

- **`frames.head_theta FLOAT`** (recorder) + `serialize_state` `octopus.head`
  gains `"theta"` + `run_store._build_state` reads it back. Additive; a missing
  column on old runs defaults to 0.
- **Analyzer:** optional — draw a short heading tick from the head at `θ`, and
  (nice-to-have) the faint base-ring circle, so rotation is legible. Limb drawing
  itself is unchanged.
- `head_theta` is technically derivable from the recorded base points, but
  recording it directly is cheaper and unambiguous for a heading indicator.

---

## 6. Phases (to-do)

- [ ] **P1 — Config.** `ring_radius`, `max_body_angular_velocity`,
      `body_torque_gain` + flat keys + `EXPECTED_FLAT_KEYS` bump + the
      `ring_radius >= 0` guard. Defaults fix the collapse; `R = 0` = legacy.
- [ ] **P2 — Base ring geometry.** Store `Limb.base_angle = init_angle`; add
      `Octopus.theta`; in `_move_ilqr` pin `center_line[0]` to the rotated ring
      point. Test: the N base node-0 positions are equally spaced on a circle of
      radius `R` about the body center at angles `θ + φ_i`; `R = 0` ⇒ all at
      center (legacy).
- [ ] **P3 — Angular dynamics.** Extend `_drift_body_by_tension` with `τ_net` →
      clamped Δθ → `θ` update; stash `last_body_torque`/`last_body_dtheta`. Test:
      symmetric tension ⇒ ~0 net rotation; a single sideways-straining arm
      rotates `θ` in the correct sign; Δθ capped at `max_body_angular_velocity`;
      limbs remain independent (no cross-limb references introduced).
- [ ] **P4 — Behavioral proof.** Record a reactive-prey run (the demo config from
      earlier). Assert the **min pairwise distance between limb base points stays
      ≥ ~R** across all frames — where today it is ~0 (full collapse). This is the
      "no overlap" acceptance.
- [ ] **P5 — Record & replay.** `frames.head_theta` through recorder →
      `run_store` → `serialize_state`; analyzer heading tick + optional base-ring
      circle. Update the v2 protocol docstring.
- [ ] **P6 — Docs.** ARCHITECTURE.md §4.2/§4.5 (body pose gains `θ`; base ring;
      linear+angular body integration), CLAUDE.md key concepts + a gotcha ("limb
      bases sit on a rotating ring, not the body center; the body integrates arm
      torque into `θ`").

**Every phase boundary keeps `python run_tests.py` and `bazel test //...`
green.**

---

## 7. Open questions / decisions to make when implementing

- **`ring_radius` default:** big enough to visibly fan the bases and give a
  useful moment arm, small relative to arm length (~4.5 cells at defaults). Start
  `1.0`; tune against the analyzer.
- **Torque sign / units:** `τ = r × F` with the arm's *tensile* base reaction.
  Confirm the sign convention rotates the fan *toward* a straining arm's target
  (a reaching arm should pull the body to face it), and flip the gain sign if
  not. Cover with the P3 sign test.
- **Rotational damping / runaway — RESOLVED.** Naively, rotating `θ` moves each
  base tangentially while the arm warm-starts in world coords; that spurious
  strain feeds back into more torque and pins `Δθ` at the cap forever (a runaway
  spin — observed in the first cut). Fix: **carry the arms rigidly with the
  body's rotation** (`_move_ilqr` rotates the free nodes about the body center
  by `body_dtheta` before warm-starting), so rotation adds no strain and the
  body only turns from genuine off-axis reaching, then settles. Guarded by
  `test_rotation_settles_no_runaway`.
- **Threat avoidance via body drift — ADDRESSED (follow-up to this plan).**
  Originally the body only felt *tensile* arm reactions, so a recoiling arm
  (which compresses its base segment) produced no flee (legacy `ring=0` threat
  response ≈ 0). Now the base segment is **two-sided (rod, not rope) while the
  arm is recoiling from a sensed threat**: a compressed segment then *pushes*
  the body away, so it flees (fixed-threat response ≈ −1.3 vs 0 before). The
  push is **gated on a sensed threat** — with no threat we stay rope-like,
  because idle arms settle a hair compressed *asymmetrically* and an ungated
  two-sided spring jitters the body at max speed. Threats are also **sensed from
  the whole arm** (nearest centerline node), not just the tip, so a threat near
  an arm's middle is avoided. Guarded by `TestThreatResponse`.
- **Base ring for non-iLQR modes:** the collapse is an iLQR artifact, but the
  base ring + `θ` are generally sensible. Scope P2–P3 to `MovementMode.ILQR`
  first; `_drift_body_by_tension` is shared, so LUMPED_SPRING/SPRING_CHAIN could
  adopt the ring later behind the same `ring_radius` knob.
- **Interaction with the exploration plan:** exploration adds a weak reach
  target; with a rotating body the octopus can *turn* toward an unexplored
  region instead of only translating — the two features compose, no conflict.
- **Does node 0 stay pinned (not a decision variable)?** Yes — node 0 remains the
  fixed ring anchor; only nodes 1..tip are solved, exactly as today (just from a
  ring point instead of the center).
