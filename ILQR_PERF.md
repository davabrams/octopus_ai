# iLQR performance — what's been tried (READ before re-optimizing)

The per-limb iLQR solve is the simulator's hotspot. This file records what was
profiled, what was **kept**, what was **tried and reverted**, and what was
**considered and rejected** — so nobody re-runs a dead end. Dates are 2026-07.

## How to profile

```bash
python simulator/headless_runner.py --frames 120 --profile          # compiled backward (default)
python simulator/headless_runner.py --frames 120 --profile --eager-backward   # the slow reference path
```

`simulator/profiling.py` is a hierarchical span profiler; the analyzer also shows
the report after a Simulate (server logs it + ships it in `simulate_complete`).
The tree is `frame → octo.move → limb.move → ilqr.solve → {ilqr.backward,
ilqr.forward, ilqr.rollout}`; the eager solver additionally splits the backward
into `bwd.derivs` (autodiff Jacobians) and `bwd.riccati` (Q assembly + Cholesky +
value propagation). The compiled backward is one graph, so it shows only
`ilqr.backward`.

## Baseline profile

Measured on a headless run (numbers are % of whole-sim wall time):

| phase | eager backward | compiled backward (default) |
|-------|---------------:|----------------------------:|
| `ilqr.solve` | ~84% | ~69% |
| ↳ `ilqr.backward` | ~73% | ~51% |
| ↳↳ `bwd.riccati` | ~36% | (inside the graph) |
| ↳↳ `bwd.derivs` | ~27% | (inside the graph) |
| ↳ `ilqr.forward` | ~7% | ~12% |
| ↳ `ilqr.rollout` | ~3% | ~5% |
| camouflage + record | ~6% + ~5% | ~12% + ~9% |

The matrices are small: `control_dim = 2·n_free = 32` at the default 16-node arm,
horizon 10, ~8–9 iterations/solve. The backward Riccati recursion is **sequential
over the horizon** (each step needs the next step's value function). Arms are
**independent** and MUST stay so — the goal is eventually one process per arm, so
**do NOT batch across arms**.

## KEPT — graph-compiled backward pass (`solver_parallel.py`)

The original hotspot was Python→TF **op-dispatch overhead**: the eager backward is
a Python double loop (per iteration × per horizon step) firing ~15k tiny TF ops at
~1 ms each — almost all launch overhead, not 6×6/32×32 arithmetic.

`make_solver_parallel` moves the WHOLE backward pass into one `@tf.function`
(horizon unrolled at trace). The Cholesky early-break can't be a Python `break`
in a graph, so it returns an `ok` flag checked once per iteration — identical
outer-loop semantics to `make_solver`. It reuses `make_solver`'s kernels verbatim,
so it's numerically equal to the eager path within float32 op-reorder noise
(~1e-3). Opt-in via `octo_ilqr_compiled_backward` (**on by default**);
`ArmController.compiled_backward`; `headless_runner --eager-backward` selects the
reference path.

**Measured** (16 nodes, H=10, ~200 solves, micro-benchmark):
- eager: **175 ms/solve**, backward 31.8 s.
- compiled: **52 ms/solve**, backward 7.1 s — **3.3× overall, 4.5× backward**.
- full sim (20 frames, 8 arms): wall 47 s → 27 s (~1.7×).

## TRIED AND REVERTED — XLA (`jit_compile=True`) on the backward

**Do not re-add `jit_compile=True` to `backward_pass` unless the trade-offs below
have changed.** It was measured and deliberately reverted.

**Measured** (250 solves): compiled non-XLA **49.3 ms/solve** (backward 8.6 s) →
compiled **+XLA 30.9 ms/solve** (backward 4.0 s) = **1.6× overall, 2.1× backward**.
Correct (equivalent to eager within float32 noise), kept autodiff, one-line change.

Reverted anyway because:
1. **Per-controller XLA compile slows the test suite.** XLA compiles the backward
   on first call *per `ArmController` instance*, and tests build many fresh
   octopuses → constant recompiles: `test_body_rotation` 137 s→190 s, `test_ilqr`
   18 s→26 s, `test_exploration` 112 s→141 s. (In the *sim* this is a one-time
   ~6 s startup — 8 arms compile once, reused across all frames — but the dev
   test-loop friction was judged not worth it.)
2. **XLA's float32 fusion tips a marginally-stable test.**
   `test_rotation_settles_with_reachable_stimulus` uses an artificial
   `max_iters=3` (under-converged, so the float path is amplified); XLA pushed the
   settled rotation from ~0.01 to 0.043 (threshold 0.02, cap 0.1). At the **real**
   `max_iters=30` XLA settles to 0.023 vs non-XLA 0.027 — i.e. XLA does NOT hurt
   the real sim's rotation; only that fragile test.

Takeaway that IS still useful: XLA only got 1.6×, not 5×+, so the backward is **not
purely overhead-bound** — a real chunk of the ~51% is genuine 32×32 linalg +
autodiff compute. That informs the levers below.

## CONSIDERED AND REJECTED (for now)

- **Rewrite the backward in C.** Rejected: you'd lose autodiff (hand-derive and
  hand-code every residual's Jacobian, re-deriving on *every* cost-term change in
  `residuals.py` — a maintenance/bug tax on the code we iterate on most), plus a
  C-extension build (Cython/pybind11/CFFI), LAPACK interop for the Cholesky, Bazel
  wiring, and harder debugging. The compute is small (a 32×32 Cholesky is ~11k
  FLOPs), so C's win would be mostly avoiding framework overhead — which cheaper
  options address. Only worth it for a tiny dependency-free kernel you're willing
  to own permanently.
- **Batch the solve across arms.** Rejected by design: arms must stay independent
  (future: one process per arm). Batching would be the biggest single-machine win
  but conflicts with that goal.
- **Parallel-scan / associative LQR** (parallelize the Riccati over the horizon).
  Premature at horizon 10 — the constant factors eat the O(N)→O(log N) win, and
  it's a substantial rewrite. Revisit only if the horizon grows a lot.

## REMAINING LEVERS (future, roughly cheapest → most)

1. **Sub-profile `bwd.derivs` vs `bwd.riccati` in the COMPILED path** (they were
   ~even in the eager path; re-measure). Tells us whether autodiff or the linalg
   dominates before spending effort.
2. **Hand-code the analytic Jacobians in TF/NumPy** (skip `GradientTape`). The
   dynamics is single-integrator (`f_x = I`, `f_u = dt·I` — trivial) and the
   residuals are known, so the Gauss-Newton Hessian can be assembled directly.
   Removes autodiff overhead, stays in TF, no new deps or build. Do this if #1
   shows derivs dominate.
3. **JAX** rewrite of the solver (jit + XLA + *keeps* autodiff, purpose-built for
   small-matrix jitted numerics). Preferred over raw C if it becomes genuinely
   compute-bound. Cost: a TF→JAX dependency swap for the solver tier.

See ARCHITECTURE.md §11 for the compute-placement rationale.
