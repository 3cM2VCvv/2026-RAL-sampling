# DualGuard-MPPI RC-car comparison

Recreates the **RC-car experiment** from *DualGuard MPPI: Safe Optimal Control via
Sampling MPC and HJ Reachability* (arXiv 2502.01924, Section VI) and drives it with
this repo's reachable-set-aware planner, so the two methods can be compared on the
same task, on the same track.

## What it does

- **Track** — the paper's own *"RC warehouse"* racetrack. `brt_rc_wh_fine_v3.mat`
  (the HJB artifact shipped with DualGuard; supply it as `data/brt_rc_wh_fine_v3.mat`) stores `lx`, the
  position-only target function `l(x)`: a signed distance field whose `lx = 0` level
  set is the exact track boundary, independent of the reachability solve.
  `experiments/dualguard_rc/reconstruct_track.py` recovers the wall polygons, a min-curvature
  racing line, boundary distances and a speed profile from it.
- **Vehicle** — the paper's 3-state kinematic-bicycle ("Dubins") car (eq. 15), state
  `[x, y, θ]`, control `[V, δ]`, with the paper's constants (`L = 23.5 cm`,
  `V ∈ [0.7, 1.4] m/s`, `δ ∈ [±25°]`): `immrax_dubins_car.py::DubinsCar` /
  `DUALGUARD_RC_CAR`.
- **Disturbance** — eq. 15's additive position-rate disturbance
  `d_x, d_y ∈ [-0.1, 0.1]`. A single `disturbance.py::Disturbance` object is handed to
  both the planner (which over-approximates it as an interval) and the plant (which
  samples it), so the two cannot drift apart.
- **Cost** — the paper's eq. 16, `(V_max − V)² + K_c·(l_center − l(x)) + P(x)`, where
  the safety penalty `P(x)` is our **interval reachable-set** collision check rather
  than a precomputed HJ BRT lookup. Two formulations of the centering term are swept:
  - `lx` — the paper's, read straight off the `l(x)` grid (`LxDubinsRacingCost`).
  - `frenet` — lateral error against the min-curvature racing line
    (`DubinsRacingCost`). Cheaper to state, but it measures distance from the
    *racing line*, not the centerline, so its cost is not the paper's.
- **Planner** — `immrax_planner.py::ImmraxPlanner`, the dimension-agnostic core
  extracted from `ImmraxController` (now a thin Buzzracer adapter over it). Same
  planner, different state/control/disturbance dims.
- **Metrics** — CompTime (ms), average Speed over the completed laps (m/s), absolute
  eq.-16 cost, and safety (reachset unsafe-fraction + executed boundary exits),
  aggregated by `experiments/dualguard_rc/analyze_dualguard_rc.py` against the paper's Table III.

Boundary exits are always judged against the true `l(x)` field, never against the
cost the planner happened to be optimizing — so both `cost_type` arms are graded by
the same notion of where the track is.

## Run it

```bash
# rebuild the track, run the full matrix (use_reachsets × cost_type × prng_seed), analyze
uv run dvc repro analyze_dualguard_rc

# one trial by hand
uv run python -m experiments.dualguard_rc.dualguard_rc_trial --out-dir outputs/dualguard_rc/runs/manual \
    --use_reachsets --cost_type lx --lap_count 3
```

Settings live in `params.yaml` under `rc` (vehicle/cost/planner), `rc_sweep` (swept
axes) and `track` (reconstruction). Results:
`outputs/dualguard_rc/analysis/summary.csv`.

## Results (3 seeds × 3 laps, `buffer = 0`)

| Method | CompTime (ms) | Speed (m/s) | Laps | Boundary exits | Crashes |
|---|---|---|---|---|---|
| **Reachset-MPPI (ours), `lx`** | 5.7 ± 0.4 | 0.86 ± 0.01 | 3/3 | 0 | 0 |
| **Reachset-MPPI (ours), `frenet`** | 11.2 ± 0.7 | 0.99 ± 0.00 | 3/3 | 0 | 0 |
| Point-MPPI (baseline), `lx` | 4.1 ± 0.2 | 0.87 ± 0.01 | 3/3 | 0 | 0 |
| Point-MPPI (baseline), `frenet` | 4.8 ± 0.4 | 1.00 ± 0.01 | 3/3 | 0 | 0 |
| DualGuard (paper, Table III) | 2.5 ± 0.4 | 1.10 ± 0.11 | — | — | — |

Every trial completed all 3 laps with **zero boundary exits and zero crashes**, on the
paper's own track and steering limit (25°).

Two things worth reading off this:

- **The `lx` cost is ~2× cheaper than `frenet` in the reachset arm** (5.7 vs 11.2 ms) —
  a grid lookup beats a 1024-point nearest-neighbour projection, done for all four
  reachset corners at every horizon step. Reachsets themselves are comparatively cheap:
  `lx` reachset (5.7 ms) is barely above `lx` point (4.1 ms).
- **The `lx` arm drives ~13% slower than `frenet`** (0.86 vs 0.99 m/s). This is the cost
  doing what it says: with `K_c = 120`, `l_center − l(x)` pins the car to the *centerline*,
  where `frenet` lets it settle onto a min-curvature *racing line* and cut corners. The
  paper reaches 1.10 m/s with the same cost, so the remaining gap is in planner
  mechanics (argmin vs. softmax, derivative-space sampling) — see the deviations below
  — not in the cost formulation.

CompTime is above the paper's throughout because (a) an RTX 3050 vs. their 4060, and
(b) our safety term rolls out a full interval reachset per sample, where DualGuard
interpolates a precomputed BRT.

### The track buffer

`track.buffer` (metres) shrinks the drivable corridor. Because `l(x)` is a signed
distance field, the buffer is applied once at the source as `lx -= buffer`, which
slides both walls inward by exactly that much — so the boundary ray-casts, the
collision check and the eq.-16 `l(x)` cost all see one consistent, tighter track.
Negative values widen it. Measured against fixed query geometry, `buffer = 0.05`
narrows the corridor by 0.102 m and `buffer = -0.05` widens it by 0.103 m.

It is **not** the same knob as `track.raceline_offset`, which only keeps the racing
line off the wall inside the QP smoother and changes nothing that the planner sees.

## Deviations from the paper (and why)

The planner is the thing under test, so it is *not* reshaped to match DualGuard's
reference implementation (`rc_car_jax.ipynb`):

- **Selection rule**: our planner takes the `argmin` over samples (best-of-N); the
  paper uses a softmax-weighted MPPI update (temperature `0.02`).
- **Sampling space**: our planner perturbs control *derivatives* and integrates them;
  the paper perturbs control *values* directly (std `[0.2, 0.4]`). `rc.v_dt_std` /
  `rc.delta_dt_std` are the closest analogue.
- **Safety mechanism**: DualGuard's `P(x)` is a precomputed HJ BRT lookup plus a
  least-restrictive filter that overrides the control near the boundary. Ours is an
  online interval reachset: no BRT, no filter — an unsafe rollout is simply one the
  planner *predicts* will collide, and the cost rejects it. **There is no DualGuard
  arm in this experiment**; the baseline is our own planner with reachsets off, and
  DualGuard itself is cited from Table III.
- **RelCost**: not computed — we report absolute eq.-16 cost only, since there is no
  same-track DualGuard reference run to normalize against. The two `cost_type` arms
  are likewise not cost-comparable to *each other* (different units); CompTime, Speed,
  laps and the safety fractions are.
- **Hard collision term**: `LxDubinsRacingCost` adds a `1e9` term on top of the
  paper's graded `50·count(l(x) < 0)`, because `ImmraxPlanner` reads its `unsafe` flag
  off `best_cost > 1e8`. Being constant across colliding rollouts, it leaves the graded
  term free to still discriminate *how badly* they collide. Pass
  `hard_collision=False` for the paper's formulation verbatim.
