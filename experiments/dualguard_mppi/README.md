# DualGuard-MPPI

The DualGuard-MPPI safety filter ([arXiv 2502.01924]) run on two vehicle models, imported
from the authors' fork at `../DualGuard-MPPI`. It is the comparison baseline for this
project's reachable-set MPC: both certify safety against a precomputed set, but DualGuard
filters against an HJ backward reachable tube solved offline in MATLAB, where
`ImmraxController` propagates mixed-monotone reachable sets online.

[arXiv 2502.01924]: https://arxiv.org/abs/2502.01924

Two guards, both active:

| | |
|---|---|
| guard 1 | least-restrictive filter inside **every rollout step** |
| guard 2 | least-restrictive filter on the **executed control** |

## Experiments

| | model | track | BRT |
|---|---|---|---|
| `scripts/rc_car_jax.py` | 3-state kinematic bicycle | `rc_wh` warehouse | `outputs/dualguard_mppi/matlab/brt_rc_wh_track.mat`, 608 x 408 x 181 |
| `scripts/buzzracer_jax.py` | 6-state dynamic bicycle | RCP tile | `outputs/dualguard_mppi/matlab/brt_buzzracer_environment.mat`, 39 x 57 x 34 x 28 x 5 x 11 (heading axis wrapped) |

The buzzracer model is a MATLAB port of this project's own
`immrax_dynamic_bicycle_cartesian.py`, so it drives the same car on the same track as the
reachset sweep — see "Dynamics provenance" below for the one caveat.

These were notebooks. They are plain scripts now because a notebook's bytes change on
every execution by design, which is exactly the thing DVC's content hashing cannot
version: every re-run looked like a source change to the stage that consumed it. The
numerics are carried over unchanged, `nbconvert`'s two headless accommodations are gone
(matplotlib is set to `Agg` directly, and the `ipywidgets` slider became a GIF), and the
prose that was in markdown cells is now module and function docstrings.

Each script runs until the car **crashes or completes `--laps` laps** (default 3),
whichever comes first, with `--max-sim-time` as a backstop for a run that does neither.
A crash is a negative margin function: for buzzracer the exact 1 cm signed distance
`lx_ref`, deliberately not the coarse `lx` the 6-D solver resolved; for the rc car the
BRT's own `l(x)`, which is already at 1 cm. Lap counting is
`scripts/lap_tracking.py` — cumulative arc length along the racing line, the method
`experiments/dualguard_rc/dualguard_rc_trial.py` uses for the reachset planner on these same tracks,
so the baseline and the thing it is a baseline for count laps the same way.

Layout, with the utility modules shared by both experiments:

```
scripts/_env.py          sys.path setup + canonical input paths
scripts/grids.py         NearestGrid: O(1) lookup on the BRT's uniform axes
scripts/lap_tracking.py  FrenetProjector + LapCounter
scripts/runner.py        Run: per-step history, the stopping rule, the summary
scripts/plotting.py      one figure per file, and the real-speed GIF
scripts/*_model.py       dynamics, BRT tables and the MPPI update, per vehicle
scripts/*_jax.py         the experiment: CLI, sim loop, figures
```

### Outputs

Each stage writes to `runs/<experiment>/`: one PNG per plot (dvc `plots`, so
`dvc plots show` renders them), `animation.gif`, and `summary.json` as an uncached
metric — the run's numbers stay in git, which is the role the notebooks' committed cell
outputs used to play.

`animation.gif` is the interactive track/value-function/rollout view, animated. It plays
at **real speed**: the delay stored in each frame is exactly the simulated time between
the frames it separates. GIF can only express that delay in centiseconds, so
`plotting.real_time_stride` only picks frame strides whose delay is a whole number of
them (at `DT = 0.02` and the default `--gif-fps 25`, every second step, 40 ms a frame).

## Pipeline

This directory has its own `dvc.yaml`, so `dvc repro` at the repo root does not reach it:

```bash
uv run dvc repro experiments/dualguard_mppi/dvc.yaml            # everything
uv run dvc repro experiments/dualguard_mppi/dvc.yaml:run_rc_car # one stage
uv run dvc status experiments/dualguard_mppi/dvc.yaml           # check before reproducing
```

```
vendor/Buzzracer/buzzracer/data/raceline.p
        |
        v  raceline_to_mat            (python, seconds)
outputs/dualguard_mppi/matlab/buzzracer_track_data.mat
        |
        v  brt_buzzracer              (MATLAB, 115 h)
outputs/dualguard_mppi/matlab/brt_buzzracer_environment.mat -> run_buzzracer -> outputs/dualguard_mppi/runs/buzzracer/

outputs/dualguard_mppi/matlab/rc_wh_track_data.mat  (raw)
        |
        v  brt_rc_wh                  (MATLAB, 38 h)
outputs/dualguard_mppi/matlab/brt_rc_wh_track.mat           -> run_rc_car   -> outputs/dualguard_mppi/runs/rc_car/
                                          ^
outputs/dualguard_rc/warehouse_track.npz ------------- '   (root pipeline: reconstruct_track)
```

`run_rc_car` is the one stage with a dependency outside this directory: it takes its
racing line from `WarehouseTrack`, the root pipeline's `reconstruct_track` output, so
lap counting runs against the same rc_wh geometry `experiments/dualguard_rc/dualguard_rc_trial.py`
drives. Reproducing it therefore needs `outputs/dualguard_rc/warehouse_track.npz` on disk
(run `uv run dvc repro reconstruct_track`, which needs `data/brt_rc_wh_fine_v3.mat`, or
supply the `.npz` manually).

**The two `brt_*` stages are multi-day MATLAB solves.** Their outputs predate this
pipeline, so they were seeded into `dvc.lock` with `dvc commit` rather than executed, and
they read as up to date. Touching any of their dependencies — including the `.m` files —
will schedule a re-solve. Run `dvc status` before `dvc repro`.

They need MATLAB with helperOC and ToolboxLS on the path; nothing else here does.
`matlab/verify_buzzracer_port.m` checks the port without running a solve.

## Data

Everything under `matlab/*.mat` is generated by the pipeline and is not included in this
copy; produce it with the stages below or supply it manually:

| file | size | origin |
|---|---|---|
| `brt_buzzracer_environment.mat` | 2.4 GB | `brt_buzzracer` stage |
| `brt_rc_wh_track.mat` | 754 MB | `brt_rc_wh` stage — read by `rc_car_jax.py` |
| `brt_rc_wh_environment.mat` | 1.1 GB | `brt_rc_wh` stage — gradient export, unused here |
| `buzzracer_track_data.mat` | 228 KB | `raceline_to_mat` stage |
| `buzzracer_dynamics_reference.mat` | 484 KB | raw, pinned — see below; no stage reads it |
| `rc_wh_track_data.mat`, `rc_wh_lx_reference.mat` | 190 KB | raw, recovered from the paper's BRT |

Two things deliberately **not** duplicated from the sibling repo:

- **The raceline pickle.** That repo kept its own copy; `utils/raceline_to_mat.py` here
  reads `vendor/Buzzracer/buzzracer/data/raceline.p` directly, so the track geometry has one
  source of truth. Verified to produce byte-identical contents (24/24 keys, max numeric
  difference 0).
- **`brt_rc_wh_fine_v3.mat`**, the paper's original 3-state BRT. `brt_rc_wh_track.m`
  reconstructs it — its own generating script was lost, and every parameter was recovered
  from the file's contents — so `rc_car_jax.py` reads that reconstruction instead. Same grid
  and same variable layout; the failure set is rebuilt from the boundary polygons rather
  than copied, which agrees to 3.5e-5 rms on the track. The repo root still tracks
  `brt_rc_wh_fine_v3.mat` — nothing in this directory reads it, but the root pipeline's
  `reconstruct_track` does, so it is needed indirectly to reproduce `run_rc_car`.

## Dynamics provenance

`@BuzzracerDynamicBicycle`, `buzzracer_dynamics_reference.mat` and the 2.4 GB BRT solved
from them all correspond to `immrax_dynamic_bicycle_cartesian.py` **as of commit 6094bd6**,
whose motor term ran ungated.

`9ffdb56` then added a negative-velocity gate — `a = 0` when `v_forward < 0`, restoring
behaviour the source model always had. That moves `d/dt v_forward` by up to 8.17 on the
kinematic branch (`v_forward < 0.1`, and the BRT grid runs down to -0.40), so regenerating
the parity fixture against HEAD produces one the MATLAB port and `buzzracer_model.py`
disagree with by design.

The fixture is therefore pinned as a raw artifact and `utils/gen_dynamics_reference.py`
is **not** a pipeline stage. Reconciling the port with HEAD means re-solving the 115-hour
BRT against the gated model.

The buzzracer notebook used to assert `f_continuous` against that fixture on every run.
That check has been verified and removed; `matlab/verify_buzzracer_port.m` still holds the
MATLAB side of the port to it, and `utils/gen_dynamics_reference.py` still regenerates it.

## Not imported

The upstream Safe Planar Navigation experiments (section V-A): `src/*.py`, the dubins BRT
and its MATLAB support, `utils/mat_to_h5.py`, and the smoke-grid BRT. The scripts
reference `controller_mppi.py` in prose to explain what they reproduce, but import nothing
from it.
