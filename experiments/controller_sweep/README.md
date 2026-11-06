# Controller × track × disturbance sweep

The `controller_trial` DVC matrix compares `CarController` implementations on
the `saved` track under the same two disturbances used by `reachset_trial`
(`uniform` and `adversarial_away`, `lateral_only`, magnitude `0.1`), with every
controller asked to complete 10 laps.

Axes live in `params.yaml::controller_sweep`:

```yaml
controller_sweep:
  controller: [immrax, shieldmppi, mppi, tubemppi]
  track: [saved]
  state_noise_direction: [uniform, adversarial_away]
  lap_count: 10
  wall_clock: 180
```

`lap_count` and `wall_clock` override the reachset-sweep values in `sim`; the
trial command passes them after `${sim}`, so the later flags win.

Fixed settings:

```yaml
controller:
  use_reachsets: true
  centering_weight_factor: 1.0
  velocity_scale_factor: 1.0
  # Shield-MPPI only (ignored by ImmraxController):
  shield_num_samples: 1024
  shield_param_lambda: 1.0
  shield_param_alpha: 0.0
  shield_param_beta: 0.5
  shield_cbf_cost_weight: 1000.0
  shield_w_T: 0.0
  shield_repair_horizon: 10
  shield_repair_steps: 20
  # MPPI only (ignored by ImmraxController / ShieldMPPIController):
  mppi_num_samples: 1024
  mppi_param_lambda: 1.0
  # Tube-MPPI only (ignored by ImmraxController / ShieldMPPIController):
  tube_control_cost_lambda: 0.03
  tube_ancillary_horizon: 0
  tube_ancillary_iterations: 1
  tube_ancillary_mu: 0.01
  tube_ancillary_mu_factor: 5.0
  tube_ancillary_q_scale: 1.0
  tube_ancillary_r_scale: 1.0
  tube_ancillary_terminal_scale: 10.0
```

That is 8 trials, keyed `<controller>-<track>-<disturbance>` under
`outputs/controller_sweep/runs/`. `analyze_controller_sweep` writes `runs.csv` and
`summary.csv` to `outputs/controller_sweep/analysis/`, summarizing only the
tracks currently listed in `controller_sweep.track`; DVC leaves outputs of
dropped matrix combinations on disk, so runs from other tracks are ignored
rather than reported.

`shieldmppi` (`shield_mppi_controller.py`) drives the `Shield-MPPI` submodule's
MPPI solver with the same 6-state dynamic bicycle, control bounds and state cost
terms as `ImmraxController`. MPPI samples a 2D control `[steering, throttle]`
whose std is mapped from Immrax's derivative-space `SampleBounds`. Rollouts are
deterministic; the DCBF penalty is supplied by the solver's `compute_cbf_cost`,
with `w_T = 2 * min(min(left), min(right))` (the tightest one-sided clearance,
applied uniformly over the trial). The rollout cost holds only Immrax's state
terms (the hard collision penalty is omitted; the soft arctan boundary remains);
input regularization comes from the solver's own free-energy term. After the
MPPI update, the paper's local repair runs BFGS on the first
`shield_repair_horizon` controls; the repaired first control is executed, while
the pre-repair mean warm-starts the next step.

`mppi` (`mppi_controller.py`) is vanilla MPPI. It reuses the JAX
`DynamicBicycleCartesian` rollouts and `PointTrajectoryCost` that
`ImmraxController` uses with reachsets off, with the cost's control-difference
regularization disabled (`control_diff_weight=0`); the instantaneous
control-magnitude term remains. Each step samples `mppi_num_samples` full
control sequences around the current plan with the per-step std derived from
Immrax's derivative-space `SampleBounds`, scores the rollouts, and replaces the
plan with the softmax(`mppi_param_lambda`)-weighted mean. The plan then shifts
forward by one control. Rollouts use zero disturbance.

`tubemppi` (`tube_mppi_controller.py`, `TubeMPPI(MPPI)`) implements the
Tube-MPPI algorithm of Williams et al. (2018). It keeps a disturbance-free
nominal state; each tick runs MPPI from the nominal state and from the actual
state (Algorithm 2), and adopts the actual-state solution when its noiseless
cost is within `tube_accept_margin` of the nominal solution's cost (default:
the smaller of the collision and boundary cost weights, `10.0`). The nominal
controller uses the paper's update (raw-noise mean from Algorithm 1) and the
paper's cost (`SigmaWeightedControlCost`): the `PointTrajectoryCost` state terms
including the sparse collision indicator, plus `lambda * u^T Sigma^-1 u` weighted
by `tube_control_cost_lambda`. An iLQG ancillary controller (`tube_ancillary.py`)
tracks the accepted nominal trajectory over `tube_ancillary_horizon` steps
(default `horizon // 2`) and supplies the applied control.

```bash
uv run dvc repro build_track_fixtures     # normalize tracks (one-off)
uv run dvc repro analyze_controller_sweep # all trials + analysis
```

To run one cell:
```bash
uv run dvc repro 'controller_trial@immrax-saved-uniform'
```

## Plots

`dvc plots show <id>` renders the comparison plots defined in `dvc.yaml` from
`analysis_output/controller_sweep/runs.csv` (they also show up in `dvc exp`):

| Plot | Answers |
|---|---|
| `crash_rate_by_track` | Which maps are hardest (tracks sorted by crash rate) |
| `crash_rate_by_controller` | Which controller is safest overall |
| `outcomes_by_track` | Why maps are hard: finished / crash / stall counts |
| `control_time_by_controller` | Which controller is fastest to compute |
| `lap_time_by_controller` | Which controller is fastest among completed laps |
| `lap_time_by_track` | Same comparison against each track's baseline lap time |

## Recorded metrics

- `outcome` / `stop_reason`: `race_finished`, `crash`, `stall`, or `incomplete`,
  using the same framework as `vendor/Buzzracer/.../auto_stopper.py`.
- `control_time_ms_mean/std/median`: wall-clock time per `controller.control()`
  call, excluding the first 5 steps. `planner_compute_ms_mean` is also logged
  for controllers that expose `_timing_log`.
- `average_lap_time`, `num_laps_complete`: mean lap time over completed laps.
- Progress/safety context: `sim_duration`, `distance_progressed`,
  `collision_count`, `oob_fraction`, and (when available) `unsafe_fraction`.

## Fair-comparison choices

- The per-trial seed depends only on `(track, disturbance)`, not the
  controller, so every controller on a cell sees the same plant-noise stream.
  `np.random` is seeded too, since the simulator uses the global NumPy RNG.
- `SimTimeAutoStopper` (in `experiments/controller_sweep/_extensions.py`) keeps the stock
  `AutoStopper` logic but measures the wall-clock and stall limits in
  simulated seconds, so slower controllers are not judged stalled sooner.
- `use_reachsets` is fixed enabled.
- Every track is presented through the same `RacelineTrack` API (nearest-point
  projection, spline-derived curvature, periodic speed interpolation), so the
  Stanley warm start and the Immrax cost see identical geometry regardless of
  the native track class.

## Track preparation

`build_track_fixtures` writes `outputs/controller_sweep/track_fixtures/<name>.npz`; each trial loads
one through `raceline_track.RacelineTrack`. No submodule source file is edited.

| Track | Status as shipped | How the fixture is completed |
|---|---|---|
| `saved` | complete | used directly (QP-smoothed `full` layout from `raceline.p`) |
| `small`, `easy`, `big` | missing derived arrays | QP-smoothed with `QpSmooth`, then `reconstruct_raceline()` + `prepare_discretized_raceline()` |
| `skidpad` | missing arrays, `sToV`, `curvature_s` | arrays derived from the circle spline; curvature-limited speed profile |
| `nascar` | missing arrays, `sToV`, `curvature_s`; 5-column discretization | `common` import aliased; arrays derived from the spline; 6-column schema |
| `orca` | missing arrays; wrong data path; no `local_trajectory` | parent adapter reads `RL/copg/.../ORCA_OPT`; arrays built from the raw files |
| `empty` | no raceline | excluded |
| `full` | same layout as `saved` | excluded to avoid a duplicate |
| `warehouse` | complete, but parent-defined | not part of this sweep |

## Adding a controller

Register one entry in `experiments/controller_sweep/_controller_registry.py`:

```python
register("stanley", "StanleyCarController")  # class in Buzzracer
register("mine", "MyController", MyController)  # class in this repo
```

Then add the name to `params.yaml::controller_sweep.controller`. A parent-repo
class is injected into `buzzracer.controllers` at runtime; no submodule file
changes. If the controller needs new fixed knobs, add the corresponding flags
in `controller_sweep.py` and keys to `params.yaml::controller`.

## Adding a track

Add an adapter branch in `experiments/controller_sweep/_track_setup.py::build_track`, add the
name to `SUPPORTED_TRACKS` and `params.yaml::controller_sweep.track`, and rerun
`dvc repro build_track_fixtures`.
