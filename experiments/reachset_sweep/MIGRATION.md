# Sacred → DVC migration manifest

The reachset_sweep workflow was migrated from [Sacred](https://github.com/IDSIA/sacred)
(`FileStorageObserver`) to [DVC](https://dvc.org) + [DVCLive](https://dvc.org/doc/dvclive).
This file records the provenance of the pre-migration results so it survives independently of
git/DVC metadata.

## What changed
- `experiments/reachset_sweep/reachset_sweep.py` — one trial, now instrumented with **DVCLive** (`params.yaml`
  + `metrics.json` per trial) instead of Sacred. Driven by CLI flags.
- `experiments/reachset_sweep_all.py` — **deleted**; replaced by the `reachset_trial` **matrix**
  stage in `dvc.yaml` (axes defined in `params.yaml`).
- `experiments/reachset_sweep/analyze_reachset_sweep.py` — now walks the DVCLive trial dirs under
  `outputs/reachset_sweep/runs/` instead of Sacred run dirs.
- DVC remote: unset in this copy; configure one to fetch archived runs.

## Migrated historical results

Each raw Sacred run directory was `dvc add`-ed on the `dvc-migration` branch in its own tagged
commit; where a corresponding analysis-output snapshot exists, it is git-committed **in the same
commit** as the sim data it was produced from. The source commit is the git hash embedded in
each run's Sacred `run.json` (`experiment.repositories[].commit`); all were recorded with
`dirty: True` (the working tree had uncommitted changes when the sweeps were run), so the source
commit is the best-effort nearest-ancestor, not an exact tree.

| Runs dir | Analysis dir | Runs | Source commit | DVC tag |
|---|---|---|---|---|
| `runs/2026-04-15_reachset_sweep_mem_leak` | — | 1083 | `20d624b` (tmp: sim experiments) | `dvc/2026-04-15-mem-leak` |
| `runs/2026-04-19a_reachset_sweep_no_final_pos` | — | 1090 | `fba048a` (tmp: comparison sweep with mem leak) | `dvc/2026-04-19a-no-final-pos` |
| `runs/2026-04-19b-reachset_sweep_w_final_pos` | `experiments/analysis_output/2026-04-28_reachset_sweep_w_final_pos` | 1080 | `fba048a` (tmp: comparison sweep with mem leak) | `dvc/2026-04-19b-w-final-pos` |
| `runs/2026-04-29_reachset_sweep` | `experiments/analysis_output/2026-04-29_reachset_sweep` | 1080 | `e22ef67` (tmp: more analysis experiments) | `dvc/2026-04-29` |
| `runs/2026-06-20_reachset_sweep` | `outputs/reachset_sweep/analysis` | 1080 | `b9a3678` (tmp: clean up experiments directory) | `dvc/2026-06-20-current` |

Each `experiments/analysis_output/*` snapshot is git-tracked, committed alongside its sim data
on `dvc-migration` (see the "Analysis dir" column). `outputs/reachset_sweep/analysis/`
doubles as the live output of the `analyze_reachset_sweep` DVC stage (the whole folder is the
stage's `cache: false` output); its snapshot rides with the `2026-06-20` data commit.

## Removed: cbf_sweep (Sacred, not migrated)

The `cbf_sweep` experiment (a CBF-parameter sweep on a Van der Pol reachset) was the
last remaining Sacred experiment. Rather than port it to DVC, it was **removed** along
with the rest of the Sacred plumbing (`SacredExperiment` extension, `sacred_utils`, the
`sacred` dependency). Its 81 pre-removal Sacred runs were `dvc add`-ed for provenance
before deletion.

| Runs dir | Runs | Date | DVC tag |
|---|---|---|---|
| `runs/cbf_sweep` | 81 | 2026-06-25 | `sacred/cbf_sweep-2026-06-25` |

Deleted code: `experiments/cbf_sweep.py`, `experiments/cbf_sweep_all.py`,
`experiments/analyze_cbf_sweep.py`.

## Restoring a historical result
```bash
git checkout <tag>      # e.g. dvc/2026-04-29 or sacred/cbf_sweep-2026-06-25
dvc checkout            # materializes runs/<dir> from the DVC cache/remote
```
