# sweet.tv RecSys-Exam

## Purpose

Predict the top-5 TV programmes for each of the 1,260 submission users for the
weeks 31-42 schedule, using only viewing history and programme schedules. The
task is evaluated with MAP@5.

The final model is a three-part blend of:

- `history_days` share (weight `0.30`)
- matrix factorization with `k=16` (weight `0.02`)
- popularity share (weight `0.68`)

The final Kaggle submission is **`submissions/mf_blend.csv`**, with private
score **0.33537**.

## Data placement

Place the raw competition files in `data/` using these exact names:

| File | Description |
|---|---|
| `data/dataset11-30.csv` | Raw view log: `user_id`, `vsetv_id`, `start_time`, `stop_time`, `duraton` |
| `data/export_arh_11-20-final.csv` | Schedule and programme metadata for weeks 11-20 |
| `data/export_arh_21-30-final.csv` | Schedule and programme metadata for weeks 21-30 |
| `data/export_arh_31-42-final.csv` | Schedule and programme metadata for weeks 31-42 |
| `data/submission.csv` | 1,260 submission users and the required output format |

`src/join.py` creates `data/interactions.parquet`, which contains the labelled,
qualifying view-programme interactions used by the model scripts.

`data/` is intentionally **not committed**. `.gitignore` excludes `data/`,
`*.parquet`, and `__pycache__/`. Commit only source code, submissions, and
result files.

## Environment

Versions actually used:

- Python 3.11.9
- pandas 2.3.3
- numpy 2.3.5
- scipy 1.17.1

The scripts were run on Windows with PowerShell using `python` (not `python3`).
No dependencies beyond pandas, numpy, and scipy are required.

## Exact run order

Run all commands from the repository root. This is the order used in the
session:

| Step | Command | Output |
|---:|---|---|
| 1 | `python src/explore.py` | Optional profiling report; no files |
| 2 | `python src/join.py` | `data/interactions.parquet` |
| 3 | `python src/popularity.py` | `submissions/popularity.csv` |
| 4 | `python src/validate.py` | Validation diagnostics and history table; no files |
| 5 | `python src/blend.py` | `submissions/blend_w02.csv` |
| 6 | `python src/validate_methods.py` | `results/validation.csv` |
| 7 | `python src/submit_methods.py` | `submissions/mf_k16.csv`, `submissions/content.csv`, `submissions/mf_blend.csv` |
| 8 | `python src/examples.py` | `results/examples.csv`, `results/examples.md` |

`src/explore.py` is optional. `src/join.py` must run before all modelling
scripts. `src/validate_methods.py` should run before `src/submit_methods.py`
and `src/examples.py` so the final weights and validation results are fixed.

## Seeds and determinism

- Validation seed: `SEED = 0` in `src/validate.py`.
- MF decomposition: `scipy.sparse.linalg.svds(..., random_state=0)`.
- Paired bootstrap: `BOOTSTRAP_SEED = 0`, 2,000 resamples.
- Tie-breaking: sort by `tv_show_id` ascending.
- Content TF-IDF, popularity, joins, and submission generation are
  deterministic. Re-running the submission scripts produces the same files.

## Approximate runtimes

Measured on the development machine; timings will vary.

| Script | Approx. runtime |
|---|---:|
| `src/explore.py` | 58 s |
| `src/join.py` | 47 s |
| `src/popularity.py` | 4 s |
| `src/validate.py` | 10 s |
| `src/blend.py` | 5 s |
| `src/validate_methods.py` | 68 s |
| `src/submit_methods.py` | 15 s |
| `src/examples.py` | 17 s |

The full pipeline is roughly 3-4 minutes on this machine.

## Validation and method notes

- Validation split: train on ISO weeks 11-20; truth on weeks 21-30.
- Candidate pool: 6,363 unique labelled programmes (`tv_show_id != 0`) in
  `export_arh_21-30-final.csv`.
- Real split: history/MF on weeks 21-30; popularity on weeks 26-30; candidates
  are the 6,636 unique labelled programmes in
  `export_arh_31-42-final.csv`.
- Join rule: a view is matched to a schedule airing on the same channel when
  their overlap is at least `min(300 seconds, 0.5 * programme duration)`.
  Only qualifying pairs with a labelled programme (`tv_show_id != 0`) enter
  `interactions.parquet`.
- `tv_show_id = 0` means unlabelled airtime. It is never used as a
  recommendation item, candidate, or truth item. It is used only as viewing
  context: the channel-affinity experiment uses all raw views, including views
  that matched only unlabelled airtime.
- Truth sets are per-user and repeats are allowed in the raw truth. Evaluation
  uses unique programmes, so repeated watches of the same programme do not
  create extra credit. Recommendations are top-5 sets ordered by score, with
  `tv_show_id` ascending as the tie-break.
- Cold start: any method with fewer than 5 usable recommendations is filled
  with popularity top-5 from weeks 16-20 in validation and weeks 26-30 for the
  real submission. The final blend has a popularity component, so it needs no
  fill.
- Content-only and channel-affinity were tested and rejected. Channel affinity
  alone reached validation MAP@5 `0.57947`; the best 15-combination four-part
  blend reached `0.79924`, below the final blend's `0.80307`, with bootstrap CI
  `[-0.01064, +0.00314]`. No channel submission was written.
- The Kaggle scores were: `mf_blend` 0.33537, `blend_w02` 0.32857,
  `mf_k16` 0.32437, `popularity` 0.27750, `content` 0.15438. The ordering
  matches the validation ordering.

## Repository outputs

- `submissions/mf_blend.csv`: final submission, Kaggle private 0.33537.
- `submissions/mf_k16.csv`, `submissions/content.csv`: required method
  submissions.
- `submissions/blend_w02.csv`, `submissions/popularity.csv`: earlier
  baselines.
- `results/validation.csv`: validation table with bootstrap CIs where computed.
- `results/examples.csv`, `results/examples.md`: validation recommendation
  examples for one heavy and one light user.
