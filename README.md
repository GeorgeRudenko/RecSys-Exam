# sweet.tv RecSys-Exam

## Purpose

Predict the top-5 TV programmes for each of the 1,260 submission users for the
weeks 31-42 schedule, using only viewing history and programme schedules. The
task is evaluated with MAP@5.

The revised final model is the reranker from `src/rerank2.py`: a LightGBM
classifier plus a LambdaRank model, both trained on strict watch labels and
combined by averaging their per-user min-max scores.

The final Kaggle submission is **`submissions/rerank2.csv`**, with private
score **0.43195** and public score **0.44421**. The previous `mf_blend.csv`
submission scored 0.33537 private.

## Data placement

Place the raw competition files in `data/` using these exact names:

| File | Description |
|---|---|
| `data/dataset11-30.csv` | Raw view log: `user_id`, `vsetv_id`, `start_time`, `stop_time`, `duraton` |
| `data/export_arh_11-20-final.csv` | Schedule and programme metadata for weeks 11-20 |
| `data/export_arh_21-30-final.csv` | Schedule and programme metadata for weeks 21-30 |
| `data/export_arh_31-42-final.csv` | Schedule and programme metadata for weeks 31-42 |
| `data/submission.csv` | 1,260 submission users and the required output format |

`src/join.py` creates `data/interactions.parquet`, the labelled,
qualifying view-programme interactions used by the earlier methods.
`src/occurrences.py` creates `data/occurrences.parquet`, which holds the
per-view/per-airing overlap fraction used to construct the strict watch label.

`data/` is intentionally **not committed**. `.gitignore` excludes `data/`,
`*.parquet`, and `__pycache__/`. Commit only source code, submissions, and
result files.

## Strict watch label

The revised experiments use a strict watch label following the
fraction > 0.8 convention after Melentev 2021:

- An occurrence is a raw view and a scheduled programme airing with positive
  overlap on the same channel.
- The occurrence fraction is the overlap duration divided by the programme
  duration.
- A user-programme pair is **strict positive** in a target window when at least
  one matching occurrence in that window has `fraction > 0.8`.
- `tv_show_id = 0` means unlabelled airtime. It is never a candidate, truth
  item, or recommendation. It is used only as viewing context where noted.

The strict label was used for model selection and for the final reranker. The
older evaluation is retained in `results/validation_strict.csv` as
`map5_old_truth`, but it is not the selection metric.

## Final model

The final model is the C configuration in `src/rerank2.py`.

- Validation split: history weeks 11-20, strict truth weeks 21-30, 1,260 users.
- Candidates per user: union of repeat top-50, strict-popularity top-20, and
  MF top-20.
- Features: 21 base reranker features plus 4 slot-affinity features
  (`slot_score_sum`, `slot_score_max`, `slot_top10_airings`,
  `user_channel_share_sum`).
- Models: LightGBM classifier and LightGBM LambdaRank, both with 300 trees,
  15 leaves, learning rate 0.05, seed 0.
- Score: mean of the classifier and LambdaRank scores after per-user
  min-max normalization.
- Validation score: strict MAP@5 **0.59086**.
- Final refit folds: 11-15 -> 16-20, 11-17 -> 18-20, 11-20 -> 21-30, and
  11-25 -> 26-30.
- Final prediction history: weeks 11-30; final candidate pool: weeks 31-42.

`src/rerank.py` is the earlier reranker baseline (strict MAP@5 0.58578). The
final C reranker improved on it by `+0.00509`, with paired-bootstrap 95% CI
`[+0.00006, +0.01035]`.

## Exact run order

Run from the repository root. This is the order used for the revised work:

| Step | Command | Output |
|---:|---|---|
| 1 | `python src/explore.py` | Optional profiling; no files |
| 2 | `python src/join.py` | `data/interactions.parquet` |
| 3 | `python src/occurrences.py` | `data/occurrences.parquet` |
| 4 | `python src/popularity.py` | `submissions/popularity.csv` |
| 5 | `python src/validate.py` | Validation diagnostics; no files |
| 6 | `python src/blend.py` | `submissions/blend_w02.csv` |
| 7 | `python src/validate_methods.py` | `results/validation.csv` |
| 8 | `python src/submit_methods.py` | `submissions/mf_k16.csv`, `submissions/content.csv`, `submissions/mf_blend.csv` |
| 9 | `python src/repeat.py` | `results/validation_strict.csv` |
| 10 | `python src/repeat_submit.py` | `submissions/repeat.csv` |
| 11 | `python src/rerank.py` | `submissions/rerank.csv` |
| 12 | `python src/rerank2.py` | `submissions/rerank2.csv` (final) |
| 13 | `python src/examples.py` | `results/examples.csv`, `results/examples.md` |

`src/join.py` must run before all modelling scripts. `src/occurrences.py` must
run before `src/repeat.py`, `src/repeat_submit.py`, `src/rerank.py`, and all
rerank variants. `src/rerank2.py` requires `src/rerank.py` as an imported
module but writes only `submissions/rerank2.csv`.

The rejected experiments are optional and are not required to reproduce the
final submission:

```text
python src/rerank3.py
python src/rerank_bag.py
python src/rerank4.py
python src/rerank5.py
python src/rerank6.py
python src/rerank7.py
python src/rerank8.py
python src/rerank9.py
```

These scripts write a submission only if their own paired-bootstrap condition
is met. No rejected experiment met it, so only `rerank.csv` and
`rerank2.csv` exist from the reranker family.

## Strict validation ladder

Selected results from `results/validation_strict.csv`:

| Method | Strict MAP@5 |
|---|---:|
| popularity | 0.41323 |
| history_days | 0.49609 |
| mf_k16 | 0.51346 |
| mf_blend | 0.52844 |
| repeat, threshold 0.30, fit 11-20, fill b | 0.53910 |
| rerank LightGBM | 0.58578 |
| rerank2 C average | **0.59086** |

## Rejected experiments

| Script / configuration | Strict MAP@5 | Result |
|---|---:|---|
| `rerank_bag.py`, seed-bagged C | 0.59114 | `+0.00028` vs C; 95% CI `[-0.00246, +0.00298]`; rejected |
| `rerank3.py`, C without slot features | 0.58910 | CI `[-0.00630, +0.00273]`; rejected |
| `rerank4.py`, best added block G | 0.58924 | D 0.57883, E 0.58864, F 0.58812, G 0.58924; all below base 0.59086 |
| `rerank5.py`, base + F3+F4 | 0.58825 | base + F3+F4+F5 0.58560; CI `[-0.00658, +0.00141]`; rejected |
| `rerank6.py`, AR candidates/features | 0.58993 | best rank fusion 0.59178; CI `[-0.00213, +0.00417]`; rejected |
| `rerank7.py`, window normalization + time features | 0.58943 | N 0.58922; T 0.58718; N+T 0.58943; rejected |
| `rerank8.py`, L2 soft labels | 0.58282 | L1 0.58123; CI `[-0.01290, -0.00366]`; rejected |
| `rerank9.py`, combined candidates/features/early stopping | 0.58421 | CI `[-0.01269, -0.00067]`; rejected |

The rejected scripts left earlier outputs untouched. Their detailed outputs
and the reasons for rejection are also documented in the session log, which is
not committed.

## Repository outputs

- `submissions/rerank2.csv`: final submission; Kaggle private 0.43195,
  public 0.44421.
- `submissions/rerank.csv`: earlier reranker; strict MAP@5 0.58578.
- `submissions/repeat.csv`: strict-repeat submission.
- `submissions/mf_blend.csv`, `submissions/mf_k16.csv`,
  `submissions/content.csv`, `submissions/blend_w02.csv`,
  `submissions/popularity.csv`: earlier baselines and required method
  submissions.
- `results/validation_strict.csv`: 16-method strict/old validation table.
- `results/validation.csv`: earlier method validation results.
- `results/examples.csv`, `results/examples.md`: validation recommendation
  examples; `examples.md` also includes the appended strict-repeat and
  reranker examples.

The old Kaggle scores were: `mf_blend` 0.33537, `blend_w02` 0.32857,
`mf_k16` 0.32437, `popularity` 0.27750, `content` 0.15438. The revised
`rerank2` submission is the best validation and Kaggle result.

## Seeds and determinism

- Validation seed and model seed: `0`.
- MF decomposition: `scipy.sparse.linalg.svds(..., random_state=0)`.
- LightGBM classifier and LambdaRank: `random_state=0`.
- Paired bootstrap: `BOOTSTRAP_SEED = 0`, 2,000 resamples.
- Tie-breaking: sort by `tv_show_id` ascending.
- Popularity, joins, occurrence construction, and submission generation are
  deterministic.

## Environment

Versions used:

- Python 3.11.9
- pandas 2.3.3
- numpy 2.3.5
- scipy 1.17.1
- lightgbm 4.7.0
- scikit-learn 1.9.1

The scripts were run on Windows with PowerShell using `python` (not
`python3`).

## Approximate runtimes

Measured on the development machine:

| Script | Approx. runtime |
|---|---:|
| `src/rerank.py` | 61 s |
| `src/rerank2.py` | 239 s |
| `src/rerank3.py` | 154 s |
| `src/rerank_bag.py` | 171 s |
| `src/rerank4.py` | 274 s |
| `src/rerank5.py` | 242 s |
| `src/rerank6.py` | 163 s |
| `src/rerank7.py` | 307 s |
| `src/rerank8.py` | 148 s |
| `src/rerank9.py` | 258 s |

The final rerank2 pipeline is roughly 4 minutes including fold construction
and the final refit.
