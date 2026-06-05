# Unified Dclean Pipeline

`unified_pipeline.py` is now the single implementation of the statistical data
cleaning flow for the six target datasets:

- `hospital`
- `beers`
- `flights`
- `rayyan`
- `adult`
- `soccer`

The old per-dataset scripts are no longer called by the unified runner.  The
pipeline stages, validation, feature construction, statistical detection,
repair-candidate generation, ranking, and evaluation are implemented in one code
path inside `unified_pipeline.py`.  Each dataset only supplies configuration:
input paths, output directories, legacy-compatible filenames, and a few
thresholds.

## Unified detection flow

Every target dataset uses this same detection flow:

```text
build_rule_pool
  -> shrink_search_space
  -> summarize_full_contexts
  -> cluster_and_sample
  -> annotate
  -> label_propagation
  -> build_final_training_set
  -> build_loading_set
  -> detector_loop
  -> evaluate
```

What the stages do:

1. `build_rule_pool` profiles columns and builds not-null, pattern, rare-value,
   and FD-like statistical rules.
2. `shrink_search_space` scores suspicious cells using the unified rule pool.
3. `summarize_full_contexts` builds JSONL/CSV cell contexts.
4. `cluster_and_sample` assigns deterministic buckets and representative samples.
5. `annotate` creates deterministic statistical labels for sampled cells.
6. `label_propagation` propagates labels to all candidates.
7. `build_final_training_set` writes a legacy-compatible training table.
8. `build_loading_set` writes detector inference input.
9. `detector_loop` writes unified detector predictions.
10. `evaluate` compares clean/dirty tables when a clean table is available.

## Unified correction flow

Every target dataset uses this same correction flow:

```text
wrong_position
  -> revise_candidates
  -> build_features
  -> build_infer_whitelist_features
  -> llm_ranking
  -> repair_loop
  -> evaluate
```

What the stages do:

1. `wrong_position` derives repair-allowed cells from detector output or a
   clean/dirty diff.
2. `revise_candidates` generates candidates from column top values and FD-like
   context.
3. `build_features` computes shared repair features.
4. `build_infer_whitelist_features` filters candidates to allowed cells.
5. `llm_ranking` writes deterministic teacher-style pairwise/top-1 labels.
6. `repair_loop` ranks candidates and emits top-1 repairs.
7. `evaluate` compares repairs with the clean table when available.

## Result consistency strategy

The unified implementation is designed to keep outputs close to each dataset's
previous pipeline while eliminating duplicated control flow:

- It preserves legacy-compatible output filenames such as
  `candidate_cells_*.csv`, `candidate_llm_contexts_flat_*.csv`,
  `final_train_dataset_*.csv`, `repair_candidate_features_v2.csv`, and
  `inference_top1_repairs.csv`.
- Dataset-specific differences live in `DatasetConfig` and the two in-code
  domain hooks: paths, thresholds, output names, disabled optional stages,
  validation rules, and repair candidate normalization.
- The same pipeline/stage functions are used for all six datasets; dataset hooks
  are called from those shared functions rather than by dispatching to old
  scripts.
- Optional stages that previously did not exist for a dataset can be disabled so
  the unified flow does not create extra outputs.

## Preserved dataset-specific logic

The unified code does not merely preserve output filenames.  It also carries
forward important per-dataset semantics through in-code hooks:

- `flights`: time-format validation/canonicalization and numeric duration or
  distance checks.
- `beers`: US state checks, ABV/IBU/ounces plausibility, and beer-field
  normalization candidates.
- `hospital`: state and score validity plus MeasureCode/Condition family
  consistency.
- `rayyan`: ISSN, date, and language-field consistency.
- `adult`: age, hours-per-week, and income-label canonicalization.
- `soccer`: score/date/team-field validation and score canonicalization.

These hooks are implemented inside `unified_pipeline.py` as
`dataset_domain_violations()` and `dataset_repair_candidates()`, so the six
datasets still run through one codebase while retaining the high-value logic that
made their original pipelines dataset-aware.

## Runtime dependencies

`list` and `plan` do not require data-science dependencies. Running cleaning
stages requires `pandas`, because the unified implementation reads and writes CSV
tables in-process. If `pandas` is missing, the runner fails with a clear message
when a data-cleaning stage is executed.

## Usage

List supported flows:

```bash
python unified_pipeline.py list
```

Inspect a plan:

```bash
python unified_pipeline.py plan --task detection --dataset hospital
python unified_pipeline.py plan --task correction --dataset beers
python unified_pipeline.py plan --task detection --dataset all --json
```

Validate configured inputs and stages:

```bash
python unified_pipeline.py validate --task detection --dataset hospital
python unified_pipeline.py validate --task correction --dataset rayyan
```

Dry-run prints the selected unified stages without validating input files.

Run a full flow:

```bash
python unified_pipeline.py run --task detection --dataset hospital
python unified_pipeline.py run --task correction --dataset hospital
```

Run selected stages:

```bash
python unified_pipeline.py run --task detection --dataset rayyan \
  --from-step build_rule_pool --to-step build_loading_set

python unified_pipeline.py run --task correction --dataset soccer \
  --only-step build_features
```

For one-off tests or alternate local data paths, override the configured paths:

```bash
python unified_pipeline.py run --task detection --dataset beers \
  --dirty-csv /tmp/dirty.csv \
  --clean-csv /tmp/clean.csv \
  --workdir /tmp/unified_detection_beers
```

`movies` and `tax` are intentionally outside this target set.  If a local
checkout is missing `Error Detection soccer`, `Error Correction soccer`, or
`Error Correction Adult`, validation reports the missing directory; the unified
code is already configured for those folders once they are present.
