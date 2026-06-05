# Unified Dclean Flow

`unified_pipeline.py` now represents the repeated dataset code as **one canonical
flow per task** instead of one hand-written pipeline per dataset.

The important design choice is that the flow is unified at the logical level,
while dataset-specific code is only an implementation binding for each logical stage.  In
other words, every supported detection dataset follows the same canonical stage
sequence, and every supported correction dataset follows the same canonical stage
sequence.  The binding layer only answers questions such as "hospital uses
`build_rule_pool_new.py` for `build_rule_pool`" or "soccer uses
the default `revise_candidate.py` for `revise_candidates`."

The unified pipeline no longer shells out to a dataset-specific pipeline script.
It owns the stage ordering, validation, partial execution, and execution context,
then runs each bound stage implementation in-process with the original dataset
directory as `cwd`. That keeps constants, relative paths, model code, seeds, and
output filenames as close as possible to the previous results while centralizing
the pipeline logic.

## Canonical flows

### Detection flow

All six target detection datasets are mapped onto this single flow:

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
  -> evaluate  # only when the original dataset flow has an equivalent evaluation step
```

Supported detection datasets:

- `hospital`
- `beers`
- `flights`
- `rayyan`
- `adult`
- `soccer`

### Correction flow

Correction follows one analogous flow:

```text
wrong_position
  -> revise_candidates
  -> build_features
  -> build_infer_whitelist_features
  -> llm_ranking
  -> repair_loop
  -> evaluate  # only when the original dataset flow has an equivalent evaluation step
```

Supported correction datasets:

- `hospital`
- `beers`
- `flights`
- `rayyan`
- `adult`
- `soccer`

`movies` and `tax` are intentionally not part of this unified target set. The
unified target set is exactly `hospital`, `beers`, `flights`, `rayyan`, `adult`,
and `soccer`. If a local checkout is missing one of the expected dataset
directories, `validate` reports that missing binding instead of silently
substituting another dataset.

## Inspect the unified flow

List all task/dataset bindings:

```bash
python unified_pipeline.py list
```

Print a plan for one dataset:

```bash
python unified_pipeline.py plan --task detection --dataset hospital
python unified_pipeline.py plan --task correction --dataset rayyan
```

Print the same canonical stage for all six detection datasets:

```bash
python unified_pipeline.py plan --task detection --dataset all --only-step detector_loop
```

Emit JSON for tooling:

```bash
python unified_pipeline.py plan --task detection --dataset all --json
```

## Validate configured script bindings

```bash
python unified_pipeline.py validate --task detection --dataset all
python unified_pipeline.py validate --task correction --dataset all
```

Validation checks that every implementation bound to the selected canonical stages exists
under the dataset directory. Optional stages are reported as optional if a script
is absent. Stages disabled in `DATASETS` are intentionally skipped so the unified
flow does not create extra outputs beyond the original dataset flow.

## Run the unified flow

Dry run first:

```bash
python unified_pipeline.py run --task detection --dataset hospital --dry-run
```

Then run for real:

```bash
python unified_pipeline.py run --task detection --dataset hospital
```

Run all datasets registered for a task:

```bash
python unified_pipeline.py run --task detection --dataset all
```

## Partial execution

The canonical stage names are shared across datasets, so the same command shape
works for every dataset:

```bash
# Run feature preparation after detection LLM labeling.
python unified_pipeline.py run --task detection --dataset rayyan \
  --from-step label_propagation --to-step build_loading_set

# Check local preprocessing without LLM/API-dependent stages.
python unified_pipeline.py run --task correction --dataset beers \
  --skip-llm --dry-run

# Re-run one canonical stage.
python unified_pipeline.py run --task correction --dataset soccer \
  --only-step build_features
```

## Result consistency rules

The runner follows these rules to keep each dataset output aligned with its old
flow:

1. Keep the original dataset directory as `cwd` for every stage implementation.
2. Execute stage implementations in-process from one unified flow instead of
   spawning each dataset's old script sequence as separate shell commands.
3. Preserve stage order from checked-in `pipeline.sh` / `run_pipeline.sh` files
   where they exist.
4. Disable canonical stages that would add outputs not produced by the original
   dataset flow, such as rayyan detection's final `evaluate.py`.
5. Keep dataset-specific algorithm bodies behind canonical stage names, so future
   refactors can replace those bodies stage-by-stage without changing the pipeline
   contract.

## Dataset-specific bindings

The unified flow keeps all dataset differences in `DATASETS` inside
`unified_pipeline.py`. These bindings are result-preserving: they use the
implementation file, order, and working directory that the original dataset flow
used whenever that flow exists in the repository:

- beers detection binds `annotate` to `annotate_with_llm.py` and `detector_loop`
  to `control_iterations.py`.
- hospital detection binds the canonical detection stages to the checked-in
  `_new.py` script family and `evaluate_v4.py`.
- rayyan detection matches the checked-in `run_pipeline.sh` order and disables
  `evaluate.py` by default because that shell pipeline stopped after
  `control_iteration.py`.
- hospital correction disables `build_infer_whitelist_features` to preserve its
  existing default controller path.
- Adult detection binds `detector_loop` to two scripts, `train_mlp.py` followed
  by `infer_mlp.py`, because the checked-in controller in this checkout
  references non-existent `*_adult.py` names.
- Soccer uses the default canonical script names (`build_rule_pool.py`,
  `revise_candidate.py`, and so on), matching the expected soccer detection and
  correction folders.

This structure makes the target six datasets share one flow while minimizing result
changes from the previous dataset-specific scripts. If a dataset later gains a
new legacy shell pipeline, update only the `DATASETS` binding/disable list rather
than copying the flow.
