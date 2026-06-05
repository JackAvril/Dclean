#!/usr/bin/env python3
"""
Unified dataset pipeline runner for error detection and error correction.

This file expresses the six dataset workflows as one canonical flow per task.
Dataset differences are limited to a small configuration layer that selects the
existing script name for each canonical stage.  Each selected script is still run
inside a preserved dataset execution context, so constants, relative paths,
random seeds, model code, and output filenames are preserved as much as possible
without shelling out to each dataset script as a separate process.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parent
ScriptOverride = tuple[str, ...] | None


@dataclass(frozen=True)
class FlowStage:
    """One logical stage in a canonical task flow."""

    name: str
    default_scripts: tuple[str, ...]
    description: str
    llm: bool = False
    optional: bool = False


@dataclass(frozen=True)
class RunnableStep:
    """One configured stage for one dataset."""

    name: str
    scripts: tuple[str, ...]
    description: str = ""
    llm: bool = False
    optional: bool = False

    def implementation_paths(self) -> list[str]:
        return list(self.scripts)


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset-specific mapping from canonical stages to existing scripts."""

    dataset: str
    detection_dir: str | None = None
    correction_dir: str | None = None
    overrides: dict[str, dict[str, ScriptOverride]] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelineConfig:
    """A concrete canonical flow for one task and one dataset."""

    task: str
    dataset: str
    directory: str
    steps: tuple[RunnableStep, ...]
    notes: tuple[str, ...] = ()

    @property
    def workdir(self) -> Path:
        return REPO_ROOT / self.directory


DETECTION_FLOW: tuple[FlowStage, ...] = (
    FlowStage("build_rule_pool", ("build_rule_pool.py",), "learn rule/statistical profile pool"),
    FlowStage("shrink_search_space", ("shrink_search_space.py",), "select suspicious candidate cells"),
    FlowStage("summarize_full_contexts", ("summarize_full_contexts.py",), "build rich and flat candidate contexts"),
    FlowStage("cluster_and_sample", ("cluster_and_sample_candidates.py",), "cluster candidates and sample LLM labeling set"),
    FlowStage("annotate", ("annotate_with_deepseek.py",), "LLM-label sampled candidate cells", llm=True),
    FlowStage("label_propagation", ("label_propagation_for_candidates.py",), "propagate LLM labels inside clusters/neighborhoods"),
    FlowStage("build_final_training_set", ("build_final_training_set.py",), "merge labels into detector training data"),
    FlowStage("build_loading_set", ("build_loading_set.py",), "build detector inference input"),
    FlowStage("detector_loop", ("control_iteration.py",), "run detector train/infer and hard-case feedback loop", llm=True),
    FlowStage("evaluate", ("evaluate.py",), "evaluate detected error cells", optional=True),
)


CORRECTION_FLOW: tuple[FlowStage, ...] = (
    FlowStage("wrong_position", ("wrong_position.py",), "prepare/evaluate wrong cell positions", optional=True),
    FlowStage("revise_candidates", ("revise_candidate.py",), "generate repair candidates"),
    FlowStage("build_features", ("build_features.py",), "build repair candidate features"),
    FlowStage("build_infer_whitelist_features", ("build_infer_candidate_features_from_whitelist.py",), "filter candidates to detected/allowed cells"),
    FlowStage("llm_ranking", ("llm_ranking.py",), "collect LLM pairwise ranking labels", llm=True),
    FlowStage("repair_loop", ("control_iteration.py",), "run LTR train/infer and hard-case feedback loop", llm=True),
    FlowStage("evaluate", ("evaluate.py",), "evaluate final repairs", optional=True),
)


CANONICAL_FLOWS: dict[str, tuple[FlowStage, ...]] = {
    "detection": DETECTION_FLOW,
    "correction": CORRECTION_FLOW,
}


TARGET_DATASETS: tuple[str, ...] = ("hospital", "beers", "flights", "rayyan", "adult", "soccer")


DATASETS: dict[str, DatasetConfig] = {
    "hospital": DatasetConfig(
        dataset="hospital",
        detection_dir="Error Detection hospital",
        correction_dir="Error Correction hospital",
        overrides={
            "detection": {
                "build_rule_pool": ("build_rule_pool_new.py",),
                "shrink_search_space": ("shrink_search_space_new.py",),
                "summarize_full_contexts": ("summarize_full_contexts_new.py",),
                "cluster_and_sample": ("cluster_and_sample_candidates_new.py",),
                "annotate": ("annotate_with_deepseek_new.py",),
                "label_propagation": ("label_propagation_for_candidates_new.py",),
                "build_final_training_set": ("building_final_training_set_new.py",),
                "build_loading_set": ("build_loading_set_new.py",),
                "evaluate": ("evaluate_v4.py",),
            },
            "correction": {
                # The existing hospital repair flow does not use the whitelist feature
                # builder in its default controller path. Disable this canonical stage
                # for result-preserving behavior instead of deleting the stage globally.
                "build_infer_whitelist_features": None,
            },
        },
        notes=(
            "Hospital detection uses the checked-in *_new.py/evaluate_v4.py script family; "
            "hospital correction preserves its existing non-whitelist default path.",
        ),
    ),
    "beers": DatasetConfig(
        dataset="beers",
        detection_dir="Error Detection beers",
        correction_dir="Error Correction beers",
        overrides={
            "detection": {
                "annotate": ("annotate_with_llm.py",),
                "detector_loop": ("control_iterations.py",),
            }
        },
    ),
    "flights": DatasetConfig(
        dataset="flights",
        detection_dir="Error Detection flights",
        correction_dir="Error Correction flights",
        overrides={
            "detection": {
                "detector_loop": ("control_iteration.py",),
            }
        },
    ),
    "rayyan": DatasetConfig(
        dataset="rayyan",
        detection_dir="Error Detection rayyan",
        correction_dir="Error Correction rayyan",
        overrides={
            "detection": {
                # The checked-in Error Detection rayyan/run_pipeline.sh ends at
                # control_iteration.py, so do not add evaluate.py by default.
                "evaluate": None,
            }
        },
    ),
    "adult": DatasetConfig(
        dataset="adult",
        detection_dir="Error Detection Adult",
        correction_dir="Error Correction Adult",
        overrides={
            "detection": {
                # The checked-in Adult controller references non-existent *_adult.py
                # files in the current checkout, so this dataset keeps the same
                # canonical detector-loop stage but materializes it as direct train
                # then infer scripts unless the dataset-specific controller is fixed.
                "detector_loop": ("train_mlp.py", "infer_mlp.py"),
            }
        },
    ),
    "soccer": DatasetConfig(
        dataset="soccer",
        detection_dir="Error Detection soccer",
        correction_dir="Error Correction soccer",
    ),
}



class PipelineError(RuntimeError):
    """Raised for invalid pipeline configuration or failed execution."""


def task_dir(dataset_config: DatasetConfig, task: str) -> str | None:
    if task == "detection":
        return dataset_config.detection_dir
    if task == "correction":
        return dataset_config.correction_dir
    raise PipelineError(f"Unknown task {task!r}")


def configured_scripts(dataset_config: DatasetConfig, task: str, stage: FlowStage) -> ScriptOverride:
    return dataset_config.overrides.get(task, {}).get(stage.name, stage.default_scripts)


def build_pipeline(task: str, dataset: str) -> PipelineConfig:
    task = task.lower()
    dataset = dataset.lower()
    if task not in CANONICAL_FLOWS:
        raise PipelineError(f"Unknown task {task!r}. Available tasks: {', '.join(sorted(CANONICAL_FLOWS))}")
    try:
        dataset_config = DATASETS[dataset]
    except KeyError as exc:
        raise PipelineError(f"Unknown dataset {dataset!r}. Available datasets: {', '.join(TARGET_DATASETS)}") from exc

    directory = task_dir(dataset_config, task)
    if directory is None:
        raise PipelineError(f"Dataset {dataset!r} does not have a configured {task} flow in this repository")

    steps: list[RunnableStep] = []
    for stage in CANONICAL_FLOWS[task]:
        scripts = configured_scripts(dataset_config, task, stage)
        if scripts is None:
            continue
        steps.append(
            RunnableStep(
                name=stage.name,
                scripts=scripts,
                description=stage.description,
                llm=stage.llm,
                optional=stage.optional,
            )
        )
    return PipelineConfig(task=task, dataset=dataset, directory=directory, steps=tuple(steps), notes=dataset_config.notes)


def available_tasks() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for task in sorted(CANONICAL_FLOWS):
        datasets = [name for name in TARGET_DATASETS if task_dir(DATASETS[name], task) is not None]
        out[task] = datasets
    return out


def expand_datasets(task: str, dataset: str) -> list[str]:
    dataset = dataset.lower()
    if dataset != "all":
        return [dataset]
    return available_tasks()[task.lower()]


def slice_steps(
    steps: Sequence[RunnableStep],
    *,
    from_step: str | None = None,
    to_step: str | None = None,
    only_steps: Sequence[str] = (),
    skip_steps: Sequence[str] = (),
    include_llm: bool = True,
) -> tuple[RunnableStep, ...]:
    selected = list(steps)
    names = [s.name for s in selected]

    if from_step:
        if from_step not in names:
            raise PipelineError(f"Unknown --from-step {from_step!r}. Valid steps: {', '.join(names)}")
        selected = selected[names.index(from_step) :]
        names = [s.name for s in selected]
    if to_step:
        if to_step not in names:
            raise PipelineError(f"Unknown --to-step {to_step!r}. Valid steps after slicing: {', '.join(names)}")
        selected = selected[: names.index(to_step) + 1]

    if only_steps:
        only = set(only_steps)
        unknown = only.difference(s.name for s in steps)
        if unknown:
            raise PipelineError(f"Unknown --only-step value(s): {', '.join(sorted(unknown))}")
        selected = [s for s in selected if s.name in only]

    if skip_steps:
        skip = set(skip_steps)
        unknown = skip.difference(s.name for s in steps)
        if unknown:
            raise PipelineError(f"Unknown --skip-step value(s): {', '.join(sorted(unknown))}")
        selected = [s for s in selected if s.name not in skip]

    if not include_llm:
        selected = [s for s in selected if not s.llm]

    return tuple(selected)


def validate_pipeline(config: PipelineConfig, steps: Iterable[RunnableStep] | None = None) -> list[str]:
    messages: list[str] = []
    if not config.workdir.is_dir():
        messages.append(f"missing workdir: {config.workdir}")
        return messages

    for step in steps or config.steps:
        for script in step.scripts:
            script_path = config.workdir / script
            if not script_path.is_file():
                level = "optional missing" if step.optional else "missing"
                messages.append(f"{level}: {step.name} -> {script_path}")
    return messages


@contextmanager
def dataset_execution_context(workdir: Path, script_path: Path):
    """Preserve legacy relative-path and argv behavior while staying in-process."""

    old_cwd = Path.cwd()
    old_argv = sys.argv[:]
    old_path = sys.path[:]
    try:
        os.chdir(workdir)
        sys.argv = [str(script_path)]
        sys.path.insert(0, str(workdir))
        yield
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv
        sys.path[:] = old_path


def execute_stage_implementation(script_path: Path, workdir: Path) -> None:
    """Execute one stage implementation inside the unified pipeline process.

    The implementation file is run with ``__name__ == "__main__"`` so existing
    stage code keeps its original behavior, but the unified pipeline controls the
    stage ordering, dataset binding, validation, and execution context.
    """

    with dataset_execution_context(workdir, script_path):
        runpy.run_path(str(script_path), run_name="__main__")


def print_plan(config: PipelineConfig, steps: Sequence[RunnableStep]) -> None:
    print(f"Pipeline: {config.task}/{config.dataset}")
    print(f"Workdir : {config.workdir}")
    for note in config.notes:
        print(f"Note    : {note}")
    print("Flow    : one canonical task flow with dataset-specific script bindings")
    print("Steps   :")
    for idx, step in enumerate(steps, start=1):
        flags = []
        if step.llm:
            flags.append("LLM")
        if step.optional:
            flags.append("optional")
        suffix = f" [{' / '.join(flags)}]" if flags else ""
        print(f"  {idx:02d}. {step.name}{suffix}")
        for script in step.implementation_paths():
            print(f"      implementation: {script}")
        if step.description:
            print(f"      # {step.description}")


def run_pipeline(config: PipelineConfig, steps: Sequence[RunnableStep], *, dry_run: bool) -> None:
    problems = [m for m in validate_pipeline(config, steps) if not m.startswith("optional missing:")]
    if problems:
        raise PipelineError("Pipeline validation failed:\n" + "\n".join(f"- {m}" for m in problems))

    print_plan(config, steps)
    if dry_run:
        print("\n[DRY-RUN] No commands executed.")
        return

    for step in steps:
        for script in step.implementation_paths():
            script_path = config.workdir / script
            if step.optional and not script_path.exists():
                print(f"\n[SKIP optional missing] {step.name}: {script_path}")
                continue
            print(f"\n[RUN] {config.task}/{config.dataset}:{step.name} -> {script}", flush=True)
            execute_stage_implementation(script_path, config.workdir)


def add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", choices=sorted(CANONICAL_FLOWS), required=True)
    parser.add_argument("--dataset", required=True, help="Dataset name, or 'all' for every dataset configured for the task")
    parser.add_argument("--from-step", default=None, help="Start at this logical step name")
    parser.add_argument("--to-step", default=None, help="Stop after this logical step name")
    parser.add_argument("--only-step", action="append", default=[], help="Run/print only this step; can be repeated")
    parser.add_argument("--skip-step", action="append", default=[], help="Skip this step; can be repeated")
    parser.add_argument("--skip-llm", action="store_true", help="Exclude LLM/API-dependent steps from the selected plan")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified runner for the canonical Dclean dataset flows")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List supported unified flows and datasets")

    plan = sub.add_parser("plan", help="Print the selected flow without executing it")
    add_selection_args(plan)
    plan.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    validate = sub.add_parser("validate", help="Validate that scripts referenced by a flow exist")
    add_selection_args(validate)

    run = sub.add_parser("run", help="Execute the selected flow")
    add_selection_args(run)
    run.add_argument("--dry-run", action="store_true", help="Print commands without executing them")

    return parser.parse_args(argv)


def selected_pipelines(args: argparse.Namespace) -> list[tuple[PipelineConfig, tuple[RunnableStep, ...]]]:
    pipelines = []
    for dataset in expand_datasets(args.task, args.dataset):
        config = build_pipeline(args.task, dataset)
        steps = slice_steps(
            config.steps,
            from_step=args.from_step,
            to_step=args.to_step,
            only_steps=args.only_step,
            skip_steps=args.skip_step,
            include_llm=not args.skip_llm,
        )
        pipelines.append((config, steps))
    return pipelines


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        if args.command == "list":
            for task, datasets in available_tasks().items():
                print(f"{task}: {', '.join(datasets)}")
            return 0

        pipelines = selected_pipelines(args)

        if args.command == "plan":
            if args.json:
                payloads = [pipeline_payload(config, steps) for config, steps in pipelines]
                print(json.dumps(payloads, ensure_ascii=False, indent=2))
            else:
                for index, (config, steps) in enumerate(pipelines, start=1):
                    if index > 1:
                        print()
                    print_plan(config, steps)
            return 0

        if args.command == "validate":
            failed = False
            for config, steps in pipelines:
                messages = validate_pipeline(config, steps)
                errors = [m for m in messages if not m.startswith("optional missing:")]
                if messages:
                    print(f"{config.task}/{config.dataset}:")
                    for msg in messages:
                        print(f"  {msg}")
                if errors:
                    failed = True
                else:
                    print(f"OK: {config.task}/{config.dataset} ({len(steps)} selected steps)")
            return 1 if failed else 0

        if args.command == "run":
            for config, steps in pipelines:
                run_pipeline(config, steps, dry_run=args.dry_run)
                if config is not pipelines[-1][0]:
                    print()
            return 0

    except PipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    raise AssertionError(f"Unhandled command: {args.command}")


def pipeline_payload(config: PipelineConfig, steps: Sequence[RunnableStep]) -> dict[str, object]:
    return {
        "task": config.task,
        "dataset": config.dataset,
        "workdir": str(config.workdir),
        "notes": list(config.notes),
        "flow": [stage.name for stage in CANONICAL_FLOWS[config.task]],
        "steps": [
            {
                "name": step.name,
                "implementations": list(step.scripts),
                "description": step.description,
                "llm": step.llm,
                "optional": step.optional,
            }
            for step in steps
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
