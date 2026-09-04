#!/usr/bin/env python3
"""
LBS-IntentBench Evaluation Runner
=================================

Dispatches to the per-task ``evaluate`` functions. Prediction files use JSONL
for all tasks.

Usage
-----
  python scripts/run_evaluation.py \\
      --task task1_mii \\
      --predictions preds.jsonl \\
      --ground-truth data/task1_mii/mobility_intent_inference.csv

  python scripts/run_evaluation.py \\
      --task task2_cci \\
      --predictions preds.jsonl \\
      --ground-truth data/task2_cci/contextual_constraint_inference.csv

  python scripts/run_evaluation.py \\
      --task task3_gmt \\
      --subtask next_step_mobility_prediction \\
      --predictions preds.jsonl \\
      --ground-truth data/task3_gmt/next_step_mobility_prediction.csv

  # POI Semantic Understanding takes a --direction (forward|backward)
  python scripts/run_evaluation.py \\
      --task task3_gmt \\
      --subtask poi_semantic_understanding \\
      --direction forward \\
      --predictions preds_fwd.jsonl \\
      --ground-truth data/task3_gmt/poi_semantic_understanding.csv
"""
import argparse
import importlib
import json
import os
import sys

# Make the project root importable when the script is invoked directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

TASK12_MODULES = {
    "task1_mii": "evaluation.task1_mii",
    "task2_cci": "evaluation.task2_cci",
}

TASK3_SUBTASKS = (
    "poi_semantic_understanding",
    "trajectory_fact_retrieval",
    "next_step_mobility_prediction",
    "location_time_reasoning",
    "mobility_preference_summarization",
    "mobility_reason_inference",
    "counterfactual_anomaly_detection",
)

TASK_CHOICES = tuple(TASK12_MODULES) + ("task3_gmt",)


def _run_task12(task: str, predictions: str, ground_truth: str):
    mod = importlib.import_module(TASK12_MODULES[task])
    return mod.evaluate(predictions, ground_truth)


def _run_task3(subtask: str, predictions: str, ground_truth: str,
               direction):
    mod = importlib.import_module(f"evaluation.task3_gmt.{subtask}")
    if subtask == "poi_semantic_understanding":
        if direction is None:
            raise SystemExit("--direction is required for poi_semantic_understanding")
        return mod.evaluate(predictions, ground_truth, direction=direction)
    return mod.evaluate(predictions, ground_truth)

def main() -> int:
    parser = argparse.ArgumentParser(
        description="LBS-IntentBench Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--task", required=True, choices=TASK_CHOICES,
                        help="Task to evaluate.")
    parser.add_argument("--subtask", default=None, choices=TASK3_SUBTASKS,
                        help="Required for --task task3_gmt.")
    parser.add_argument("--direction", default=None,
                        choices=["forward", "backward"],
                        help="Required for poi_semantic_understanding.")
    parser.add_argument("--predictions", required=True,
                        help="Submission JSONL file.")
    parser.add_argument("--ground-truth", required=True,
                        help="Benchmark CSV file with reference answers.")
    args = parser.parse_args()

    if args.task in TASK12_MODULES:
        if args.subtask is not None:
            parser.error("--subtask is only valid for task3_gmt")
        if args.direction is not None:
            parser.error("--direction is only valid for poi_semantic_understanding")
        metrics = _run_task12(args.task, args.predictions, args.ground_truth)
    elif args.task == "task3_gmt":
        if args.subtask is None:
            parser.error("--subtask is required for task3_gmt")
        metrics = _run_task3(args.subtask, args.predictions,
                             args.ground_truth, args.direction)
    else:  # pragma: no cover - guarded by argparse choices.
        parser.error(f"Unsupported task: {args.task}")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())
