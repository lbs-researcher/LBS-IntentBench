#!/usr/bin/env python3
"""
LBS-IntentBench Evaluation Runner
=================================

Dispatches to the per-subtask ``evaluate`` functions under
``evaluation.task3_gmt``.  Predictions must be in JSONL format.

Usage
-----
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

TASK3_SUBTASKS = (
    "poi_semantic_understanding",
    "trajectory_fact_retrieval",
    "next_step_mobility_prediction",
    "location_time_reasoning",
    "mobility_preference_summarization",
    "mobility_reason_inference",
    "counterfactual_anomaly_detection",
)

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
    parser.add_argument("--task", required=True, choices=["task3_gmt"],
                        help="(Currently only task3_gmt is implemented.)")
    parser.add_argument("--subtask", default=None, choices=TASK3_SUBTASKS,
                        help="Required for --task task3_gmt.")
    parser.add_argument("--direction", default=None,
                        choices=["forward", "backward"],
                        help="Required for poi_semantic_understanding.")
    parser.add_argument("--predictions", required=True,
                        help="Submission JSONL file.")
    parser.add_argument("--ground-truth", required=True,
                        help="Gold CSV file.")
    args = parser.parse_args()

    if args.task == "task3_gmt":
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
