"""
Task 3.3: Next-Step Mobility Prediction Evaluation
==================================================

Submission JSONL schema (one record per CSV row):
    {"sample_id": "user_000001",
     "answer1": "商店_0001",
     "answer2": "购物;建材;石材"}

Returned metrics (Table 3 columns)
----------------------------------
- ``dest_poi_acc``  : Acc (Dest. POI)  — exact-match on POI name.
- ``dest_type_acc`` : Acc (Dest. Type) — exact-match on the type tag string
                      (whitespace-stripped).  The prompt instructs the model
                      to copy a tag verbatim from the history trajectory, so
                      exact-match is the intended evaluation.
"""
from __future__ import annotations

from evaluation.task3_gmt._common import (
    align,
    load_csv,
    load_jsonl,
    percent,
)


def _strip(s) -> str:
    return str(s or "").strip()


def evaluate(predictions_file: str, ground_truth_file: str) -> dict:
    preds = load_jsonl(predictions_file)
    gt = load_csv(ground_truth_file)
    pairs, audit = align(preds, gt)
    n = len(pairs)

    poi_ok = type_ok = 0
    for pred, g in pairs:
        if _strip(pred.get("answer1")) == _strip(g.get("q4_poi_ground_truth")):
            poi_ok += 1
        if _strip(pred.get("answer2")) == _strip(g.get("q5_tag_ground_truth")):
            type_ok += 1

    return {
        "dest_poi_acc":  percent(poi_ok, n),
        "dest_type_acc": percent(type_ok, n),
        "num_samples":   n,
        "_audit":        audit,
    }
