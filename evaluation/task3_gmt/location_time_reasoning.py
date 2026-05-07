"""
Task 3.4: Location-Time Reasoning Evaluation
============================================

Two 4-choice multiple-choice questions:

    Q1: at a given future timestamp, where will the user be?  (poi_answer)
    Q2: when will the user next arrive at a given POI?         (time_answer)

Submission JSONL schema (one record per CSV row):
    {"sample_id": "user_000001", "answer1": "B", "answer2": "B"}

Returned metrics (Table 3 columns)
----------------------------------
- ``loc_pred_acc``  : Acc (Loc. Pred.)  — exact letter match on Q1.
- ``time_pred_acc`` : Acc (Time Pred.)  — exact letter match on Q2.
"""
from __future__ import annotations

from evaluation.task3_gmt._common import (
    align,
    load_csv,
    load_jsonl,
    norm_letter,
    percent,
)


def evaluate(predictions_file: str, ground_truth_file: str) -> dict:
    preds = load_jsonl(predictions_file)
    gt = load_csv(ground_truth_file)
    pairs, audit = align(preds, gt)
    n = len(pairs)

    loc_ok = time_ok = 0
    for pred, g in pairs:
        if norm_letter(pred.get("answer1")) == norm_letter(g.get("poi_answer")):
            loc_ok += 1
        if norm_letter(pred.get("answer2")) == norm_letter(g.get("time_answer")):
            time_ok += 1

    return {
        "loc_pred_acc":  percent(loc_ok, n),
        "time_pred_acc": percent(time_ok, n),
        "num_samples":   n,
        "_audit":        audit,
    }
