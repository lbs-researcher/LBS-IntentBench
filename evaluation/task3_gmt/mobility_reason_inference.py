"""
Task 3.6: Mobility Reason Inference Evaluation
==============================================

A multi-label classification task over the candidate reason set
{A=commuting, B=short-term intent, C=frequently visited, D=exploration}.

Submission JSONL schema (one record per CSV row):
    {"sample_id": "user_000001", "answer": "BC"}
The value may also be a list like ["B", "C"].

Returned metrics (Table 3 column)
---------------------------------
- ``exact_match`` : EM (Multi-choice) — predicted option set must equal the
                    gold option set (order-insensitive).
"""
from __future__ import annotations

from evaluation.task3_gmt._common import (
    align,
    load_csv,
    load_jsonl,
    norm_letter_set,
    percent,
)


def evaluate(predictions_file: str, ground_truth_file: str) -> dict:
    preds = load_jsonl(predictions_file)
    gt = load_csv(ground_truth_file)
    pairs, audit = align(preds, gt)
    n = len(pairs)

    em = 0
    for pred, g in pairs:
        p = norm_letter_set(pred.get("answer"))
        gold = norm_letter_set(g.get("reason_answer"))
        if p == gold:
            em += 1

    return {
        "exact_match": percent(em, n),
        "num_samples": n,
        "_audit":      audit,
    }
