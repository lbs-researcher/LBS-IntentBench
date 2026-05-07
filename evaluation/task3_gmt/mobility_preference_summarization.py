"""
Task 3.5: Mobility Preference Summarization Evaluation
======================================================

Two open-ended questions about the user's travel-mode habits at home:

    Q1: most-used travel mode overall.            (gold = q1_top1_mode)
    Q2: most-used travel mode toward {dest_type}. (gold = q2_answer)

Submission JSONL schema (one record per CSV row):
    {"sample_id": "user_000001", "answer1": "公共交通", "answer2": "公共交通"}

CSV gold field naming (heads-up — the names look mismatched but are
intentional):

    q1_top1_mode      -> single-mode gold answer for Q1 (e.g. "公共交通")
    q1_ground_truth   -> raw distribution string for Q1, kept for auditing
                         only (e.g. "步行(20%)，公共交通(80%)"). NOT used
                         for scoring.
    q2_answer         -> single-mode gold answer for Q2 (already collapsed
                         to one travel mode per sample)
    q2_ground_truth   -> raw per-destination-type distribution string for
                         Q2, kept for auditing only. NOT used for scoring.

In other words, the two ``*_ground_truth`` columns are reference distributions
held out for human inspection; the actual gold labels for scoring are
``q1_top1_mode`` and ``q2_answer``.

Returned metrics (Table 3 columns)
----------------------------------
- ``overall_pref_acc``  : Acc (Overall Pref.)
- ``category_pref_acc`` : Acc (Category Pref.)

Travel-mode synonyms are normalised before comparison
(e.g. "打车" / "出租车" -> "驾车", "公交" / "地铁" -> "公共交通").
"""
from __future__ import annotations

from evaluation.task3_gmt._common import (
    align,
    load_csv,
    load_jsonl,
    percent,
)

# ---------------------------------------------------------------------------
# Travel-mode normalisation (mirrors rec_judge/eval_trajectory_preference_v2.py)
# ---------------------------------------------------------------------------

# Canonical travel-mode set (mirrors rec_judge ``VALID_MODES``).
VALID_MODES = {"驾车", "步行", "公共交通", "骑行", "摩托车"}

# Synonym -> canonical mapping (verbatim copy of rec_judge ``MODE_ALIASES``).
MODE_ALIASES = {
    "打车": "驾车", "出租车": "驾车", "taxi": "驾车", "开车": "驾车",
    "自驾": "驾车", "car": "驾车", "出行车": "驾车", "出行交通": "公共交通",
    "公交": "公共交通", "地铁": "公共交通", "bus": "公共交通",
    "walk": "步行", "bike": "骑行", "ride": "骑行", "motor": "摩托车",
}


def normalize_travel_mode(value) -> str:
    """Strip + alias lookup, falling back to the raw stripped string.

    Verbatim port of ``rec_judge.eval_trajectory_preference_v2.normalize_mode``.
    """
    if not value:
        return ""
    stripped = str(value).strip()
    return MODE_ALIASES.get(stripped, stripped)


def evaluate(predictions_file: str, ground_truth_file: str) -> dict:
    preds = load_jsonl(predictions_file)
    gt = load_csv(ground_truth_file)
    pairs, audit = align(preds, gt)
    n = len(pairs)

    q1_ok = q2_ok = 0
    for pred, g in pairs:
        if normalize_travel_mode(pred.get("answer1")) == normalize_travel_mode(g.get("q1_top1_mode")):
            q1_ok += 1
        if normalize_travel_mode(pred.get("answer2")) == normalize_travel_mode(g.get("q2_answer")):
            q2_ok += 1

    return {
        "overall_pref_acc":  percent(q1_ok, n),
        "category_pref_acc": percent(q2_ok, n),
        "num_samples":       n,
        "_audit":            audit,
    }
