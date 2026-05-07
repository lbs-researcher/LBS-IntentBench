"""
Task 3.7: Counterfactual Anomaly Detection Evaluation
=====================================================

Given two trajectory snippets, the model must:
    1. pick the more-reasonable snippet ("片段1" or "片段2");
    2. localise the anomalous record in the *unreasonable* snippet by
       reporting its ``seq_no``.

Submission JSONL schema (one record per CSV row):
    {"sample_id": "user_000001",
     "more_reasonable": "片段1",
     "anomaly_seq_no": 9}

``anomaly_seq_no`` may also be a list (``[9]`` / ``[9, 10]``) or a comma /
slash-separated string (``"9"`` / ``"9,10"``).  Localisation is counted as
correct as long as the gold ``modified_seq_no`` appears anywhere in the
predicted set, since models commonly report a small range around the
suspected anomaly.

Returned metrics (Table 3 columns)
----------------------------------
- ``plausibility_acc`` : Acc (Plausibility)  — picked the truly more-reasonable
                         snippet.
- ``anomaly_loc_acc``  : Loc Acc (Anomaly Loc.) — conditioned on a correct
                         plausibility pick, did the model also report the
                         correct seq_no?  (A wrong plausibility pick
                         automatically counts as a wrong localisation.)
"""
from __future__ import annotations

import re

from evaluation.task3_gmt._common import (
    align,
    load_csv,
    load_jsonl,
    norm_text,
    percent,
)


def _norm_snippet_label(s) -> str:
    """Map any of '片段1'/'snippet1'/'1'/1 -> '1' (or '2')."""
    s = str(s or "").strip()
    if not s:
        return ""
    if s in ("片段1", "snippet1", "snippet_1", "1"):
        return "1"
    if s in ("片段2", "snippet2", "snippet_2", "2"):
        return "2"
    return s[-1] if s and s[-1] in ("1", "2") else s


def _coerce_seq_set(value) -> set[int]:
    """Normalise an ``anomaly_seq_no`` payload to a set of ints.

    Accepts: ``9``, ``"9"``, ``[9, 10]``, ``"9, 10"``, ``"9;10"``, ``"9/10"``.
    Anything that fails to parse is silently dropped, mirroring the
    permissive style of the other Task 3 evaluators.
    """
    if value is None:
        return set()
    if isinstance(value, bool):  # bool is an int subclass; reject it explicitly.
        return set()
    if isinstance(value, int):
        return {value}
    if isinstance(value, (list, tuple, set)):
        out: set[int] = set()
        for v in value:
            out.update(_coerce_seq_set(v))
        return out
    s = str(value).strip()
    if not s:
        return set()
    out = set()
    for tok in re.split(r"[,;/\s]+", s):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.add(int(tok))
        except ValueError:
            continue
    return out

def _gold_unreasonable_label(gt: dict) -> str:
    """Return '1' or '2' depending on which snippet is the unreasonable one."""
    a1 = norm_text(gt.get("answer_1"))
    a2 = norm_text(gt.get("answer_2"))
    if "不合理" in a1 or "unreasonable" in a1:
        return "1"
    if "不合理" in a2 or "unreasonable" in a2:
        return "2"
    return ""


def evaluate(predictions_file: str, ground_truth_file: str) -> dict:
    preds = load_jsonl(predictions_file)
    gt = load_csv(ground_truth_file)
    pairs, audit = align(preds, gt)
    n = len(pairs)

    plaus_ok = loc_ok = 0
    for pred, g in pairs:
        gold_unreason = _gold_unreasonable_label(g)
        gold_reason = "1" if gold_unreason == "2" else ("2" if gold_unreason == "1" else "")
        pred_reason = _norm_snippet_label(pred.get("more_reasonable"))

        plaus_correct = bool(gold_reason) and pred_reason == gold_reason
        if plaus_correct:
            plaus_ok += 1

        pred_seqs = _coerce_seq_set(pred.get("anomaly_seq_no"))
        try:
            gold_seq = int(str(g.get("modified_seq_no", "")).strip())
        except (TypeError, ValueError):
            gold_seq = None
        if plaus_correct and gold_seq is not None and gold_seq in pred_seqs:
            loc_ok += 1

    return {
        "plausibility_acc": percent(plaus_ok, n),
        "anomaly_loc_acc":  percent(loc_ok, n),
        "num_samples":      n,
        "_audit":           audit,
    }
