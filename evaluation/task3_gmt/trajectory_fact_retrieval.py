"""
Task 3.2: Trajectory Fact Retrieval Evaluation
==============================================

Three QA questions are asked over a long, time-ordered trajectory:

    Q1 (Time Localization)  -> POI name where the user was at q1_query_time
    Q2 (Stay Segment)       -> all stay intervals at q2_query_poi, as a list
                               of {start, end} dicts
    Q3 (Interest POI)       -> {poi: [...], count: int}

Submission JSONL schema (one record per CSV row):
    {"sample_id": "user_000001",
     "answer1": "餐饮_0009",
     "answer2": [{"start": "2026-03-15 12:52:20", "end": "2026-03-15 12:56:50"}],
     "answer3": {"poi": ["住宅区_0001"], "count": 11}}

Returned metrics (Table 3 columns)
----------------------------------
- ``q1_acc``       : Acc (Time Loc.)         — exact-match POI name
- ``q2_acc``       : Acc (Stay Seg.)         — set equality of {start, end} pairs
- ``q3_poi_acc``   : POI set equality
- ``q3_count_acc`` : count integer equality
- ``q3_joint_acc`` : Joint Acc (Interest POI) — both POI set AND count correct
"""
from __future__ import annotations

import json
import re

from evaluation.task3_gmt._common import (
    align,
    load_csv,
    load_jsonl,
    percent,
)


# ---------------------------------------------------------------------------
# Normalization helpers (mirrors rec_judge/eval_retrieval_v2_all.py)
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _strip_all_ws(s) -> str:
    """rec_judge Q1 rule: drop *every* whitespace char before equality."""
    return _WS_RE.sub("", str(s or ""))


def _normalize_poi_set(value) -> set[str]:
    """
    Mirrors rec_judge ``normalize_poi_set``:
        list  -> {p.strip() for p in value if p.strip()}
        str   -> split by ','  (some old GTs encode the set as "a,b,c")
        else  -> empty set
    """
    if isinstance(value, list):
        return {str(p).strip() for p in value if p and str(p).strip()}
    if isinstance(value, str):
        return {p.strip() for p in value.split(",") if p.strip()}
    return set()


def _parse_intervals(raw) -> list[tuple[str, str]]:
    """Convert a JSON-encoded list of {start, end} dicts to a list of tuples."""
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
    out: list[tuple[str, str]] = []
    if isinstance(raw, list):
        for it in raw:
            if isinstance(it, dict):
                s = str(it.get("start", "")).strip()
                e = str(it.get("end", "")).strip()
                if s or e:
                    out.append((s, e))
    return out


def evaluate(predictions_file: str, ground_truth_file: str) -> dict:
    preds = load_jsonl(predictions_file)
    gt = load_csv(ground_truth_file)
    pairs, audit = align(preds, gt)
    n = len(pairs)

    q1_correct = 0
    q2_correct = 0
    q3_poi_correct = 0
    q3_cnt_correct = 0
    q3_joint_correct = 0

    for pred, g in pairs:
        # --- Q1: Time Localization (whitespace-insensitive equality) ---
        if _strip_all_ws(pred.get("answer1")) == _strip_all_ws(g.get("q1_ground_truth")):
            q1_correct += 1

        # --- Q2: Stay Segment (every gold interval hit AND equal count) ---
        pred_intervals = _parse_intervals(pred.get("answer2"))
        gold_intervals = _parse_intervals(g.get("q2_ground_truth"))
        if gold_intervals:
            hits = sum(1 for gs, ge in gold_intervals
                       if any(gs == ps and ge == pe for ps, pe in pred_intervals))
            if hits == len(gold_intervals) and len(pred_intervals) == len(gold_intervals):
                q2_correct += 1

        # --- Q3: Interest POI (poi set + count) ---
        q3 = pred.get("answer3") or {}
        pred_pois = _normalize_poi_set(q3.get("poi"))
        gold_pois = _normalize_poi_set(g.get("q4_ground_truth_poi"))
        try:
            pred_cnt = int(q3.get("count"))
        except (TypeError, ValueError):
            pred_cnt = None
        try:
            gold_cnt = int(str(g.get("q4_ground_truth_cnt", "")).strip())
        except (TypeError, ValueError):
            gold_cnt = None

        poi_ok = bool(gold_pois) and pred_pois == gold_pois
        cnt_ok = pred_cnt is not None and gold_cnt is not None and pred_cnt == gold_cnt
        if poi_ok:
            q3_poi_correct += 1
        if cnt_ok:
            q3_cnt_correct += 1
        if poi_ok and cnt_ok:
            q3_joint_correct += 1

    return {
        "q1_acc":       percent(q1_correct, n),
        "q2_acc":       percent(q2_correct, n),
        "q3_poi_acc":   percent(q3_poi_correct, n),
        "q3_count_acc": percent(q3_cnt_correct, n),
        "q3_joint_acc": percent(q3_joint_correct, n),
        "num_samples":  n,
        "_audit":       audit,
    }
