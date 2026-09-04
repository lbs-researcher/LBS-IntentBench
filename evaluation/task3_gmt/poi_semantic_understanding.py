"""
Task 3.1: POI Semantic Understanding Evaluation
================================================

This subtask is run **twice** with different prompts and produces two
disjoint sets of metrics that line up with the three columns of the
"POI Semantic Understanding" group in Table 3 of the paper:

    | Location Acc (Region Loc.) | Tag F1 (Pos. Tag) | Hit@5 (Neg. Recog.) |
    |        forward             |     forward       |      backward       |

Therefore the function exposes a ``direction`` flag — please pass
``direction='forward'`` when evaluating the prompt
``poi_semantic_understanding.json`` and ``direction='backward'`` when
evaluating ``poi_name_guess.json``.

Submission JSONL schemas
------------------------
forward (one record per CSV row):
    {"sample_id": "poi_000001",
     "pcdr": "福建省;泉州市;鲤城区;金洲街",
     "tags": ["酒店", "民宿"]}

backward (one record per CSV row):
    {"sample_id": "poi_000001",
     "candidates": ["悦鲤侘寂凤民宿", "...", "...", "...", "..."]}

Returned metrics
----------------
forward  -> {"location_acc": float, "tag_f1": float, "num_samples": int}
backward -> {"hit_at_5": float, "num_samples": int}

All accuracies / F1 are reported as percentages in [0, 100].
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from evaluation.task3_gmt._common import (
    align,
    load_csv,
    load_jsonl,
    percent,
)


# ---------------------------------------------------------------------------
# String helpers (mirrors rec_judge/eval_poi_tasks.py)
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    """Strip parenthesised suffix (Chinese or ASCII) and surrounding spaces."""
    if not name:
        return ""
    return re.sub(r"[（(][^）)]*[）)]", "", str(name)).strip()


def _fuzzy_eq(a: str, b: str) -> bool:
    """Substring containment OR character similarity >= 0.6."""
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.6


def _split_pcdr(pcdr: str) -> tuple[str, str, str]:
    """
    Return the (province, city, district) triple from a pcdr string.

    Canonical format is "province;city;district;street".  When ';' separators
    are missing we recover the three parts via Chinese administrative-suffix
    regexes (covers cases like "云南省大理白族自治州大理市").
    """
    s = (pcdr or "").strip()
    if ";" in s:
        parts = s.split(";")
        return (
            parts[0].strip() if len(parts) > 0 else "",
            parts[1].strip() if len(parts) > 1 else "",
            parts[2].strip() if len(parts) > 2 else "",
        )
    flat = s
    province = city = district = ""
    m = re.match(r"(.*?(?:省|自治区|市))", flat)
    if m:
        province = m.group(1)
        rest = flat[len(province):]
        m2 = re.match(r"(.*?(?:市|地区|自治州|盟))", rest)
        if m2:
            city = m2.group(1)
            district = rest[len(city):]
        else:
            district = rest
    return province, city, district


# ---------------------------------------------------------------------------
# Forward / Backward evaluators
# ---------------------------------------------------------------------------

def _evaluate_forward(pairs: list[tuple[dict, dict]]) -> dict:
    """
    Location Acc (Region Loc.):
        Province ∧ City ∧ District three-level exact match.  The street
        segment is intentionally ignored - the model is only given the POI
        name and cannot reasonably infer the street.

    Tag F1 (Pos. Tag):
        Sample-level F1 averaged over the dataset, using fuzzy tag matching
        (substring containment OR SequenceMatcher >= 0.6) so that synonymous
        categorisations like "幼儿园" vs "学前教育" still match.
    """
    n = len(pairs)
    loc_correct = 0
    f1_sum = 0.0

    for pred, gt in pairs:
        # --- Location: 3-level EM (Region Loc.) ---
        gp, gc, gd = _split_pcdr(gt.get("pcdr", ""))
        pp, pc, pd = _split_pcdr(pred.get("pcdr", ""))
        if pp == gp and pc == gc and pd == gd:
            loc_correct += 1

        # --- Tag F1: per-sample fuzzy precision / recall ---
        pt_raw = pred.get("tags") or []
        if isinstance(pt_raw, str):
            pred_tags = [t.strip() for t in pt_raw.split("|") if t.strip()]
        else:
            pred_tags = [str(t).strip() for t in pt_raw if str(t).strip()]
        gold_tags = [t.strip() for t in str(gt.get("t_tag", "")).split("|") if t.strip()]

        if not pred_tags and not gold_tags:
            f1_sum += 1.0
            continue
        if not pred_tags or not gold_tags:
            continue

        recall_hits = sum(1 for g in gold_tags if any(_fuzzy_eq(p, g) for p in pred_tags))
        prec_hits   = sum(1 for p in pred_tags if any(_fuzzy_eq(p, g) for g in gold_tags))
        recall = recall_hits / len(gold_tags)
        precision = prec_hits / len(pred_tags)
        f1_sum += (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "location_acc": percent(loc_correct, n),
        "tag_f1":       round(f1_sum / n * 100, 4) if n else 0.0,
        "num_samples":  n,
    }


def _evaluate_backward(pairs: list[tuple[dict, dict]]) -> dict:
    """
    Hit@5 (Neg. Recog.):
        Top-5 fuzzy hit using parenthesis-stripped equality, mirroring the
        "HitRate@5 模糊(去括号)" column in rec_judge.  This avoids penalising
        candidates that recover the core POI name but drop store-branch
        qualifiers like "(腊山北路)" that the model has no way to know.
    """
    n = len(pairs)
    hits = 0
    for pred, gt in pairs:
        cands = pred.get("candidates") or []
        if isinstance(cands, str):
            cands = [cands]
        gold = _normalize_name(gt.get("poi_name", ""))
        if not gold:
            continue
        if any(_normalize_name(c) == gold for c in cands[:5]):
            hits += 1
    return {
        "hit_at_5":    percent(hits, n),
        "num_samples": n,
    }


def evaluate(predictions_file: str, ground_truth_file: str,
             direction: str = "forward") -> dict:
    """
    Args:
        predictions_file: JSONL of submissions (see module docstring).
        ground_truth_file: data/task3_gmt/poi_semantic_understanding.csv.
        direction: 'forward' or 'backward'.

    Returns:
        Metrics dict (see module docstring).  Includes a "_audit" sub-dict
        describing how many predictions were aligned with GT.
    """
    if direction not in ("forward", "backward"):
        raise ValueError("direction must be 'forward' or 'backward'")

    preds = load_jsonl(predictions_file)
    gt = load_csv(ground_truth_file)
    pairs, audit = align(preds, gt)

    metrics = (_evaluate_forward if direction == "forward"
               else _evaluate_backward)(pairs)
    metrics["_audit"] = audit
    return metrics
