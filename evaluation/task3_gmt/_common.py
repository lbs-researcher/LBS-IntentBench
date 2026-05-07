"""
Tiny shared helpers for Task 3 evaluators.

Kept intentionally small: only IO and text/letter normalization that every
subtask needs.  Anything subtask-specific lives in the subtask's own .py
file so that each evaluator stays self-contained and easy to read.
"""
from __future__ import annotations

import csv
import json
import re
from typing import Any, Iterable

csv.field_size_limit(10 ** 8)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    """Read a JSONL file. Empty / commented lines are skipped silently."""
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln} invalid JSON line: {e}") from e
    return rows


def load_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def index_by(records: Iterable[dict], key: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in records:
        k = str(r.get(key, "")).strip()
        if k:
            out[k] = r
    return out


def align(predictions: list[dict], ground_truth: list[dict],
          pred_key: str = "sample_id", gt_key: str = "id"
          ) -> tuple[list[tuple[dict, dict]], dict]:
    """
    Pair every prediction with its ground-truth record by sample id.

    Returns (paired_list, audit). Predictions missing a GT match are
    excluded; an audit summary is returned for transparency.
    """
    gt_idx = index_by(ground_truth, gt_key)
    paired: list[tuple[dict, dict]] = []
    miss_gt: list[str] = []
    for p in predictions:
        sid = str(p.get(pred_key, "")).strip()
        if sid in gt_idx:
            paired.append((p, gt_idx[sid]))
        else:
            miss_gt.append(sid)
    pred_ids = {str(p.get(pred_key, "")).strip() for p in predictions}
    miss_pred = [k for k in gt_idx if k not in pred_ids]
    return paired, {
        "num_predictions": len(predictions),
        "num_ground_truth": len(gt_idx),
        "num_aligned": len(paired),
        "num_missing_in_gt": len(miss_gt),
        "num_missing_in_predictions": len(miss_pred),
        "missing_in_gt_examples": miss_gt[:5],
        "missing_in_predictions_examples": miss_pred[:5],
    }


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def norm_text(s: Any) -> str:
    """Strip whitespace, lowercase. Used for case/space-insensitive equality."""
    if s is None:
        return ""
    return _WS_RE.sub("", str(s).strip()).lower()


def norm_letter(s: Any) -> str:
    return str(s or "").strip().upper()


def norm_letter_set(s: Any) -> set[str]:
    """
    Normalise a multi-select option payload to a set of upper-case letters.
    Accepts: "AC", "A,C", ["A","C"], "ac", etc.
    """
    if s is None:
        return set()
    if isinstance(s, (list, tuple, set)):
        return {norm_letter(x) for x in s if str(x).strip()}
    return set(re.sub(r"[^A-Za-z]", "", str(s)).upper())


def percent(n: int, d: int) -> float:
    return round(n / d * 100, 4) if d else 0.0
