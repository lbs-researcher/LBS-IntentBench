"""
Task 1: Mobility Intent Inference (MII) Evaluation

This module evaluates models on the Mobility Intent Inference task, which requires
ranking candidate intent scenarios by likelihood given user profile, behavior history,
and spatio-temporal context.

Metrics:
    - Exact Match (%): Full candidate ranking must be correct.
    - Top-1 Accuracy (%): Whether the highest-priority intent is correctly identified.

Submission Format (JSONL):
    JSONL file with one object per sample:
        {"sample_id": "mii_001", "predicted_ranking": "2,1,3,4"}
    Example:
        {"sample_id": "mii_001", "predicted_ranking": "2,1,3,4"}

Reference Data (CSV):
    The published benchmark CSV uses id, content, answer. The evaluator also
    accepts expanded columns sample_id and ground_truth_ranking.
"""

import csv
import json
import re
import warnings

_LETTER_TO_NUM = {chr(ord("A") + i): str(i + 1) for i in range(26)}


def _normalize_token(token: str) -> str:
    token = token.strip().upper()
    return _LETTER_TO_NUM.get(token, token)


def _parse_ranking(ranking_str: str) -> list[str]:
    if isinstance(ranking_str, (list, tuple)):
        tokens = ranking_str
    else:
        text = str(ranking_str).strip()
        if re.fullmatch(r"[A-Za-z]+", text) and len(text) > 1:
            tokens = list(text)
        else:
            tokens = re.split(r"\s*(?:>|,|，)\s*|\s+", text)
    return [_normalize_token(str(t)) for t in tokens if str(t).strip()]


def _load_predictions(predictions_file: str) -> list[dict]:
    rows = []
    with open(predictions_file, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{predictions_file}:{line_no} is not valid JSONL: {exc}"
                ) from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{predictions_file}:{line_no} must be a JSON object.")

            sample_id = str(obj.get("sample_id") or "").strip()
            if not sample_id:
                raise ValueError(f"{predictions_file}:{line_no} missing sample_id.")
            if "predicted_ranking" not in obj:
                raise ValueError(
                    f"{predictions_file}:{line_no} missing predicted_ranking."
                )
            rows.append(
                {
                    "sample_id": sample_id,
                    "predicted_ranking": obj["predicted_ranking"],
                }
            )
    return rows


def _first_present(row: dict, keys: tuple[str, ...]):
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def _load_ground_truth(ground_truth_file: str) -> list[dict]:
    with open(ground_truth_file, "r", encoding="utf-8-sig", newline="") as f:
        rows = []
        for line_no, row in enumerate(csv.DictReader(f), 2):
            sample_id = _first_present(row, ("sample_id", "id"))
            ranking = _first_present(row, ("ground_truth_ranking", "answer"))
            if sample_id is None:
                raise ValueError(f"{ground_truth_file}:{line_no} missing sample_id/id.")
            if ranking is None:
                raise ValueError(
                    f"{ground_truth_file}:{line_no} missing ground_truth_ranking/answer."
                )
            rows.append(
                {
                    **row,
                    "sample_id": str(sample_id).strip(),
                    "ground_truth_ranking": ranking,
                }
            )
        return rows


def _align_predictions(predictions: list[dict], ground_truth: list[dict]) -> list[dict]:
    gt_by_id = {
        str(row.get("sample_id", "")).strip(): row
        for row in ground_truth
        if str(row.get("sample_id", "")).strip()
    }
    merged = []
    for pred in predictions:
        gt = gt_by_id.get(pred["sample_id"])
        if gt is not None:
            merged.append({**gt, **pred})
    return merged


def evaluate(predictions_file: str, ground_truth_file: str) -> dict:
    """
    Evaluate predictions for the Mobility Intent Inference task.

    Args:
        predictions_file: Path to the predictions JSONL file.
        ground_truth_file: Path to the ground truth CSV file.

    Returns:
        A dictionary of evaluation metrics, e.g.:
        {
            "exact_match": float,
            "top1_accuracy": float
        }
    """
    predictions = _load_predictions(predictions_file)
    ground_truth = _load_ground_truth(ground_truth_file)

    merged = _align_predictions(predictions, ground_truth)
    if len(merged) == 0:
        warnings.warn("No matching sample_id found between predictions and ground truth.")
        return {}
    if len(merged) < len(ground_truth):
        warnings.warn(
            f"Only {len(merged)}/{len(ground_truth)} ground truth samples matched in predictions."
        )

    for row in merged:
        row["pred_ranking"] = _parse_ranking(row["predicted_ranking"])
        row["gt_ranking"] = _parse_ranking(row["ground_truth_ranking"])

    n = len(merged)

    exact_match = sum(
        1 for r in merged if r["pred_ranking"] == r["gt_ranking"]
    ) / n

    top1_accuracy = sum(
        1
        for r in merged
        if r["pred_ranking"]
        and r["gt_ranking"]
        and r["pred_ranking"][0] == r["gt_ranking"][0]
    ) / n

    results = {
        "exact_match": round(exact_match, 4),
        "top1_accuracy": round(top1_accuracy, 4),
    }

    return results
