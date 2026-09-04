"""
Task 2: Contextual Constraint Inference (CCI) Evaluation

This module evaluates models on the Contextual Constraint Inference task, which
requires identifying behavioral constraints and decision logic from candidates
given user profile, behavior history, and spatio-temporal triggers.

The task includes both single-choice and multiple-choice questions.

Metrics:
    - Overall Accuracy (%): Sample-count-weighted average of single and multi accuracy.
    - Single-choice Accuracy (%): Accuracy on single-choice questions.
    - Multi-choice Accuracy (%): Exact match accuracy on multiple-choice questions.

Submission Format (JSONL):
    JSONL file with one object per sample:
        {"sample_id": "cci_001", "predicted_options": "A,C"}
    Example:
        {"sample_id": "cci_001", "predicted_options": "A,C"}

Reference Data (CSV):
    The published benchmark CSV uses id, content, answer. The evaluator also
    accepts expanded columns sample_id, ground_truth_options, and question_type.
"""

import csv
import json
import re
import warnings


def _parse_options(options_str: str) -> set[str]:
    if isinstance(options_str, (list, tuple, set)):
        tokens = options_str
    else:
        text = str(options_str).strip()
        tokens = re.split(r"[,，\s]+", text) if re.search(r"[,，\s]", text) else [text]

    parsed = set()
    for token in tokens:
        cleaned = str(token).strip().upper()
        if not cleaned:
            continue
        if re.fullmatch(r"[A-Z]+", cleaned):
            parsed.update(cleaned)
        else:
            parsed.add(cleaned)
    return parsed


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
            if "predicted_options" not in obj:
                raise ValueError(
                    f"{predictions_file}:{line_no} missing predicted_options."
                )
            rows.append(
                {
                    "sample_id": sample_id,
                    "predicted_options": obj["predicted_options"],
                }
            )
    return rows


def _first_present(row: dict, keys: tuple[str, ...]):
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def _question_type_from_content(raw) -> str:
    if raw is None or raw == "":
        return ""
    try:
        content = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(content, dict):
        return ""
    for key in ("question_format", "题型", "问题类型"):
        value = content.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _normalize_question_type(raw) -> str:
    text = str(raw or "").strip().lower()
    if text in {"single", "single-choice", "single_choice", "单选", "单项选择"}:
        return "single"
    if text in {
        "multi",
        "multiple",
        "multiple-choice",
        "multiple_choice",
        "多选",
        "多项选择",
    }:
        return "multi"
    return text


def _load_ground_truth(ground_truth_file: str) -> list[dict]:
    with open(ground_truth_file, "r", encoding="utf-8-sig", newline="") as f:
        rows = []
        for line_no, row in enumerate(csv.DictReader(f), 2):
            sample_id = _first_present(row, ("sample_id", "id"))
            options = _first_present(row, ("ground_truth_options", "answer"))
            question_type = _first_present(row, ("question_type", "question_format"))
            if question_type is None:
                question_type = _question_type_from_content(row.get("content"))
            if sample_id is None:
                raise ValueError(f"{ground_truth_file}:{line_no} missing sample_id/id.")
            if options is None:
                raise ValueError(
                    f"{ground_truth_file}:{line_no} missing ground_truth_options/answer."
                )
            rows.append(
                {
                    **row,
                    "sample_id": str(sample_id).strip(),
                    "ground_truth_options": options,
                    "question_type": question_type or "",
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
    Evaluate predictions for the Contextual Constraint Inference task.

    Args:
        predictions_file: Path to the predictions JSONL file.
        ground_truth_file: Path to the ground truth CSV file.

    Returns:
        A dictionary of evaluation metrics, e.g.:
        {
            "overall_accuracy": float,
            "single_accuracy": float,
            "multi_accuracy": float
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
        row["pred_options"] = _parse_options(row["predicted_options"])
        row["gt_options"] = _parse_options(row["ground_truth_options"])
        row["is_correct"] = row["pred_options"] == row["gt_options"]

    single_rows = [
        row
        for row in merged
        if _normalize_question_type(row.get("question_type")) == "single"
    ]
    multi_rows = [
        row
        for row in merged
        if _normalize_question_type(row.get("question_type")) == "multi"
    ]

    n_single = len(single_rows)
    n_multi = len(multi_rows)

    single_accuracy = (
        sum(1 for row in single_rows if row["is_correct"]) / n_single
        if n_single > 0
        else None
    )
    multi_accuracy = (
        sum(1 for row in multi_rows if row["is_correct"]) / n_multi
        if n_multi > 0
        else None
    )

    if single_accuracy is not None and multi_accuracy is not None:
        overall_accuracy = (single_accuracy * n_single + multi_accuracy * n_multi) / (
            n_single + n_multi
        )
    elif single_accuracy is not None:
        overall_accuracy = single_accuracy
    elif multi_accuracy is not None:
        overall_accuracy = multi_accuracy
    else:
        overall_accuracy = None

    results = {
        "overall_accuracy": overall_accuracy,
        "single_accuracy": single_accuracy,
        "multi_accuracy": multi_accuracy,
    }

    return {k: round(v, 4) if v is not None else None for k, v in results.items()}
