# LBS-IntentBench Metrics

## Overview

LBS-IntentBench adopts task-specific evaluation metrics to capture different levels of difficulty and aspects of model behavior.

### Task 1: Mobility Intent Inference (MII)

| Metric | Description |
|:-------|:------------|
| Exact Match (%) | Full candidate ranking must be correct |
| Top-1 Accuracy (%) | Whether the highest-priority intent is correctly identified |
| Temporal Misalignment Acc (%) | Accuracy on temporal misalignment distractors |
| Profile Violation Acc (%) | Accuracy on profile constraint violation distractors |
| Deep Conflict Acc (%) | Accuracy on causal inversion distractors |

### Task 2: Contextual Constraint Inference (CCI)

| Metric | Description |
|:-------|:------------|
| Overall Accuracy (%) | Sample-count-weighted average of single and multi accuracy |
| Single-choice Accuracy (%) | Accuracy on single-choice questions |
| Multi-choice Accuracy (%) | Exact match accuracy on multiple-choice questions |
| Deep Conflict Acc (%) | Accuracy on contextual attribution traps |
| Temporal Misalignment Acc (%) | Accuracy on spatio-temporal validity traps |
| Profile Violation Acc (%) | Accuracy on intent-fulfillment confusion traps |

### Task 3: General Mobility Tasks (GMT)

| Subtask | Metrics |
|:--------|:--------|
| POI Semantic Understanding | Location Accuracy, Tag F1, Hit@5 |
| Trajectory Fact Retrieval | Q1 Accuracy, Q2 Accuracy, Q3 Joint Accuracy |
| Next-Step Mobility Prediction | Destination POI Accuracy, Destination Type Accuracy |
| Location-Time Reasoning | Location Prediction Accuracy, Time Prediction Accuracy |
| Mobility Preference Summarization | Overall Preference Accuracy, Category Preference Accuracy |
| Mobility Reason Inference | Exact Match |
| Counterfactual Anomaly Detection | Plausibility Accuracy, Anomaly Localization Accuracy |
