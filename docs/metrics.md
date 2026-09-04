# Metrics

Task 1/2 metrics are fractions in `[0, 1]`; Task 3 metrics are percentages in `[0, 100]`.

## Task 1 — Mobility Intent Inference

| Metric | Meaning |
|:--|:--|
| `exact_match` | Full candidate ranking equals the reference ranking |
| `top1_accuracy` | First-ranked candidate equals the reference Top-1 candidate |

## Task 2 — Contextual Constraint Inference

| Metric | Meaning |
|:--|:--|
| `overall_accuracy` | Exact-set accuracy over all items |
| `single_accuracy` | Exact-set accuracy over single-choice items |
| `multi_accuracy` | Exact-set accuracy over multi-choice items |

## Task 3 — General Mobility Tasks

| Subtask | Metrics |
|:--|:--|
| POI Semantic Understanding — forward | `location_acc`, `tag_f1` |
| POI Semantic Understanding — backward | `hit_at_5` |
| Trajectory Fact Retrieval | `q1_acc`, `q2_acc`, `q3_poi_acc`, `q3_count_acc`, `q3_joint_acc` |
| Next-Step Mobility Prediction | `dest_poi_acc`, `dest_type_acc` |
| Location–Time Reasoning | `loc_pred_acc`, `time_pred_acc` |
| Mobility Preference Summarization | `overall_pref_acc`, `category_pref_acc` |
| Mobility Reason Inference | `exact_match` |
| Counterfactual Anomaly Detection | `plausibility_acc`, `anomaly_loc_acc` |

Task 3 evaluators also return `num_samples` and `_audit`. These describe prediction/ground-truth alignment and are not benchmark metrics.
