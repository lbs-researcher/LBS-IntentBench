"""
Task 3: General Mobility Tasks (GMT) Evaluation
================================================

The 7 subtasks under General Mobility Tasks (one .py per subtask):

    1. POI Semantic Understanding   (forward + backward, same module)
    2. Trajectory Fact Retrieval
    3. Next-Step Mobility Prediction
    4. Location-Time Reasoning
    5. Mobility Preference Summarization
    6. Mobility Reason Inference
    7. Counterfactual Anomaly Detection

Each subtask is a self-contained module exposing an ``evaluate`` function:

    from evaluation.task3_gmt import next_step_mobility_prediction
    metrics = next_step_mobility_prediction.evaluate(
        "preds.jsonl",
        "data/task3_gmt/next_step_mobility_prediction.csv",
    )

POI Semantic Understanding additionally takes a ``direction`` argument
because the same gold CSV is used for both the forward (POI → location/tags)
and backward (description → POI candidates) prompts:

    from evaluation.task3_gmt import poi_semantic_understanding
    fwd = poi_semantic_understanding.evaluate(
        "preds_fwd.jsonl",
        "data/task3_gmt/poi_semantic_understanding.csv",
        direction="forward",
    )
    bwd = poi_semantic_understanding.evaluate(
        "preds_bwd.jsonl",
        "data/task3_gmt/poi_semantic_understanding.csv",
        direction="backward",
    )

Each ``evaluate`` returns a dict whose keys exactly match the columns of
Table 3 ("Results on General Mobility Tasks") in the LBS-IntentBench paper.
"""

from evaluation.task3_gmt import (  # noqa: F401
    counterfactual_anomaly_detection,
    location_time_reasoning,
    mobility_preference_summarization,
    mobility_reason_inference,
    next_step_mobility_prediction,
    poi_semantic_understanding,
    trajectory_fact_retrieval,
)
