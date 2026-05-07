"""
LBS-IntentBench Evaluation Module

This package provides standardized evaluation scripts for all tasks in the
LBS-IntentBench benchmark. Each evaluation script follows a unified interface:

    def evaluate(predictions_file: str, ground_truth_file: str) -> dict:
        '''Load CSV files, compute metrics, and return a dictionary of results.'''

Supported tasks:
    - Task 1: Mobility Intent Inference (MII)
    - Task 2: Contextual Constraint Inference (CCI)
    - Task 3: General Mobility Tasks (GMT) with 7 subtasks
"""
