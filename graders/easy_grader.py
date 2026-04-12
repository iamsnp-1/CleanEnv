"""
Easy grader for DataCleanEnv-X.
Scores based on: missing value reduction (50%) + type error fix (50%).
"""

import pandas as pd
import numpy as np
from graders.utils import count_missing_values, count_type_errors, strict_score


def grade_easy(original_df: pd.DataFrame, cleaned_df: pd.DataFrame, ground_truth: dict = None) -> float:
    """
    Grade the easy task.

    Criteria:
    - 50% weight: missing value reduction
    - 50% weight: type error reduction (non-numeric values in numeric columns)

    Returns:
        Score between 0 and 1.
    """
    # Missing values
    orig_missing = count_missing_values(original_df)
    clean_missing = count_missing_values(cleaned_df)
    missing_score = (orig_missing - clean_missing) / max(orig_missing, 1)
    missing_score = strict_score(missing_score)

    # Type errors (non-numeric in age/income)
    orig_type_err = count_type_errors(original_df)
    clean_type_err = count_type_errors(cleaned_df)
    type_score = (orig_type_err - clean_type_err) / max(orig_type_err, 1)
    type_score = strict_score(type_score)

    score = 0.5 * missing_score + 0.5 * type_score

    # convert to float explicitly
    score = float(score)

    # clamp BEFORE strict_score to avoid rounding to 1.0
    score = min(score, 0.9999)
    score = max(score, 0.0001)

    # final normalization
    score = strict_score(score)

    assert 0.0 < score < 1.0

    return score


if __name__ == "__main__":
    from tasks.easy import get_easy_task

    task = get_easy_task(seed=42)
    score = grade_easy(task["dataset"], task["dataset"])
    print("Score:", score)
