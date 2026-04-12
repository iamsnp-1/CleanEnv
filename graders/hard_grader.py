"""
Hard grader for DataCleanEnv-X.
Scores based on: missing (20%) + types (20%) + duplicates (20%) + outliers (20%) + flagging (20%).
"""

import pandas as pd
import numpy as np
from graders.utils import (
    count_missing_values,
    count_type_errors,
    count_duplicates,
    count_outliers,
    count_format_issues,
)


def grade_hard(original_df: pd.DataFrame, cleaned_df: pd.DataFrame, ground_truth: dict = None) -> float:
    """
    Grade the hard task.

    Criteria (equal weight):
    - 20%: missing value reduction
    - 20%: type error reduction
    - 20%: duplicate removal
    - 20%: outlier handling
    - 20%: format / invalid value fix

    Returns:
        Score between 0.0 and 1.0.
    """
    # Missing values
    orig_missing = count_missing_values(original_df)
    clean_missing = count_missing_values(cleaned_df)
    missing_score = (orig_missing - clean_missing) / max(orig_missing, 1)
    missing_score = max(0.0, min(1.0, missing_score))

    # Type errors
    orig_type = count_type_errors(original_df)
    clean_type = count_type_errors(cleaned_df)
    type_score = (orig_type - clean_type) / max(orig_type, 1)
    type_score = max(0.0, min(1.0, type_score))

    # Duplicates
    orig_dup = count_duplicates(original_df)
    clean_dup = count_duplicates(cleaned_df)
    dup_score = (orig_dup - clean_dup) / max(orig_dup, 1)
    dup_score = max(0.0, min(1.0, dup_score))

    # Outliers
    orig_out = count_outliers(original_df)
    clean_out = count_outliers(cleaned_df)
    outlier_score = (orig_out - clean_out) / max(orig_out, 1)
    outlier_score = max(0.0, min(1.0, outlier_score))

    # Format / invalid values
    orig_fmt = count_format_issues(original_df)
    clean_fmt = count_format_issues(cleaned_df)
    fmt_score = (orig_fmt - clean_fmt) / max(orig_fmt, 1)
    fmt_score = max(0.0, min(1.0, fmt_score))

    score = 0.2 * missing_score + 0.2 * type_score + 0.2 * dup_score + 0.2 * outlier_score + 0.2 * fmt_score
    # enforce strict bounds
    score = max(0.0001, min(0.9999, score))
    return score


if __name__ == "__main__":
    from tasks.hard import get_hard_task

    task = get_hard_task(seed=42)
    score = grade_hard(task["dataset"], task["dataset"])
    print("Score:", score)
