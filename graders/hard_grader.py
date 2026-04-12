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
    strict_score,
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
        Score between 0 and 1.
    """
    # Missing values
    orig_missing = count_missing_values(original_df)
    clean_missing = count_missing_values(cleaned_df)
    if orig_missing == 0:
        missing_score = 0.5
    else:
        missing_score = (orig_missing - clean_missing) / orig_missing
    missing_score = strict_score(missing_score)

    # Type errors
    orig_type = count_type_errors(original_df)
    clean_type = count_type_errors(cleaned_df)
    if orig_type == 0:
        type_score = 0.5
    else:
        type_score = (orig_type - clean_type) / orig_type
    type_score = strict_score(type_score)

    # Duplicates
    orig_dup = count_duplicates(original_df)
    clean_dup = count_duplicates(cleaned_df)
    if orig_dup == 0:
        dup_score = 0.5
    else:
        dup_score = (orig_dup - clean_dup) / orig_dup
    dup_score = strict_score(dup_score)

    # Outliers
    orig_out = count_outliers(original_df)
    clean_out = count_outliers(cleaned_df)
    if orig_out == 0:
        outlier_score = 0.5
    else:
        outlier_score = (orig_out - clean_out) / orig_out
    outlier_score = strict_score(outlier_score)

    # Format / invalid values
    orig_fmt = count_format_issues(original_df)
    clean_fmt = count_format_issues(cleaned_df)
    if orig_fmt == 0:
        fmt_score = 0.5
    else:
        fmt_score = (orig_fmt - clean_fmt) / orig_fmt
    fmt_score = strict_score(fmt_score)

    score = 0.2 * missing_score + 0.2 * type_score + 0.2 * dup_score + 0.2 * outlier_score + 0.2 * fmt_score
    score = strict_score(score)

    assert 0.0 < score < 1.0

    return score


if __name__ == "__main__":
    from tasks.hard import get_hard_task

    task = get_hard_task(seed=42)
    score = grade_hard(task["dataset"], task["dataset"])
    print("Score:", score)
