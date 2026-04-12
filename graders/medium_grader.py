"""
Medium grader for DataCleanEnv-X.
Scores based on: missing fix (30%) + duplicate removal (40%) + format normalization (30%).
"""

import pandas as pd
import numpy as np
from graders.utils import count_missing_values, count_duplicates, count_format_issues, strict_score


def grade_medium(original_df: pd.DataFrame, cleaned_df: pd.DataFrame, ground_truth: dict = None) -> float:
    """
    Grade the medium task.

    Criteria:
    - 30% weight: missing value reduction
    - 40% weight: duplicate removal
    - 30% weight: format normalization (inconsistent casing/whitespace)

    Returns:
        Score between 0 and 1.
    """
    # Missing values
    orig_missing = count_missing_values(original_df)
    clean_missing = count_missing_values(cleaned_df)
    missing_score = (orig_missing - clean_missing) / max(orig_missing, 1)
    missing_score = strict_score(missing_score)

    # Duplicates
    orig_dup = count_duplicates(original_df)
    clean_dup = count_duplicates(cleaned_df)
    dup_score = (orig_dup - clean_dup) / max(orig_dup, 1)
    dup_score = strict_score(dup_score)

    # Format issues
    orig_fmt = count_format_issues(original_df)
    clean_fmt = count_format_issues(cleaned_df)
    fmt_score = (orig_fmt - clean_fmt) / max(orig_fmt, 1)
    fmt_score = strict_score(fmt_score)

    score = 0.3 * missing_score + 0.4 * dup_score + 0.3 * fmt_score

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
    from tasks.medium import get_medium_task

    task = get_medium_task(seed=42)
    score = grade_medium(task["dataset"], task["dataset"])
    print("Score:", score)
