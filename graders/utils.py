"""
Grading utility functions for DataCleanEnv-X.
Each function counts a specific category of issues in a DataFrame.
All counts are deterministic and reproducible.
"""

import pandas as pd
import numpy as np


def strict_score(score):
    """Enforce scores strictly within (0, 1). Zero tolerance for boundary values."""
    try:
        score = float(score)
    except (TypeError, ValueError):
        return 0.5

    if score <= 0:
        return 0.0001
    if score >= 1:
        return 0.9999

    if score > 0.9999:
        return 0.9999
    if score < 0.0001:
        return 0.0001

    return score


def count_missing_values(df: pd.DataFrame) -> int:
    """Count total missing (NaN/None) values across all columns."""
    return int(df.isna().sum().sum())


def count_type_errors(df: pd.DataFrame) -> int:
    """
    Count type errors: non-numeric values in columns that should be numeric (age, income).
    A type error is a value that exists (not NaN) but can't be parsed as a number.
    """
    errors = 0
    for col in ["age", "income"]:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            # Values that were not NaN originally but became NaN after coercion
            errors += int(numeric.isna().sum() - df[col].isna().sum())
    return max(0, errors)


def count_duplicates(df: pd.DataFrame) -> int:
    """Count duplicate rows (by 'id' column if present, else all columns)."""
    if "id" in df.columns:
        return int(df.duplicated(subset=["id"]).sum())
    return int(df.duplicated().sum())


def count_outliers(df: pd.DataFrame) -> int:
    """
    Count outliers in numeric columns using IQR method.
    An outlier is a value outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    """
    outliers = 0
    for col in df.select_dtypes(include=[np.number]).columns:
        col_data = df[col].dropna()
        if col_data.empty:
            continue
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            outliers += int(((col_data < q1 - 1.5 * iqr) | (col_data > q3 + 1.5 * iqr)).sum())
    return outliers


def count_format_issues(df: pd.DataFrame) -> int:
    """
    Count format inconsistencies in string/object columns.
    Detects: leading/trailing whitespace, inconsistent casing.
    Also checks for non-boolean values in 'is_active' and non-string values in 'category'.
    """
    issues = 0

    # Check string columns for whitespace/casing inconsistency
    for col in df.select_dtypes(include=["object"]).columns:
        values = df[col].dropna().astype(str)
        normalized = values.str.strip().str.lower()
        if normalized.nunique() != values.nunique():
            # Count how many values differ from their normalized form
            issues += int((values != values.str.strip()).sum())
            issues += int((values != values.str.lower()).sum())

    # Check is_active for non-boolean values
    if "is_active" in df.columns:
        issues += int((~df["is_active"].apply(lambda x: isinstance(x, bool) or pd.isna(x))).sum())

    # Check category for non-string values
    if "category" in df.columns:
        issues += int((~df["category"].apply(lambda x: isinstance(x, str) or pd.isna(x))).sum())

    return issues


def count_issues(df: pd.DataFrame) -> int:
    """
    Legacy function: total issue count across all categories.
    Used by the state manager for issue tracking.
    """
    return (
        count_missing_values(df)
        + count_type_errors(df)
        + count_duplicates(df)
        + count_outliers(df)
        + count_format_issues(df)
    )
