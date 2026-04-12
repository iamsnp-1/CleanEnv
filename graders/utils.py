"""
Grading utility functions for DataCleanEnv-X.
Updated strict_score to guarantee scores are ALWAYS in (0.01, 0.99).
"""

import pandas as pd
import numpy as np

def strict_score(score):
    """Force score strictly into (0.01, 0.99) — never 0.0 or 1.0.
    This matches the successful pattern ."""
    try:
        score = float(score)
    except (TypeError, ValueError):
        return 0.5

    if pd.isna(score) or score <= 0.0:
        return 0.01
    if score >= 1.0:
        return 0.99

    # Extra safety against floating-point precision issues
    if score < 0.01:
        return 0.01
    if score > 0.99:
        return 0.99

    return round(score, 6)

def count_missing_values(df: pd.DataFrame) -> int:
    """Count total missing (NaN/None) values across all columns."""
    return int(df.isna().sum().sum())

def count_type_errors(df: pd.DataFrame) -> int:
    """
    Count type errors: non-numeric values in columns that should be numeric (age, income).
    """
    errors = 0
    for col in ["age", "income"]:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
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
    """
    issues = 0
    for col in df.select_dtypes(include=["object"]).columns:
        values = df[col].dropna().astype(str)
        normalized = values.str.strip().str.lower()
        if normalized.nunique() != values.nunique():
            issues += int((values != values.str.strip()).sum())
            issues += int((values != values.str.lower()).sum())

    if "is_active" in df.columns:
        issues += int((~df["is_active"].apply(lambda x: isinstance(x, bool) or pd.isna(x))).sum())

    if "category" in df.columns:
        issues += int((~df["category"].apply(lambda x: isinstance(x, str) or pd.isna(x))).sum())

    return issues

def count_issues(df: pd.DataFrame) -> int:
    """Total issue count across all categories."""
    return (
        count_missing_values(df)
        + count_type_errors(df)
        + count_duplicates(df)
        + count_outliers(df)
        + count_format_issues(df)
    )
