"""
Grading utility functions for DataCleanEnv-X.
Updated to guarantee scores are ALWAYS strictly in (0, 1).
"""

import pandas as pd
import numpy as np

def strict_score(score: float) -> float:
    """Force score strictly into (0, 1) — never 0.0 or 1.0."""
    try:
        score = float(score)
    except (TypeError, ValueError):
        return 0.5

    if pd.isna(score) or score <= 0:
        return 0.01
    if score >= 1:
        return 0.99

    # Extra safety layer
    if score <= 0.0001:
        return 0.01
    if score >= 0.9999:
        return 0.99

    return round(float(score), 6)  # Clean float, avoid floating-point edge cases

# Rest of the file remains exactly the same
def count_missing_values(df: pd.DataFrame) -> int:
    return int(df.isna().sum().sum())

def count_type_errors(df: pd.DataFrame) -> int:
    errors = 0
    for col in ["age", "income"]:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            errors += int(numeric.isna().sum() - df[col].isna().sum())
    return max(0, errors)

def count_duplicates(df: pd.DataFrame) -> int:
    if "id" in df.columns:
        return int(df.duplicated(subset=["id"]).sum())
    return int(df.duplicated().sum())

def count_outliers(df: pd.DataFrame) -> int:
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
    return (
        count_missing_values(df)
        + count_type_errors(df)
        + count_duplicates(df)
        + count_outliers(df)
        + count_format_issues(df)
    )