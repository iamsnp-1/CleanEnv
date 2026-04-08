import pandas as pd
import numpy as np

def count_issues(df: pd.DataFrame) -> int:
    issues = 0
    # Missing values
    issues += int(df.isna().sum().sum())
    
    # Duplicates
    if "id" in df.columns:
        issues += int(df.duplicated(subset=["id"]).sum())
    else:
        issues += int(df.duplicated().sum())
        
    # Types and Outliers for numeric fields
    for col in ["age", "income"]:
        if col in df.columns:
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            # Type errors: values that couldn't be parsed to numeric (and weren't originally NaN)
            issues += int(numeric_series.isna().sum() - df[col].isna().sum())
            
            # Simple Outliers (mean +/- 4 std)
            col_data = numeric_series.dropna()
            if not col_data.empty:
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    outliers = ((col_data < q1 - 1.5 * iqr) | (col_data > q3 + 1.5 * iqr)).sum()
                    issues += int(outliers)
                    
    # Validate categorical field types roughly
    if "category" in df.columns:
        issues += int((~df["category"].apply(lambda x: isinstance(x, str) or pd.isna(x))).sum())
        
    if "is_active" in df.columns:
        issues += int((~df["is_active"].apply(lambda x: isinstance(x, bool) or pd.isna(x))).sum())

    return issues
