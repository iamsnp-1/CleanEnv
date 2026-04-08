import pandas as pd
import numpy as np

def introduce_missing_values(df, columns, pct=0.1):
    """
    Randomly set some values to NaN
    """
    df_copy = df.copy()
    n_rows = len(df)

    missing_indices = []

    for col in columns:
        num_missing = int(pct * n_rows)
        idx = np.random.choice(n_rows, num_missing, replace=False)
        df_copy.loc[idx, col] = np.nan

        for i in idx:
            missing_indices.append({"row": int(i), "column": col})

    return df_copy, missing_indices

def introduce_type_errors(df, column):
    """
    Convert some numeric values into strings
    """
    df_copy = df.copy()
    # Convert to object dtype first to avoid FutureWarning
    df_copy[column] = df_copy[column].astype(object)

    indices = np.random.choice(len(df), int(0.1 * len(df)), replace=False)

    type_error_indices = []

    for i in indices:
        df_copy.loc[i, column] = np.random.choice([
            str(df_copy.loc[i, column]),
            "unknown",
            "N/A",
            "error"
        ])
        type_error_indices.append({"row": int(i), "column": column})

    return df_copy, type_error_indices

def introduce_duplicates(df):
    """
    Duplicate random rows
    """
    df_copy = df.copy()

    duplicate_indices = np.random.choice(len(df), int(0.1 * len(df)), replace=False)
    duplicates = df.iloc[duplicate_indices].copy()

    df_copy = pd.concat([df_copy, duplicates], ignore_index=True)

    return df_copy, duplicates.to_dict(orient="records")

def introduce_outliers(df, column):
    """
    Add extreme values to numeric column
    """
    df_copy = df.copy()

    indices = np.random.choice(len(df), int(0.05 * len(df)), replace=False)

    outlier_indices = []

    for i in indices:
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        df_copy.loc[i, column] = mean_val + np.random.randn() * 5 * std_val
        outlier_indices.append({"row": int(i), "column": column})

    return df_copy, outlier_indices

def corrupt_dataset(df, task_type, seed=42):
    """
    Apply multiple corruptions and return:
    corrupted_df, ground_truth
    """
    np.random.seed(seed)

    ground_truth = {
        "missing_values": [],
        "type_errors": [],
        "duplicates": [],
        "outliers": []
    }

    df_corrupt = df.copy()

    if task_type == "easy":
        df_corrupt, missing = introduce_missing_values(df_corrupt, ["age", "income"])
        df_corrupt, type_err = introduce_type_errors(df_corrupt, "age")

        ground_truth["missing_values"] = missing
        ground_truth["type_errors"] = type_err

    elif task_type == "medium":
        df_corrupt, missing = introduce_missing_values(df_corrupt, ["category"])
        df_corrupt, dup = introduce_duplicates(df_corrupt)

        ground_truth["missing_values"] = missing
        ground_truth["duplicates"] = dup

    elif task_type == "hard":
        df_corrupt, missing = introduce_missing_values(df_corrupt, ["age", "income"])
        df_corrupt, type_err = introduce_type_errors(df_corrupt, "age")
        df_corrupt, dup = introduce_duplicates(df_corrupt)
        df_corrupt, out = introduce_outliers(df_corrupt, "income")

        ground_truth["missing_values"] = missing
        ground_truth["type_errors"] = type_err
        ground_truth["duplicates"] = dup
        ground_truth["outliers"] = out

    else:
        raise ValueError("Invalid task type")

    return df_corrupt, ground_truth

if __name__ == "__main__":
    from tasks.dataset_loader import load_dataset

    df = load_dataset("easy")
    corrupted_df, gt = corrupt_dataset(df, "easy")

    print(corrupted_df.head())
    print("Ground truth:", gt)
