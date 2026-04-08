from tasks.dataset_loader import load_dataset
from tasks.corruption_engine import corrupt_dataset

def get_hard_task(seed=42):
    """
    Hard task:
    - Missing values
    - Type errors
    - Duplicates
    - Outliers
    - Invalid rows (must be flagged)
    """

    # Step 1: Load dataset
    df = load_dataset("hard", seed=seed)

    # Step 2: Corrupt dataset
    corrupted_df, ground_truth = corrupt_dataset(df, "hard", seed=seed)

    # Step 3: Add additional complex issues

    # Invalid categories
    corrupted_df.loc[0:50, "category"] = "invalid_category"

    # Impossible ages
    corrupted_df.loc[50:100, "age"] = -10

    # Inconsistent booleans (cast to object first to avoid FutureWarning)
    corrupted_df["is_active"] = corrupted_df["is_active"].astype(object)
    corrupted_df.loc[100:150, "is_active"] = "unknown"

    # Step 4: Validation rules
    validation_rules = {
        "age": {"type": "int", "min": 0, "max": 120},
        "income": {"type": "int", "min": 0, "max": 1000000},
        "email": {"format": "valid_email"},
        "phone": {"format": "starts_with_+91"},
        "transaction_date": {"type": "date", "not_future": True}
    }

    # Step 5: Task description
    task_description = """
    Clean the dataset by:
    - Handling missing values
    - Fixing type errors
    - Removing duplicates
    - Detecting and handling outliers
    - Flagging invalid rows (do NOT delete blindly)
    """

    return {
        "dataset": corrupted_df,
        "ground_truth": ground_truth,
        "task_type": "hard"
    }


if __name__ == "__main__":
    task = get_hard_task()

    print(task["dataset"].head())
    print("\nGround Truth:\n", task["ground_truth"])
