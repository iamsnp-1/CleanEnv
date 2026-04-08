from tasks.dataset_loader import load_dataset
from tasks.corruption_engine import corrupt_dataset

def get_medium_task(seed=42):
    """
    Medium task:
    - missing emails
    - duplicate rows
    - inconsistent formatting
    """
    df = load_dataset("medium", seed=seed)
    corrupted_df, ground_truth = corrupt_dataset(df, "medium", seed=seed)

    corrupted_df.loc[0:50, "category"] = corrupted_df.loc[0:50, "category"].astype(str).str.upper()
    corrupted_df["is_active"] = corrupted_df["is_active"].astype(object)
    corrupted_df.loc[50:100, "is_active"] = None

    validation_rules = {
        "email": {"format": "lowercase_email"},
        "phone": {"format": "starts_with_+91"},
        "signup_date": {"type": "date"}
    }

    task_description = """
    Clean the dataset by:
    - Filling missing emails
    - Removing duplicates
    - Normalizing email to lowercase
    - Fixing phone numbers to include +91
    """

    return {
        "dataset": corrupted_df,
        "ground_truth": ground_truth,
        "task_type": "medium"
    }


if __name__ == "__main__":
    task = get_medium_task()
    print(task["dataset"].head())
    print("\nGround Truth:\n", task["ground_truth"])
