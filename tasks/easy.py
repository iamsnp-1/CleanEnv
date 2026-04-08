from tasks.dataset_loader import load_dataset
from tasks.corruption_engine import corrupt_dataset

def get_easy_task(seed=42):
    """
    Returns:
        {
            "dataset": corrupted dataframe,
            "ground_truth": ground truth dict,
            "validation_rules": rules dict,
            "task_description": str
        }
    """

    # Step 1: Load clean data
    df = load_dataset("easy", seed=seed)

    # Step 2: Corrupt it
    corrupted_df, ground_truth = corrupt_dataset(df, "easy", seed=seed)

    # Step 3: Define validation rules
    validation_rules = {
        "age": {"type": "int", "min": 0, "max": 100},
        "salary": {"type": "int", "min": 20000, "max": 200000},
        "name": {"type": "string"},
        "department": {"type": "string"}
    }

    # Step 4: Task description (VERY IMPORTANT for agent)
    task_description = """
    Clean the dataset by:
    - Filling missing values
    - Fixing type errors
    - Ensuring numeric columns are valid
    """

    return {
        "dataset": corrupted_df,
        "ground_truth": ground_truth,
        "task_type": "easy"
    }

if __name__ == "__main__":
    task = get_easy_task()

    print(task["dataset"].head())
    print("\nGround Truth:\n", task["ground_truth"])
