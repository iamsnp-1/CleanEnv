import pandas as pd
from graders.utils import count_issues

def grade_medium(original_df, cleaned_df, ground_truth=None):
    """
    Returns score between 0 and 1
    """
    before = count_issues(original_df)
    after = count_issues(cleaned_df)

    score = (before - after) / max(before, 1)
    return round(float(max(0.0, min(1.0, score))), 4)

if __name__ == "__main__":
    from tasks.medium import get_medium_task

    task = get_medium_task(seed=42)
    score = grade_medium(task["dataset"], task["dataset"])
    print("Score:", score)
