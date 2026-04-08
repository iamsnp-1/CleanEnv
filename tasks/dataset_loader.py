"""
Dataset loader for DataCleanEnv-X.
Generates clean synthetic datasets of varying sizes for each task difficulty.

Sizes are kept small to ensure inference runs under 20 minutes
on constrained machines (2 vCPU, 8GB RAM).
"""

import pandas as pd
import numpy as np


def load_dataset(task_type: str, seed: int = 42) -> pd.DataFrame:
    """
    Generate a clean synthetic dataset for the given task type.

    Args:
        task_type: One of 'easy', 'medium', 'hard'.
        seed: Random seed for reproducibility.

    Returns:
        A clean DataFrame before corruption.
    """
    np.random.seed(seed)

    categories = ["Tech", "Finance", "Healthcare", "Retail", "Education"]

    if task_type == "easy":
        n_rows = 200
    elif task_type == "medium":
        n_rows = 500
    elif task_type == "hard":
        n_rows = 1000
    else:
        raise ValueError(f"Invalid task type: {task_type}")

    df = pd.DataFrame({
        "id": range(1, n_rows + 1),
        "age": np.random.randint(18, 80, n_rows),
        "income": np.random.randint(30000, 150000, n_rows),
        "category": np.random.choice(categories, n_rows),
        "is_active": np.random.choice([True, False], n_rows, p=[0.8, 0.2]),
    })

    return df


if __name__ == "__main__":
    for t in ["easy", "medium", "hard"]:
        df = load_dataset(t)
        print(f"{t}: {df.shape}")
