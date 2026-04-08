import pandas as pd
import numpy as np

def load_dataset(task_type: str, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    categories = ["Tech", "Finance", "Healthcare", "Retail", "Education"]

    if task_type == "easy":
        n_rows = 1000
    elif task_type == "medium":
        n_rows = 3000
    elif task_type == "hard":
        n_rows = 8000
    else:
        raise ValueError("Invalid task type")

    df = pd.DataFrame({
        "id": range(1, n_rows + 1),
        "age": np.random.randint(18, 80, n_rows),
        "income": np.random.randint(30000, 150000, n_rows),
        "category": np.random.choice(categories, n_rows),
        "is_active": np.random.choice([True, False], n_rows, p=[0.8, 0.2])
    })
    
    return df

if __name__ == "__main__":
    df = load_dataset("easy")
    print(df.head())
    print(df.shape)
