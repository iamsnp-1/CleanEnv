import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

class StateManager:
    """
    Manages the internal state, dataset copy, and transformation history.
    """
    def __init__(self, max_steps: int = 10, seed: Optional[int] = None):
        """
        Initializes the state manager.
        """
        self.max_steps = max_steps
        self.seed = seed
        self.state: Dict[str, Any] = {}
        if seed is not None:
            np.random.seed(seed)

    def init_state_with_dataset(self, dataset: pd.DataFrame) -> None:
        """
        Initializes state from an externally-provided dataset (from task loaders).
        """
        raw_dataset = dataset.copy(deep=True)
        self.state = {
            "raw_dataset": raw_dataset.copy(deep=True),
            "working_dataset": raw_dataset.copy(deep=True),
            "issues": [],
            "issues_fixed": 0,
            "step_count": 0,
            "max_steps": self.max_steps,
            "history": []
        }
        self.state["issues"] = self.detect_issues()

    def init_state(self, task: str) -> None:
        """
        Initializes the dataset and internal tracking variables.
        """
        raw_dataset = self._generate_mock_dataset(task)
        self.state = {
            "raw_dataset": raw_dataset.copy(deep=True),
            "working_dataset": raw_dataset.copy(deep=True),
            "issues": [],
            "issues_fixed": 0,
            "step_count": 0,
            "max_steps": self.max_steps,
            # history entries will follow structure:
            # {"action": dict, "changes": dict, "reward": float}
            "history": []
        }
        self.state["issues"] = self.detect_issues()

    def _generate_mock_dataset(self, task: str) -> pd.DataFrame:
        """
        Generates a mock dataset with missing values, mixed types, and duplicates.
        """
        size = 200 if task == "easy" else 1000
        data = {
            "id": range(1, size + 1),
            "age": np.random.choice([25, 30, np.nan, 45, 50, 120, -5], size=size),
            "income": np.random.normal(50000, 15000, size=size),
            "category": np.random.choice(["A", "B", "C", None, "B "], size=size),
            "is_active": np.random.choice([True, False, "Yes", "No"], size=size)
        }
        df = pd.DataFrame(data)
        
        # Introduce some exact duplicates
        df = pd.concat([df, df.iloc[:5]], ignore_index=True)
        return df

    def detect_issues(self) -> List[Dict[str, Any]]:
        """
        Lightweight issue detection (missing values, duplicates, mixed types).
        """
        issues = []
        if "working_dataset" not in self.state:
            raise ValueError("State missing 'working_dataset'")
            
        df: pd.DataFrame = self.state["working_dataset"]

        # 1. Missing values
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                issues.append({
                    "type": "missing_values",
                    "column": col,
                    "count": int(missing_count)
                })

        for col in df.select_dtypes(include=[np.number]).columns:
            col_data = df[col].dropna()

            if col_data.empty:
                continue

            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1

            outliers = col_data[
                (col_data < q1 - 1.5 * iqr) |
                (col_data > q3 + 1.5 * iqr)
            ]

            if not outliers.empty:
                issues.append({
                    "type": "outliers",
                    "column": col,
                    "count": int(len(outliers))
                })

        # 2. Duplicate rows
        subset_cols = ["id"] if "id" in df.columns else df.columns.tolist()
        duplicate_count = df.duplicated(subset=subset_cols).sum()
        if duplicate_count > 0:
            issues.append({
                "type": "duplicates",
                "column": None,
                "count": int(duplicate_count)
            })

        # 3. Type inconsistency (mixed types in object columns)
        for col in df.select_dtypes(include=['object']).columns:
            values = df[col].dropna().astype(str)
            
            # Detect inconsistent formatting (e.g., "B" vs "B ", "Yes" vs "yes")
            normalized = values.str.strip().str.lower()
            
            if normalized.nunique() != values.nunique():
                issues.append({
                    "type": "inconsistent_format",
                    "column": col,
                    "count": int(len(values))
                })
            
            # Detect true mixed types
            non_null = df[col].dropna()
            type_counts = non_null.apply(type).nunique()
            
            if type_counts > 1:
                issues.append({
                    "type": "mixed_types",
                    "column": col,
                    "count": int(non_null.shape[0])
                })

        return issues

    def get_state_dict(self) -> Dict[str, Any]:
        """
        Returns the full internal state. Converts DataFrames to dicts/summaries.
        """
        state_copy = self.state.copy()
        if "working_dataset" in state_copy and isinstance(state_copy["working_dataset"], pd.DataFrame):
            df = state_copy["working_dataset"]
            state_copy["working_dataset_summary"] = {
                "shape": df.shape,
                "columns": list(df.columns)
            }
            # Convert a sample of the dataframe for state visibility to avoid huge dumps
            state_copy["working_dataset"] = df.where(pd.notnull(df), None).head(5).to_dict(orient="records")
            
        if "raw_dataset" in state_copy and isinstance(state_copy["raw_dataset"], pd.DataFrame):
            df_raw = state_copy["raw_dataset"]
            state_copy["raw_dataset"] = df_raw.where(pd.notnull(df_raw), None).head(5).to_dict(orient="records")
            
        return state_copy
