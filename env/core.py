import pandas as pd
from typing import Any, Dict, Tuple, Optional
from .models import ActionModel, ObservationModel, RewardModel, ColumnProfile, DetectedIssue, Progress, Severity
from .state_manager import StateManager

from tasks.easy import get_easy_task
from tasks.medium import get_medium_task
from tasks.hard import get_hard_task
from graders.easy_grader import grade_easy
from graders.medium_grader import grade_medium
from graders.hard_grader import grade_hard
from reward.reward_engine import compute_reward


class DataCleanEnv:
    """
    Base environment class for DataCleanEnv-X (DCX).
    Simulates a data cleaning pipeline where an agent performs
    structured cleaning actions to improve data quality.
    """

    def __init__(self, task: str, max_steps: int = 10, seed: Optional[int] = None):
        """
        Initialize the DataCleanEnv environment.

        Args:
            task (str): The specific task or dataset identifier for the environment.
            max_steps (int): Maximum actions allowed per episode.
            seed (Optional[int]): Random seed for reproducibility.
        """
        self.task = task
        self.max_steps = max_steps
        self.seed = seed
        self.state_manager = StateManager(max_steps=max_steps, seed=seed)

    def reset(self) -> ObservationModel:
        """
        Resets the environment to its initial state.

        Returns:
            ObservationModel: The initial observation of the dataset and its state.
        """
        if self.task == "easy":
            self.task_data = get_easy_task(seed=self.seed if self.seed is not None else 42)
        elif self.task == "medium":
            self.task_data = get_medium_task(seed=self.seed if self.seed is not None else 42)
        elif self.task == "hard":
            self.task_data = get_hard_task(seed=self.seed if self.seed is not None else 42)
        else:
            raise ValueError("Invalid task")
            
        if hasattr(self.state_manager, "init_state_with_dataset"):
            self.state_manager.init_state_with_dataset(self.task_data["dataset"])
        else:
            raw_dataset = self.task_data["dataset"].copy(deep=True)
            self.state_manager.state = {
                "raw_dataset": raw_dataset.copy(deep=True),
                "working_dataset": raw_dataset.copy(deep=True),
                "issues": [],
                "issues_fixed": 0,
                "step_count": 0,
                "max_steps": self.max_steps,
                "history": []
            }
            self.state_manager.state["issues"] = self.state_manager.detect_issues()

        self.initial_dataset = self.state_manager.state["working_dataset"].copy()
        return self._build_observation()

    def _build_observation(self) -> ObservationModel:
        """
        Builds the ObservationModel from the current internal state.
        """
        state = self.state_manager.state
        if "working_dataset" not in state:
            raise ValueError("State missing 'working_dataset'")
        df: pd.DataFrame = state["working_dataset"]
        
        # 1. Sample rows (first 20 rows, converting NA to None for JSON serialization)
        # Using astype(object) ensures None can be inserted into float/int columns correctly
        df_cleaned = df.astype(object).where(pd.notnull(df), None)
        sample_rows = df_cleaned.head(20).to_dict(orient="records")
        if not sample_rows:
            sample_rows = [{}]
        
        # 2. Column profiles
        column_profiles = {}
        total_rows = len(df)
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = float((missing_count / total_rows) * 100) if total_rows > 0 else 0.0
            unique_count = int(df[col].nunique(dropna=True))
            
            # Example values: dropna, get unique, take up to 5, convert to list
            example_values = df[col].dropna().astype(str).unique()[:5].tolist()
            if not example_values:
                example_values = [None]
            
            column_profiles[col] = ColumnProfile(
                dtype=str(df[col].dtype),
                missing_pct=missing_pct,
                unique=unique_count,
                example_values=example_values
            )
            
        # 3. Detected issues mapped to ObservationModel schema
        detected_issues = []
        for issue in state.get("issues", []):
            count = issue.get("count", 0)
            if count > 50:
                severity = Severity.HIGH
            elif count > 10:
                severity = Severity.MEDIUM
            else:
                severity = Severity.LOW
                
            detected_issues.append(DetectedIssue(
                type=issue["type"],
                column=issue.get("column"),
                severity=severity
            ))
            
        # 4. Progress tracking
        issues_remaining = sum(
            issue.get("count", 0) for issue in state.get("issues", [])
        )
        progress = Progress(
            issues_fixed=state.get("issues_fixed", 0),
            issues_remaining=issues_remaining
        )
        
        # 5. Validation rules (placeholder as requested)
        validation_rules = {
            "no_missing": False,
            "type_enforced": False
        }
        
        # 6. Step budget
        step_budget_remaining = max(
            state.get("max_steps", 10) - state.get("step_count", 0),
            0
        )
        
        action_history_length = len(state.get("history", []))

        return ObservationModel(
            sample_rows=sample_rows,
            column_profiles=column_profiles,
            detected_issues=detected_issues,
            validation_rules=validation_rules,
            progress=progress,
            step_budget_remaining=step_budget_remaining,
            action_history_length=action_history_length
        )

    def step(self, action: ActionModel) -> Tuple[ObservationModel, RewardModel, bool, Dict[str, Any]]:
        """
        Executes a single action in the environment.
        """
        # 1. Validate action (already validated via ActionModel schema)
        # 2. Get current state + dataset
        state = self.state_manager.state
        if "working_dataset" not in state:
            raise ValueError("State missing 'working_dataset'")
        df = state["working_dataset"].copy()

        handler_map = {
            "fill_value": self._handle_fill_value,
            "cast_type": self._handle_cast_type,
            "normalize_field": self._handle_normalize,
            "drop_row": self._handle_drop_row,
            "flag_invalid": self._handle_flag_invalid,
            "deduplicate": self._handle_deduplicate,
            "handle_outliers": self._handle_outliers,
            "escalate": self._handle_escalate,
            "finish": self._handle_finish,
        }

        # 3. Apply action via handler
        handler = handler_map.get(action.type.value)
        if handler:
            try:
                changes = handler(df, action)
            except Exception:
                # NEVER crash on bad input
                changes = {"status": "no_op", "reason": "handler_error"}
        else:
            changes = {"status": "no_op", "reason": "unknown_action"}

        state["working_dataset"] = df

        # 4. Update state (dataset, history, step_count)
        state["step_count"] = state.get("step_count", 0) + 1
        
        # 5. Recompute issues using StateManager.detect_issues()
        prev_issues = sum(issue.get("count", 0) for issue in state.get("issues", []))
        state["issues"] = self.state_manager.detect_issues()
        new_issues = sum(issue.get("count", 0) for issue in state.get("issues", []))

        state["issues_fixed"] += max(prev_issues - new_issues, 0)

        reward_value = compute_reward(prev_issues, new_issues, action.type.value)
        
        reward = RewardModel(
            value=reward_value,
            components={
                "issue_reduction": prev_issues - new_issues
            },
            reason="issue-based reward"
        )
        
        state.setdefault("history", []).append({
            "action": action.model_dump(),
            "changes": changes,
            "reward": float(reward_value)
        })

        # 6. Build new observation
        obs = self._build_observation()

        # 7. Compute done flag
        done = (
            state["step_count"] >= state.get("max_steps", 10)
            or action.type.value == "finish"
        )

        # 8. Return observation, reward, done, info
        info = {
            "issues_remaining": sum(issue.get("count", 0) for issue in state.get("issues", [])),
            "step_count": state["step_count"],
            "task": self.task,
            "max_steps": state.get("max_steps", self.max_steps)
        }

        if done:
            final_df = state["working_dataset"]
            if self.task == "easy":
                final_score = grade_easy(self.initial_dataset, final_df, self.task_data["ground_truth"])
            elif self.task == "medium":
                final_score = grade_medium(self.initial_dataset, final_df, self.task_data["ground_truth"])
            else:
                final_score = grade_hard(self.initial_dataset, final_df, self.task_data["ground_truth"])
                
            info["final_score"] = float(max(0.0, min(1.0, final_score)))

        return obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        """
        Returns the raw internal state representation of the environment.
        """
        return self.state_manager.get_state_dict()

    # --- ACTION HANDLERS ---

    def _handle_fill_value(self, df: pd.DataFrame, action: ActionModel) -> Dict[str, Any]:
        col = action.column
        if not col or col not in df.columns:
            return {"status": "no_op", "reason": "invalid_column"}
            
        strategy = action.parameters.get("strategy")
        if not strategy:
            return {"status": "no_op", "reason": "missing_strategy"}
            
        initial_missing = int(df[col].isna().sum())
        if initial_missing == 0:
            return {"status": "no_op", "reason": "no_missing_values"}

        if strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
            fill_val = df[col].mean()
        elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
            fill_val = df[col].median()
        elif strategy == "mode":
            mode_s = df[col].mode()
            fill_val = mode_s.iloc[0] if not mode_s.empty else None
        elif strategy == "constant":
            fill_val = action.parameters.get("value", 0)
        else:
            return {"status": "no_op", "reason": "invalid_strategy_for_dtype"}
            
        df[col] = df[col].fillna(fill_val)
        return {"status": "success", "strategy": strategy, "filled_count": initial_missing}

    def _handle_cast_type(self, df: pd.DataFrame, action: ActionModel) -> Dict[str, Any]:
        col = action.column
        if not col or col not in df.columns:
            return {"status": "no_op", "reason": "invalid_column"}
            
        dtype_target = action.parameters.get("dtype")
        if dtype_target in ["int", "float"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if dtype_target == "int":
                df[col] = df[col].astype("Int64")
        elif dtype_target == "str":
            df[col] = df[col].astype(str)
        else:
            return {"status": "no_op", "reason": "unknown_dtype"}
            
        return {"status": "success", "cast_to": dtype_target}

    def _handle_normalize(self, df: pd.DataFrame, action: ActionModel) -> Dict[str, Any]:
        col = action.column
        if not col or col not in df.columns:
            return {"status": "no_op", "reason": "invalid_column"}
            
        method = action.parameters.get("method")
        if not pd.api.types.is_object_dtype(df[col]) and not pd.api.types.is_string_dtype(df[col]):
            return {"status": "no_op", "reason": "not_string_column"}
            
        if method == "lowercase":
            df[col] = df[col].astype(str).str.lower()
        elif method == "strip":
            df[col] = df[col].astype(str).str.strip()
        else:
            return {"status": "no_op", "reason": "invalid_method"}
            
        return {"status": "success", "method": method}

    def _handle_drop_row(self, df: pd.DataFrame, action: ActionModel) -> Dict[str, Any]:
        col = action.column
        if col and col not in df.columns:
            return {"status": "no_op", "reason": "invalid_column"}
            
        condition = action.parameters.get("condition")
        initial_len = len(df)
        
        if condition == "missing":
            if col:
                df.dropna(subset=[col], inplace=True)
            else:
                df.dropna(inplace=True)
        elif condition == "outlier" and col and pd.api.types.is_numeric_dtype(df[col]):
            col_data = df[col].dropna()

            if col_data.empty:
                return {"status": "no_op", "reason": "no_data"}

            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1

            mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
            df.drop(df[mask].index, inplace=True)
        else:
            return {"status": "no_op", "reason": "invalid_condition"}
            
        return {"status": "success", "dropped": initial_len - len(df)}

    def _handle_flag_invalid(self, df: pd.DataFrame, action: ActionModel) -> Dict[str, Any]:
        col = action.column
        if not col or col not in df.columns:
            return {"status": "no_op", "reason": "invalid_column"}
            
        if "is_flagged" not in df.columns:
            df["is_flagged"] = False
            
        df.loc[df[col].isna(), "is_flagged"] = True
        return {"status": "success", "flagged": True}

    def _handle_deduplicate(self, df: pd.DataFrame, action: ActionModel) -> Dict[str, Any]:
        initial_len = len(df)
        subset = ["id"] if "id" in df.columns else df.columns.tolist()
        df.drop_duplicates(subset=subset, inplace=True)
        return {"status": "success", "dropped": initial_len - len(df)}

    def _handle_outliers(self, df: pd.DataFrame, action: ActionModel) -> Dict[str, Any]:
        col = action.column
        if not col or col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            return {"status": "no_op", "reason": "invalid_column"}
            
        method = action.parameters.get("method")
        col_data = df[col].dropna()

        if col_data.empty:
            return {"status": "no_op", "reason": "no_data"}

        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            return {"status": "no_op", "reason": "zero_iqr"}

        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        
        initial_len = len(df)
        if method == "clip":
            df[col] = df[col].clip(lower=lower, upper=upper)
        elif method == "remove":
            df.drop(df[(df[col] < lower) | (df[col] > upper)].index, inplace=True)
        else:
            return {"status": "no_op", "reason": "invalid_method"}
            
        return {"status": "success", "method": method, "dropped": initial_len - len(df)}

    def _handle_escalate(self, df: pd.DataFrame, action: ActionModel) -> Dict[str, Any]:
        return {"status": "success", "message": "issues escalated"}

    def _handle_finish(self, df: pd.DataFrame, action: ActionModel) -> Dict[str, Any]:
        return {"status": "success", "message": "workflow finished"}


if __name__ == "__main__":
    from .models import ActionModel, ActionType

    env = DataCleanEnv(task="hard", seed=42)
    obs = env.reset()

    done = False
    while not done:
        action = ActionModel(
            type=ActionType.FILL_VALUE,
            column="age",
            parameters={"strategy": "median"}
        )
        obs, reward, done, info = env.step(action)

    print(info["final_score"])
