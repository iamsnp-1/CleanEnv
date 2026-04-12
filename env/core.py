"""
DataCleanEnv-X (DCX) — Core environment for data cleaning simulation.
Implements the OpenEnv interface: reset(), step(), state().
"""

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
    OpenEnv-compliant data cleaning environment.
    Simulates a data cleaning pipeline where an agent performs
    structured cleaning actions to improve data quality.
    """

    VALID_TASKS = ("easy", "medium", "hard")

    def __init__(self, task: str = "easy", max_steps: int = 10, seed: Optional[int] = None):
        """
        Initialize the DataCleanEnv environment.

        Args:
            task: The task identifier — 'easy', 'medium', or 'hard'.
            max_steps: Maximum actions allowed per episode.
            seed: Random seed for reproducibility.
        """
        if task not in self.VALID_TASKS:
            raise ValueError(f"Invalid task '{task}'. Must be one of: {self.VALID_TASKS}")
        self.task = task
        self.max_steps = max_steps
        self.seed = seed
        self.state_manager = StateManager(max_steps=max_steps, seed=seed)
        self._done = False

    def reset(self) -> ObservationModel:
        """
        Resets the environment to its initial state.

        Returns:
            ObservationModel: The initial observation of the dataset.
        """
        self._done = False
        effective_seed = self.seed if self.seed is not None else 42

        task_loaders = {
            "easy": get_easy_task,
            "medium": get_medium_task,
            "hard": get_hard_task,
        }
        self.task_data = task_loaders[self.task](seed=effective_seed)

        self.state_manager.init_state_with_dataset(self.task_data["dataset"])
        self.initial_dataset = self.state_manager.state["working_dataset"].copy()

        return self._build_observation()

    def _build_observation(self) -> ObservationModel:
        """Builds the ObservationModel from the current internal state."""
        state = self.state_manager.state
        if "working_dataset" not in state:
            raise ValueError("State missing 'working_dataset'")
        df: pd.DataFrame = state["working_dataset"]

        # 1. Sample rows (first 20, with NA → None for JSON)
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

            example_values = df[col].dropna().astype(str).unique()[:5].tolist()
            if not example_values:
                example_values = [None]

            column_profiles[col] = ColumnProfile(
                dtype=str(df[col].dtype),
                missing_pct=round(missing_pct, 2),
                unique=unique_count,
                example_values=example_values,
            )

        # 3. Detected issues
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
                severity=severity,
            ))

        # 4. Progress tracking
        issues_remaining = sum(
            issue.get("count", 0) for issue in state.get("issues", [])
        )
        progress = Progress(
            issues_fixed=state.get("issues_fixed", 0),
            issues_remaining=issues_remaining,
        )

        # 5. Validation rules
        validation_rules = {
            "no_missing": False,
            "type_enforced": False,
        }

        # 6. Step budget
        step_budget_remaining = max(
            state.get("max_steps", 10) - state.get("step_count", 0),
            0,
        )

        action_history_length = len(state.get("history", []))

        return ObservationModel(
            sample_rows=sample_rows,
            column_profiles=column_profiles,
            detected_issues=detected_issues,
            validation_rules=validation_rules,
            progress=progress,
            step_budget_remaining=step_budget_remaining,
            action_history_length=action_history_length,
        )

    def step(self, action: ActionModel) -> Tuple[ObservationModel, RewardModel, bool, Dict[str, Any]]:
        """
        Executes a single action in the environment.

        Returns:
            Tuple of (observation, reward, done, info).
        """
        # Guard against stepping after done
        if self._done:
            obs = self._build_observation()
            reward = RewardModel(value=0.0, components={}, reason="episode already done")
            return obs, reward, True, {"error": "episode already finished"}

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

        # Apply action via handler
        handler = handler_map.get(action.type.value)
        if handler:
            try:
                changes = handler(df, action)
            except Exception:
                changes = {"status": "no_op", "reason": "handler_error"}
        else:
            changes = {"status": "no_op", "reason": "unknown_action"}

        state["working_dataset"] = df

        # Update step count
        state["step_count"] = state.get("step_count", 0) + 1

        # Recompute issues
        prev_issues = sum(issue.get("count", 0) for issue in state.get("issues", []))
        state["issues"] = self.state_manager.detect_issues()
        new_issues = sum(issue.get("count", 0) for issue in state.get("issues", []))

        state["issues_fixed"] += max(prev_issues - new_issues, 0)

        # Compute reward
        reward_value = compute_reward(prev_issues, new_issues, action.type.value)

        reward = RewardModel(
            value=reward_value,
            components={
                "issue_reduction": float(prev_issues - new_issues),
                "prev_issues": float(prev_issues),
                "new_issues": float(new_issues),
            },
            reason=f"{'improvement' if reward_value > 0 else 'no improvement' if reward_value == 0 else 'regression'}",
        )

        state.setdefault("history", []).append({
            "action": action.model_dump(),
            "changes": changes,
            "reward": float(reward_value),
        })

        # Build observation
        obs = self._build_observation()

        # Compute done flag
        done = (
            state["step_count"] >= state.get("max_steps", 10)
            or action.type.value == "finish"
        )

        # Info dict
        info = {
            "issues_remaining": new_issues,
            "step_count": state["step_count"],
            "task": self.task,
            "max_steps": state.get("max_steps", self.max_steps),
        }

        if done:
            self._done = True
            final_df = state["working_dataset"]
            grader_map = {
                "easy": grade_easy,
                "medium": grade_medium,
                "hard": grade_hard,
            }
            grader = grader_map[self.task]
            final_score = grader(self.initial_dataset, final_df, self.task_data.get("ground_truth"))
            info["final_score"] = float(max(0.0001, min(0.9999, final_score)))

        return obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        """Returns the raw internal state representation of the environment."""
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
        if dtype_target in ("int", "float"):
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
