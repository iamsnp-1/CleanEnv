from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class ActionType(str, Enum):
    """Defines all supported actions for data cleaning."""
    FILL_VALUE = "fill_value"
    CAST_TYPE = "cast_type"
    NORMALIZE_FIELD = "normalize_field"
    DROP_ROW = "drop_row"
    FLAG_INVALID = "flag_invalid"
    DEDUPLICATE = "deduplicate"
    HANDLE_OUTLIERS = "handle_outliers"
    ESCALATE = "escalate"
    FINISH = "finish"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ActionModel(BaseModel):
    """
    Represents an action taken by the data cleaning agent.
    """
    model_config = {
        "arbitrary_types_allowed": True
    }

    id: Optional[str] = None
    type: ActionType
    column: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_column_requirement(self) -> ActionModel:
        """
        Validates that the given action type has a column specified, if required.
        Most actions require a specified target column, except for global actions
        like deduplicate, escalate, or finish.
        """
        requires_column = {
            ActionType.FILL_VALUE,
            ActionType.CAST_TYPE,
            ActionType.NORMALIZE_FIELD,
            ActionType.DROP_ROW,
            ActionType.FLAG_INVALID,
            ActionType.HANDLE_OUTLIERS,
        }
        if self.type in requires_column and not self.column:
            raise ValueError(f"Action '{self.type.value}' requires a column to be specified.")
        return self

    @model_validator(mode="after")
    def validate_parameters(self) -> ActionModel:
        if self.type == ActionType.FILL_VALUE:
            if "strategy" not in self.parameters:
                raise ValueError("fill_value requires 'strategy' in parameters")
        return self


class ColumnProfile(BaseModel):
    """Profile of a single dataset column."""
    model_config = {
        "arbitrary_types_allowed": True
    }

    dtype: str
    missing_pct: float = Field(ge=0.0, le=100.0, description="Percentage of missing values (0-100).")
    unique: int = Field(ge=0, description="Number of unique values.")
    example_values: List[Any] = Field(description="Sample values from the column.")


class DetectedIssue(BaseModel):
    """An issue detected in the dataset that needs attention."""
    model_config = {
        "arbitrary_types_allowed": True
    }

    type: str
    column: Optional[str] = None
    severity: Severity


class Progress(BaseModel):
    """Progress tracking for the data cleaning task."""
    model_config = {
        "arbitrary_types_allowed": True
    }

    issues_fixed: int = Field(ge=0)
    issues_remaining: int = Field(ge=0)


class ObservationModel(BaseModel):
    """
    The observation returned by the environment at each step.
    Contains partial views of the dataset and current data quality profiles.
    """
    model_config = {
        "arbitrary_types_allowed": True
    }

    sample_rows: List[Dict[str, Any]] = Field(description="A sample of rows from the dataset.")
    column_profiles: Dict[str, ColumnProfile] = Field(description="Profile for each column in the dataset.")
    detected_issues: List[DetectedIssue] = Field(description="List of detected formatting, type, or missing issues.")
    validation_rules: Dict[str, Any] = Field(description="Schema validation rules available for the dataset.")
    progress: Progress = Field(description="Progress metrics for the episode.")
    step_budget_remaining: int = Field(ge=0, description="Number of remaining steps permitted.")
    action_history_length: int = Field(ge=0, default=0, description="Number of actions taken so far.")


class RewardModel(BaseModel):
    """
    The reward returned by the environment after an action.
    """
    model_config = {
        "arbitrary_types_allowed": True
    }

    value: float = Field(ge=-1.0, le=1.0, description="The primary scalar reward value for the transition.")
    components: Dict[str, float] = Field(
        default_factory=dict, 
        description="Detailed breakdown of the reward."
    )
    reason: str = Field(description="Textual explanation for the given reward.")
