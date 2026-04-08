---
title: DataCleanEnv-X
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# DataCleanEnv-X (DCX)

An OpenEnv-compliant simulation environment where AI agents clean real-world structured datasets. Agents must identify and fix data quality issues — missing values, type errors, duplicates, outliers, and inconsistent formatting — through structured actions.

## Motivation

Data cleaning is one of the most time-consuming tasks in data science, accounting for up to 80% of a data scientist's work. This environment provides a controlled benchmark to evaluate how well AI agents can automate this process, with progressively harder challenges.

## Tasks

| Task | Difficulty | Description | Key Challenges |
|------|-----------|-------------|----------------|
| `easy` | ⭐ Easy | Fix missing values and type errors in a small employee dataset (200 rows) | Missing NaN values, non-numeric strings in numeric columns |
| `medium` | ⭐⭐ Medium | Deduplicate, fill gaps, and normalize formatting (500 rows) | Duplicate rows, missing categories, inconsistent string casing/whitespace |
| `hard` | ⭐⭐⭐ Hard | Full cleaning pipeline across all issue types (1000 rows) | Missing values, type errors, duplicates, outliers, invalid categories, impossible ages |

## Observation Space

Each observation (`ObservationModel`) contains:

| Field | Type | Description |
|-------|------|-------------|
| `sample_rows` | `List[Dict]` | Up to 20 sample rows from the working dataset |
| `column_profiles` | `Dict[str, ColumnProfile]` | Per-column stats: dtype, missing %, unique count, example values |
| `detected_issues` | `List[DetectedIssue]` | Issues found: type (missing/duplicate/outlier/etc.), column, severity |
| `validation_rules` | `Dict[str, Any]` | Schema rules for the dataset |
| `progress` | `Progress` | Issues fixed vs. remaining |
| `step_budget_remaining` | `int` | Steps left before episode ends |
| `action_history_length` | `int` | Number of actions taken so far |

## Action Space

Each action (`ActionModel`) has three fields:

```json
{
  "type": "<action_type>",
  "column": "<column_name or null>",
  "parameters": {}
}
```

### Supported Actions

| Action | Column Required | Parameters | Description |
|--------|----------------|------------|-------------|
| `fill_value` | ✅ | `strategy`: mean/median/mode/constant; `value`: optional | Fill missing values |
| `cast_type` | ✅ | `dtype`: int/float/str | Cast column to target type |
| `normalize_field` | ✅ | `method`: lowercase/strip | Normalize string formatting |
| `drop_row` | Optional | `condition`: missing/outlier | Drop rows by condition |
| `flag_invalid` | ✅ | — | Flag rows with invalid values |
| `deduplicate` | ❌ | — | Remove duplicate rows |
| `handle_outliers` | ✅ | `method`: clip/remove | Handle statistical outliers |
| `escalate` | ❌ | — | Escalate unresolvable issues |
| `finish` | ❌ | — | End the episode |

## Reward Function

The reward function provides signal over the full trajectory:

- **Positive reward**: Proportional to the percentage of issues fixed by an action
- **No-op penalty** (-0.05): Actions that don't reduce any issues
- **Destructive penalty** (-0.2): Actions that increase the issue count
- **Finish bonus** (+0.1): Finishing with zero issues remaining
- **Premature finish penalty** (-0.1): Finishing with many issues remaining

Reward range: `[-1.0, 1.0]`

## Setup & Usage

### Requirements

```bash
pip install -r requirements.txt
```

### Run Locally

```bash
# Start the API server
uvicorn api:app --host 0.0.0.0 --port 7860

# Test health check
curl http://localhost:7860/

# Reset environment
curl http://localhost:7860/reset?task=easy
```

### Docker

```bash
docker build -t datacleanenv-x .
docker run -p 7860:7860 datacleanenv-x
```

### Run Inference

Set environment variables and run:

```bash
export API_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
export MODEL_NAME="gemini-2.5-flash"
export HF_TOKEN="your-api-key-here"

python inference.py
```

### Expected Output Format

```
[START]
task=easy

[STEP]
step=1
action_type=fill_value
column=age
reward=0.15

[STEP]
step=2
action_type=cast_type
column=age
reward=0.1

[END]
task=easy
score=0.75
```

## Baseline Scores

| Task | Expected Score Range |
|------|---------------------|
| Easy | 0.5 – 0.9 |
| Medium | 0.3 – 0.7 |
| Hard | 0.2 – 0.6 |

*Actual scores depend on the LLM model used and its ability to reason about data quality issues.*

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `API_BASE_URL` | LLM API endpoint | ✅ |
| `MODEL_NAME` | Model identifier for inference | ✅ |
| `HF_TOKEN` | API key for authentication | ✅ |

## Project Structure

```
CleanEnv/
├── api.py                 # FastAPI server (/reset, /step, /state)
├── inference.py           # Baseline inference script
├── openenv.yaml           # OpenEnv spec metadata
├── Dockerfile             # Container configuration
├── requirements.txt       # Python dependencies
├── env/
│   ├── core.py            # DataCleanEnv environment class
│   ├── models.py          # Pydantic models (Observation, Action, Reward)
│   └── state_manager.py   # Internal state and issue detection
├── tasks/
│   ├── dataset_loader.py  # Synthetic dataset generation
│   ├── corruption_engine.py # Data corruption logic
│   ├── easy.py            # Easy task configuration
│   ├── medium.py          # Medium task configuration
│   └── hard.py            # Hard task configuration
├── graders/
│   ├── easy_grader.py     # Easy task scoring
│   ├── medium_grader.py   # Medium task scoring
│   ├── hard_grader.py     # Hard task scoring
│   └── utils.py           # Shared grading utilities
└── reward/
    └── reward_engine.py   # Trajectory-aware reward function
```
