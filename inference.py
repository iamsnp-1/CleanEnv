"""
Baseline inference script for DataCleanEnv-X.
Uses the OpenAI API client to run a model against all 3 tasks.
Emits structured stdout logs in the required [START], [STEP], [END] format.

Required environment variables:
    API_BASE_URL  — The API endpoint for the LLM
    MODEL_NAME    — The model identifier to use
    HF_TOKEN      — Your API key (also used as OPENAI_API_KEY)
"""

import os
import sys
import json
import time
from openai import OpenAI
from env.core import DataCleanEnv
from env.models import ActionModel, ActionType


# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.5-flash")
API_KEY = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


# ---------------------------------------------------------------------------
# System prompt — tells the LLM how to act as a data cleaning agent
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert data cleaning agent. You inspect a dataset observation and decide the single best action to take.

Valid action types and their required parameters:
- fill_value: Fill missing values. Requires column. Parameters: {"strategy": "mean"|"median"|"mode"|"constant", "value": <optional>}
- cast_type: Cast column dtype. Requires column. Parameters: {"dtype": "int"|"float"|"str"}
- normalize_field: Normalize strings. Requires column. Parameters: {"method": "lowercase"|"strip"}
- drop_row: Drop rows. Requires column (or null for all). Parameters: {"condition": "missing"|"outlier"}
- flag_invalid: Flag invalid rows. Requires column. Parameters: {}
- deduplicate: Remove duplicate rows. No column needed. Parameters: {}
- handle_outliers: Handle outliers. Requires column. Parameters: {"method": "clip"|"remove"}
- escalate: Escalate unresolvable issues. No column needed. Parameters: {}
- finish: End the episode when the dataset is sufficiently clean. No column needed. Parameters: {}

Respond with ONLY a JSON object:
{"type": "<action_type>", "column": "<column_name_or_null>", "parameters": {}}

Do NOT include any explanation, markdown, or extra text. Only valid JSON."""


def parse_action(content: str) -> ActionModel:
    """Parse LLM response into an ActionModel, with fallback."""
    content = content.strip()

    # Strip markdown code fences if present
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines).strip()

    try:
        data = json.loads(content)
        return ActionModel(**data)
    except Exception:
        # Fallback: safe no-op action
        return ActionModel(
            type=ActionType.FINISH,
            parameters={},
        )


def call_llm(messages: list, max_retries: int = 3) -> str:
    """Call the LLM with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt * 5
                time.sleep(wait)
            else:
                raise


def run_task(task_name: str) -> float:
    """
    Run a single task and return the final score.
    Emits structured [START], [STEP], [END] logs to stdout.
    """
    env = DataCleanEnv(task=task_name, seed=42)
    obs = env.reset()

    print("[START]")
    print(f"task={task_name}")
    print()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    done = False
    step_num = 0
    max_steps = 10
    info = {}

    while not done and step_num < max_steps:
        step_num += 1

        # Send current observation to LLM
        obs_json = json.dumps(obs.model_dump(mode="json"), default=str)
        messages.append({"role": "user", "content": obs_json})

        # Get LLM response
        content = call_llm(messages)
        messages.append({"role": "assistant", "content": content})

        # Parse action
        action = parse_action(content)

        # Step environment
        obs, reward, done, info = env.step(action)

        print("[STEP]")
        print(f"step={step_num}")
        print(f"action_type={action.type.value}")
        print(f"column={action.column}")
        print(f"reward={reward.value}")
        print()

    # If we exhausted steps without finishing, do a final finish action
    if not done:
        finish_action = ActionModel(type=ActionType.FINISH, parameters={})
        obs, reward, done, info = env.step(finish_action)

        print("[STEP]")
        print(f"step={step_num + 1}")
        print(f"action_type=finish")
        print(f"column=None")
        print(f"reward={reward.value}")
        print()

    final_score = info.get("final_score", 0.0)

    print("[END]")
    print(f"task={task_name}")
    print(f"score={final_score}")
    print()

    return final_score


if __name__ == "__main__":
    scores = {}
    for task in ["easy", "medium", "hard"]:
        scores[task] = run_task(task)

    print("=" * 40)
    print("BASELINE SCORES")
    for task, score in scores.items():
        print(f"  {task}: {score}")
    print("=" * 40)