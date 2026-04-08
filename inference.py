'''
Baseline inference script for DataCleanEnv-X.
Uses the OpenAI API client to run a model against all 3 tasks.
Emits structured stdout logs in the required [START], [STEP], [END] format.

Required environment variables:
    API_BASE_URL  - The API endpoint for the LLM
    MODEL_NAME    - The model identifier to use
    HF_TOKEN      - Your API key (also used as OPENAI_API_KEY)
'''

import json
import os
import time
import traceback
from openai import OpenAI
from env.core import DataCleanEnv
from env.models import ActionModel, ActionType

API_BASE_URL = os.environ.get(
    "API_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.5-flash")
API_KEY = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = '''You are an expert data cleaning agent. You inspect a dataset observation and choose the single best next action to clean the data.

Return ONLY a JSON object with the following fields: type, column, parameters.
Valid action types:
- fill_value: Requires column. Parameters: {"strategy": "mean"|"median"|"mode"|"constant", "value": <optional>}
- cast_type: Requires column. Parameters: {"dtype": "int"|"float"|"str"}
- normalize_field: Requires column. Parameters: {"method": "lowercase"|"strip"}
- drop_row: Requires column. Parameters: {"condition": "missing"|"outlier"}
- flag_invalid: Requires column. Parameters: {}
- deduplicate: No column required. Parameters: {}
- handle_outliers: Requires column. Parameters: {"method": "clip"|"remove"}
- escalate: No column required. Parameters: {}
- finish: No column required. Parameters: {}

Do not include any explanation, markdown, or extra text. Only valid JSON.'''

FALLBACK_ACTION = ActionModel(
    type=ActionType.FILL_VALUE,
    column="age",
    parameters={"strategy": "median"}
)

def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```") and text.count("```") >= 2:
        parts = text.split("```")
        content = parts[1].strip()
        return content
    return text


def parse_action(content: str) -> ActionModel:
    content = strip_code_fence(content)
    try:
        action_dict = json.loads(content)
        return ActionModel(**action_dict)
    except Exception as exc:
        print("[WARN] Failed to parse LLM action response, using fallback action:", exc)
        return FALLBACK_ACTION


def call_llm(messages: list, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
            )
            content = getattr(response.choices[0].message, "content", None)
            if content is None:
                content = response.choices[0].get("message", {}).get("content", "")
            return content
        except Exception as exc:
            print(f"[WARN] LLM call failed (attempt {attempt + 1}): {exc}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise


def run_task(task_name: str) -> float:
    env = DataCleanEnv(task=task_name, seed=42)

    print("[START]")
    print(f"task={task_name}")
    print()

    try:
        obs = env.reset()
    except Exception as exc:
        print("[ERROR] env.reset() failed:", exc)
        print(traceback.format_exc())
        return 0.0

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    done = False
    step_num = 0
    max_steps = 10
    info = {}

    while not done and step_num < max_steps:
        step_num += 1

        try:
            obs_json = json.dumps(obs.model_dump(mode="json"), default=str)
        except Exception as exc:
            print("[ERROR] Failed to serialize observation:", exc)
            print(traceback.format_exc())
            obs_json = "{}"

        messages.append({"role": "user", "content": obs_json})

        try:
            content = call_llm(messages)
        except Exception as exc:
            print("[ERROR] LLM request failed, using fallback action:", exc)
            print(traceback.format_exc())
            action = FALLBACK_ACTION
        else:
            action = parse_action(content)
            messages.append({"role": "assistant", "content": content})

        try:
            obs, reward, done, info = env.step(action)
        except Exception as exc:
            print("[ERROR] env.step() failed:", exc)
            print(traceback.format_exc())
            info = {"final_score": 0.0}
            break

        print("[STEP]")
        print(f"step={step_num}")
        print(f"action_type={action.type.value}")
        print(f"column={action.column}")
        print(f"reward={getattr(reward, 'value', None)}")
        print()

    if not done:
        finish_action = ActionModel(type=ActionType.FINISH, parameters={})
        try:
            obs, reward, done, info = env.step(finish_action)
        except Exception as exc:
            print("[ERROR] final finish action failed:", exc)
            print(traceback.format_exc())
            info = {"final_score": 0.0}

        print("[STEP]")
        print(f"step={step_num + 1}")
        print("action_type=finish")
        print("column=None")
        print(f"reward={getattr(reward, 'value', None)}")
        print()

    final_score = info.get("final_score", 0.0)

    print("[END]")
    print(f"task={task_name}")
    print(f"score={final_score}")
    print()

    return final_score


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_task(task)
