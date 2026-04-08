import os
import json
import time
from openai import OpenAI
import openai
from env.core import DataCleanEnv
from env.models import ActionModel, ActionType

client = OpenAI(
    base_url=os.environ.get("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    api_key=os.environ.get("GEMINI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
)
MODEL = os.environ.get("MODEL_NAME", "gemini-2.5-flash")


def create_system_prompt():
    return """You are an autonomous Data Cleaning Agent evaluating a dataset.

Inspect the observation, reason about remaining data issues, and take ONE action.

Valid action types:
- fill_value: Fill missing values. Parameters: {"strategy": "mean"|"median"|"mode"|"constant", "value": "..."} 
- cast_type: Cast column dtype. Parameters: {"dtype": "int"|"float"|"str"}
- normalize_field: Normalize strings. Parameters: {"method": "lowercase"|"strip"}
- drop_row: Drop rows. Parameters: {"condition": "missing"|"outlier"}
- flag_invalid: Flag invalid rows. Parameters: {}
- deduplicate: Remove duplicate rows. Parameters: {}
- handle_outliers: Handle outliers. Parameters: {"method": "clip"|"remove"}
- escalate: Escalate unresolvable issues. Parameters: {}
- finish: End the episode. Parameters: {}

Return STRICT JSON:
{
  "thought": "Brief reasoning about what to fix next.",
  "action": {
    "type": "<action_type>",
    "column": "<column_name or null>",
    "parameters": {}
  }
}
"""


def extract_action(content):
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json\n"):
            content = content[5:]

    try:
        parsed = json.loads(content)
        action_dict = parsed.get("action", {})
        action = ActionModel(**action_dict)
        return action
    except:
        return ActionModel(
            type=ActionType.FINISH,
            column="id",
            parameters={}
        )


def run_task(task_name):
    env = DataCleanEnv(task=task_name, seed=42)
    obs = env.reset()

    print("[START]")
    print(f"task={task_name}")
    print()

    messages = [
        {"role": "system", "content": create_system_prompt()},
        {"role": "user", "content": json.dumps(obs.model_dump(mode="json"))}
    ]

    done = False
    step = 0
    max_steps = 15

    info = {}

    while not done and step < max_steps:
        step += 1

        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                break
            except openai.RateLimitError:
                if attempt < 2:
                    time.sleep(15)
                else:
                    raise

        content = response.choices[0].message.content
        messages.append({"role": "assistant", "content": content})

        action = extract_action(content)

        obs, reward, done, info = env.step(action)

        print("[STEP]")
        print(f"step={step}")
        print(f"action_type={action.type.value}")
        print(f"column={action.column}")
        print(f"reward={reward.value}")
        print()

        if not done:
            messages.append({
                "role": "user",
                "content": json.dumps(obs.model_dump(mode="json"))
            })

    final_score = info.get("final_score", 0.0) if isinstance(info, dict) else 0.0

    print("[END]")
    print(f"task={task_name}")
    print(f"score={final_score}")


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_task(task)