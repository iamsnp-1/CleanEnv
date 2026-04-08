import json
import os
import traceback
from openai import OpenAI
from env.core import DataCleanEnv
from env.models import ActionModel, ActionType

client = OpenAI(
    base_url=os.environ.get("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    api_key=os.environ.get("OPENAI_API_KEY", os.environ.get("HF_TOKEN", ""))
)

MODEL = os.environ.get("MODEL_NAME", "gemini-2.5-flash")

FALLBACK_ACTION = ActionModel(
    type=ActionType.FILL_VALUE,
    column="age",
    parameters={"strategy": "median"}
)


def parse_llm_response(response):
    content = ""
    try:
        choice = response.choices[0]
        message = getattr(choice, "message", None) or choice.get("message", {})
        content = getattr(message, "content", None) or message.get("content", "")
    except Exception:
        content = str(response)

    if not isinstance(content, str):
        content = str(content)

    content = content.strip()
    if content.startswith("```"):
        pieces = content.split("```")
        if len(pieces) >= 2:
            content = pieces[1].strip()
    if content.startswith("json\n"):
        content = content[5:].strip()

    return content


def get_action_from_llm(observation):
    prompt = f"""
You are a data cleaning agent.

Observation:
{observation}

Return ONLY a JSON action with fields:
  type, column, parameters
Example:
{{
  "type": "fill_value",
  "column": "age",
  "parameters": {{"strategy": "median"}}
}}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = parse_llm_response(response)
        action_dict = json.loads(content)
        return ActionModel(**action_dict)
    except Exception as exc:
        print("[WARN] LLM action parsing failed. Using fallback action.")
        print(exc)
        return FALLBACK_ACTION


def run_task(task_name):
    env = DataCleanEnv(task=task_name, seed=42)

    print("[START]")
    print(f"task={task_name}")
    print()

    obs = env.reset()

    done = False
    step = 0
    max_steps = 10

    while not done and step < max_steps:
        step += 1

        try:
            action = get_action_from_llm(obs.model_dump(mode="json"))
        except Exception as exc:
            print("[ERROR] Failed to get action from LLM:", exc)
            print(traceback.format_exc())
            action = FALLBACK_ACTION

        try:
            obs, reward, done, info = env.step(action)
        except Exception as exc:
            print("[ERROR] env.step() failed:", exc)
            print(traceback.format_exc())
            info = {"final_score": 0.0}
            break

        print("[STEP]")
        print(f"step={step}")
        print(f"action_type={action.type.value}")
        print(f"column={action.column}")
        print(f"reward={getattr(reward, 'value', None)}")
        print()

    print("[END]")
    print(f"task={task_name}")
    print(f"score={info.get('final_score', 0.0)}\n")

if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_task(task)