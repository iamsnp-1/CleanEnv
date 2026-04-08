import os
import json
from openai import OpenAI
from env.core import DataCleanEnv
from env.models import ActionModel, ActionType

client = OpenAI(
    base_url=os.environ.get("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    api_key=os.environ.get("OPENAI_API_KEY", "")
)

MODEL = os.environ.get("MODEL_NAME", "gemini-2.5-flash")

def get_action_from_llm(observation):
    prompt = f"""
You are a data cleaning agent.

Observation:
{observation}

Return ONLY a JSON action with fields:
type, column, parameters
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content

    # minimal safe parsing
    content = content.strip()

    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json\n"):
            content = content[5:]

    try:
        action_dict = json.loads(content)
        action = ActionModel(**action_dict)
    except Exception:
        action = ActionModel(
            type=ActionType.FILL_VALUE,
            column="age",
            parameters={"strategy": "median"}
        )

    return action

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

        action = get_action_from_llm(obs.model_dump(mode="json"))

        obs, reward, done, info = env.step(action)

        print("[STEP]")
        print(f"step={step}")
        print(f"action_type={action.type.value}")
        print(f"column={action.column}")
        print(f"reward={reward.value}")
        print()

    print("[END]")
    print(f"task={task_name}")
    print(f"score={info.get('final_score', 0.0)}\n")

if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_task(task)