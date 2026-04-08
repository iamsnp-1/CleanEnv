from fastapi import FastAPI
from env.core import DataCleanEnv
from env.models import ActionModel

app = FastAPI()
env = DataCleanEnv(task="easy", seed=42)

@app.get("/")
def root():
    return {
        "status": "DataCleanEnv-X API is running",
        "endpoints": ["/reset", "/step", "/state"]
    }

@app.get("/reset")
def reset():
    obs = env.reset()
    return obs.model_dump()

@app.post("/step")
def step(action: dict):
    action_model = ActionModel(**action)
    obs, reward, done, info = env.step(action_model)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    return env.state()
