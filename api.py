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

@app.get("/reset")from fastapi import FastAPI, HTTPException
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

# Support BOTH GET and POST (validator uses POST)
@app.api_route("/reset", methods=["GET", "POST"])
def reset():
    obs = env.reset()
    return obs.model_dump(mode="json")  # ensures JSON-safe output

@app.post("/step")
def step(action: dict):
    try:
        action_model = ActionModel(**action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    obs, reward, done, info = env.step(action_model)

    return {
        "observation": obs.model_dump(mode="json"),  # JSON-safe
        "reward": reward.model_dump(),
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    return env.state()  # already serialized in your StateManager
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
