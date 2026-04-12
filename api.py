"""
DataCleanEnv-X (DCX) — FastAPI server for OpenEnv-compliant data cleaning environment.
Exposes /reset, /step, /state endpoints as required by the OpenEnv spec.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from env.core import DataCleanEnv
from env.models import ActionModel

app = FastAPI(
    title="DataCleanEnv-X",
    description="OpenEnv-compliant data cleaning simulation environment",
    version="1.0.0",
)

# Global environment instance — task is set on reset
env: DataCleanEnv = None


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "environment": "DataCleanEnv-X",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state"],
    }


@app.api_route("/reset", methods=["GET", "POST"])
def reset(task: str = Query(default="easy", description="Task name: easy, medium, or hard")):
    """
    Reset the environment and return the initial observation.
    Supports both GET and POST for validator compatibility.
    """
    global env
    if task not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail=f"Invalid task '{task}'. Must be one of: easy, medium, hard")

    env = DataCleanEnv(task=task, seed=42)
    obs = env.reset()
    return obs.model_dump(mode="json")


@app.post("/step")
def step(action: dict):
    """
    Execute a single action and return observation, reward, done, info.
    """
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    try:
        action_model = ActionModel(**action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid action: {str(e)}")

    obs, reward, done, info = env.step(action_model)

    return {
        "observation": obs.model_dump(mode="json"),
        "reward": reward.model_dump(mode="json"),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    """Return the current internal state of the environment."""
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    return env.state()


def serve():
    """Entry point for [project.scripts] — launches the FastAPI server."""
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=7860)
