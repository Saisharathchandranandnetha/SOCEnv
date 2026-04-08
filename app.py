"""app.py - FastAPI server exposing the AI SOC Gym environment.

Keeps the HuggingFace Space alive on port 7860 and exposes
the OpenEnv interface via HTTP so agents can interact remotely.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from environment import AIGymEnv
from tasks import BruteForceSSHTask, LateralMovementTask, APTMultiStageTask
from models import Action

app = FastAPI(
    title="AI SOC Gym",
    description="RL environment where LLM agents defend against real cyberattacks.",
    version="1.0.0",
)

# Global environment instance
_env: Optional[AIGymEnv] = None

TASK_MAP = {
    "brute":   BruteForceSSHTask,
    "lateral": LateralMovementTask,
    "apt":     APTMultiStageTask,
}

# ---------- Request/Response schemas ----------

class ResetRequest(BaseModel):
    task: str = "brute"
    seed: Optional[int] = 42

class StepRequest(BaseModel):
    action: Dict[str, Any]

# ---------- Endpoints ----------

@app.get("/")
def root():
    return {
        "name": "AI SOC Gym",
        "version": "1.0.0",
        "tasks": list(TASK_MAP.keys()),
        "status": "running",
    }

@app.post("/reset")
def reset(req: ResetRequest):
    global _env
    task_cls = TASK_MAP.get(req.task)
    if task_cls is None:
        raise HTTPException(status_code=400, detail=f"Unknown task: {req.task}. Choose from {list(TASK_MAP.keys())}")
    _env = AIGymEnv(seed=req.seed)
    _env.load_task(task_cls())
    obs = _env.reset()
    return {
        "logs": [log.model_dump() for log in obs.logs],
        "metadata": obs.metadata.model_dump(),
    }

@app.post("/step")
def step(req: StepRequest):
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    try:
        action = Action(**req.action)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")
    obs, reward, done, info = _env.step(action)
    return {
        "observation": {
            "logs": [log.model_dump() for log in obs.logs],
            "metadata": obs.metadata.model_dump(),
        },
        "reward": reward.model_dump(),
        "done": done,
        "info": info.model_dump(),
    }

@app.get("/state")
def state():
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return _env.state()

@app.get("/health")
def health():
    return {"status": "ok"}
