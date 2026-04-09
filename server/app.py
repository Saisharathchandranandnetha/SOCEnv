"""app.py - FastAPI server exposing the AI SOC Gym environment.

Keeps the HuggingFace Space alive on port 7860 and exposes
the OpenEnv interface via HTTP so agents can interact remotely.
"""

import sys
import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, Dict, Any

# Ensure we can import from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# ---------- Helper: clamp all floats in reward to strictly (0, 1) ----------

def _clamp(v: float) -> float:
    """Clamp to the same strict reward bounds used by the local grader."""
    return max(0.05, min(0.95, float(v)))

def _safe_reward(reward_dict: dict) -> dict:
    """Recursively clamp all float values in a reward dict to strictly (0.05, 0.95).
    
    The OpenEnv validator calls /step via REST and reads the reward.score directly
    from the JSON response. This is the definitive safety net at the API boundary.
    """
    result = {}
    for k, v in reward_dict.items():
        if isinstance(v, float):
            result[k] = _clamp(v)
        elif isinstance(v, dict):
            result[k] = _safe_reward(v)
        else:
            result[k] = v
    return result

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
async def reset(request: Request):
    global _env
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}

    task_name = body.get("task", "brute")
    seed = body.get("seed", 42)

    task_cls = TASK_MAP.get(task_name)
    if task_cls is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task: {task_name}. Choose from {list(TASK_MAP.keys())}"
        )

    _env = AIGymEnv(seed=seed)
    _env.load_task(task_cls())
    obs = _env.reset()
    return {
        "logs": [log.model_dump() for log in obs.logs],
        "metadata": obs.metadata.model_dump(),
    }

@app.post("/step")
async def step(request: Request):
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    try:
        body = await request.json()
    except Exception:
        body = {}

    # Accept BOTH formats:
    # 1. {"action": {"type": ..., "target_type": ..., "target": ...}}  (our format)
    # 2. {"type": ..., "target_type": ..., "target": ...}              (OpenEnv validator format)
    if isinstance(body, dict) and "action" in body:
        action_data = body["action"]
    elif isinstance(body, dict) and "type" in body:
        action_data = body  # action sent directly
    else:
        raise HTTPException(status_code=422, detail="Missing required 'action' in JSON body")

    try:
        action = Action(**action_data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")

    obs, reward, done, info = _env.step(action)

    # CRITICAL: Clamp all reward floats to strictly (0.05, 0.95) at the API
    # boundary. The OpenEnv validator reads reward.score directly from this
    # JSON response — any exact 0.0 or 1.0 will fail Task Validation.
    reward_dict = _safe_reward(reward.model_dump())

    return {
        "observation": {
            "logs": [log.model_dump() for log in obs.logs],
            "metadata": obs.metadata.model_dump(),
        },
        "reward": reward_dict,
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

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
