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
        raise HTTPException(status_code=400, detail=f"Unknown task: {task_name}. Choose from {list(TASK_MAP.keys())}")
    
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
        
    if not isinstance(body, dict) or "action" not in body:
        raise HTTPException(status_code=422, detail="Missing required 'action' in JSON body")
        
    try:
        action = Action(**body["action"])
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

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
