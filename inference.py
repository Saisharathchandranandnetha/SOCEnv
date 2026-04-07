"""inference.py – run an LLM as the SOC analyst.

The script creates an `AIGymEnv`, loads a task, and then repeatedly:
1️⃣ Formats the latest observation (list of logs) into a concise prompt.
2️⃣ Sends the prompt to a configured LLM endpoint (OpenAI‑compatible or Claude).
3️⃣ Parses the returned JSON into the `Action` model.
4️⃣ Steps the environment with that action and prints a human‑readable trace.

The LLM must output **exactly** a JSON object matching the `Action` schema:

```json
{ "type": "block_ip" | "isolate_host" | "allow" | "investigate",
  "target_type": "ip" | "host" | "user",
  "target": "<identifier>" }
```

You can set the environment variables `LLM_ENDPOINT` and `LLM_API_KEY`
before launching the script.
"""

import json
import os
from typing import Any, Dict

import httpx  # lightweight HTTP client (already in requirements)

from models import Action, ActionType
from environment import AIGymEnv
from tasks import BruteForceSSHTask, LateralMovementTask, BaseTask

# ----------------------------------------------------------------------
# Helper: turn an Observation into a readable log block for the prompt.
# ----------------------------------------------------------------------
def format_observation(obs) -> str:
    lines = []
    for log in obs.logs:
        lines.append(
            f"[{log.source.upper():5}] {log.timestamp} {log.severity} {log.message}"
        )
    return "\n".join(lines)

# ----------------------------------------------------------------------
# Simple wrapper around the LLM endpoint.
# ----------------------------------------------------------------------
def call_llm(prompt: str) -> Dict[str, Any]:
    endpoint = os.getenv("LLM_ENDPOINT")
    api_key = os.getenv("LLM_API_KEY")
    if not endpoint or not api_key:
        raise RuntimeError("LLM_ENDPOINT and LLM_API_KEY must be set in the environment")

    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 512,
        "system": (
            "You are a SOC analyst. Analyse the logs and output a JSON action "
            "matching the Action schema. Respond ONLY with the JSON object."
        ),
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = httpx.post(endpoint, json=payload, headers=headers, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    # Anthropic Messages API returns content as list of blocks
    raw = data["content"][0]["text"]
    return json.loads(raw)

# ----------------------------------------------------------------------
# Main loop – runs until the environment signals `done`.
# ----------------------------------------------------------------------
def run_agent(env: AIGymEnv, max_steps: int = 30) -> None:
    obs = env.reset()
    print("\n=== INITIAL OBSERVATION ===")
    print(format_observation(obs))

    for step in range(1, max_steps + 1):
        prompt = (
            "You are a SOC analyst. Given the following log entries, decide on a single "
            "action. Respond ONLY with a JSON object matching the Action schema.\n\n"
            f"{format_observation(obs)}"
        )
        try:
            llm_out = call_llm(prompt)
        except Exception as exc:
            print(f"[LLM error] {exc}")
            break

        try:
            action = Action(**llm_out)
        except Exception as exc:
            print(f"[Action parsing error] {exc}")
            break

        obs, reward, done, info = env.step(action)

        print(f"\n--- STEP {step} ---")
        print(f"Action: {action.type.value} ({action.target_type}={action.target})")
        print(f"Info.reason:   {info.reason}")
        print(f"Info.effect:   {info.action_effect}")
        print(f"Reward.score: {reward.score:.3f}")
        print("Logs:")
        print(format_observation(obs))

        if done:
            print("\n*** EPISODE FINISHED ***")
            break

# ----------------------------------------------------------------------
# CLI entry point – pick a task via --task flag.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM SOC agent")
    parser.add_argument(
        "--task",
        choices=["brute", "lateral", "apt"],
        default="lateral",
        help="Which predefined task to load",
    )
    args = parser.parse_args()

    env = AIGymEnv(seed=42)
    if args.task == "brute":
        env.load_task(BruteForceSSHTask())
    elif args.task == "lateral":
        env.load_task(LateralMovementTask())
    else:
        raise NotImplementedError("APT task not implemented yet")

    run_agent(env)
