"""inference.py – run an LLM as the SOC analyst.

Adheres to the OpenEnv Hackathon Submission Guidelines:
  - Uses the OpenAI Python client
  - Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment
  - Emits [START] / [STEP] / [END] structured output
  - [END] is ALWAYS emitted, even on exception
"""

import os
import json
import argparse
from typing import Any, Dict, List

from openai import OpenAI

from models import Action
from environment import AIGymEnv
from tasks import BruteForceSSHTask, LateralMovementTask, APTMultiStageTask

# ------------------------------------------------------------------
# Environment variables (with required defaults per hackathon spec)
# ------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Initialize OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ------------------------------------------------------------------
# Task registry
# ------------------------------------------------------------------
TASK_MAP = {
    "brute": ("Brute-Force SSH", BruteForceSSHTask),
    "lateral": ("Lateral Movement", LateralMovementTask),
    "apt": ("APT Multi-Stage", APTMultiStageTask),
}

SYSTEM_PROMPT = (
    "You are an expert Security Operations Center (SOC) analyst. "
    "Your job is to analyze security logs and take the optimal defensive action.\n\n"
    "RULES:\n"
    "1. Respond ONLY with a single JSON object — no explanation, no markdown.\n"
    "2. The JSON must match this exact schema:\n"
    '   { "type": "<action>", "target_type": "<target>", "target": "<value>" }\n'
    "3. Valid actions: block_ip, isolate_host, allow, investigate\n"
    "4. Valid target_types: ip, host, user\n"
    "5. STRATEGY:\n"
    "   - First INVESTIGATE suspicious IPs/hosts to gather intelligence.\n"
    "   - Then BLOCK_IP the attacker's external IP to cut command-and-control.\n"
    "   - Then ISOLATE_HOST any compromised internal machines.\n"
    "   - Only use ALLOW if you are confident there is no threat.\n"
    "6. Look for: failed SSH logins, unusual process spawns (powershell, rundll32),\n"
    "   lateral movement (internal SSH hops), credential dumps (LSASS access),\n"
    "   and data exfiltration (large outbound transfers, DNS tunneling).\n"
)


def format_observation(obs) -> str:
    lines = []
    for log in obs.logs:
        lines.append(
            f"[{log.source.upper():5}] {log.timestamp} {log.severity} {log.message}"
        )
    return "\n".join(lines)


# ------------------------------------------------------------------
# Main agent loop — one episode
# ------------------------------------------------------------------
def run_agent(task_key: str, max_steps: int = 30) -> None:
    task_label, task_cls = TASK_MAP[task_key]
    benchmark_name = "ai-soc-gym"

    env = AIGymEnv(seed=42)
    env.load_task(task_cls())

    # 1. [START]
    print(
        f"[START] task={task_key} env={benchmark_name} model={MODEL_NAME}",
        flush=True,
    )

    obs = env.reset()
    all_rewards: List[float] = []
    final_success = False

    try:
        for step in range(1, max_steps + 1):
            prompt = (
                "Analyze these security logs and respond with ONE JSON action:\n\n"
                f"{format_observation(obs)}"
            )

            last_action_error = "null"
            action_str = "null"
            reward_score = 0.00
            is_done = False

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                raw_content = response.choices[0].message.content

                # Strip markdown code fences if present
                if raw_content.startswith("```json"):
                    raw_content = raw_content[7:]
                if raw_content.startswith("```"):
                    raw_content = raw_content[3:]
                if raw_content.endswith("```"):
                    raw_content = raw_content[:-3]
                raw_content = raw_content.strip()

                llm_out = json.loads(raw_content)
                action = Action(**llm_out)
                action_str = json.dumps(llm_out, separators=(",", ":"))

                obs, reward, done, info = env.step(action)
                reward_score = reward.score
                is_done = done

                # Check if the defender successfully mitigated the attack
                if done and not env._state.data_exfiltrated:
                    final_success = True

            except Exception as exc:
                last_action_error = str(exc).replace("\n", " ").replace("\r", "")
                is_done = True

            all_rewards.append(reward_score)

            # 2. [STEP]
            done_str = "true" if is_done else "false"
            print(
                f"[STEP] step={step} action={action_str} reward={reward_score:.2f} "
                f"done={done_str} error={last_action_error}",
                flush=True,
            )

            if is_done:
                break

    except Exception:
        pass  # Ensure [END] is always printed below

    # 3. [END] — ALWAYS emitted, even on exception
    success_str = "true" if final_success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in all_rewards) if all_rewards else "0.00"
    print(
        f"[END] success={success_str} steps={len(all_rewards)} rewards={rewards_str}",
        flush=True,
    )


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM SOC agent")
    parser.add_argument(
        "--task",
        choices=["brute", "lateral", "apt", "all"],
        default="all",
        help="Which task to evaluate (default: all)",
    )
    args = parser.parse_args()

    if args.task == "all":
        for key in TASK_MAP:
            run_agent(key)
    else:
        run_agent(args.task)
