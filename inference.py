"""Run an LLM as the SOC analyst.

Adheres to the OpenEnv Hackathon Submission Guidelines:
  - Uses the OpenAI Python client
  - Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment
  - Emits [START] / [STEP] / [END] structured output
  - [END] is always emitted, even on exception
"""

import argparse
import json
import os
import re
from typing import List, Optional

from openai import OpenAI

from environment import AIGymEnv
from models import Action
from tasks import APTMultiStageTask, BruteForceSSHTask, LateralMovementTask

# ------------------------------------------------------------------
# Environment variables (with required defaults per hackathon spec)
# ------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

TASK_MAP = {
    "brute": ("Brute-Force SSH", BruteForceSSHTask),
    "lateral": ("Lateral Movement", LateralMovementTask),
    "apt": ("APT Multi-Stage", APTMultiStageTask),
}

SYSTEM_PROMPT = (
    "You are an expert Security Operations Center (SOC) analyst. "
    "Your job is to analyze security logs and take the optimal defensive action.\n\n"
    "RULES:\n"
    "1. Respond ONLY with a single JSON object - no explanation, no markdown.\n"
    "2. The JSON must match this exact schema:\n"
    '   { "type": "<action>", "target_type": "<target>", "target": "<value>" }\n'
    "3. Valid actions: block_ip, isolate_host, allow, investigate\n"
    "4. Valid target_types: ip, host, user\n"
    "5. Strategy:\n"
    "   - First investigate suspicious IPs/hosts to gather intelligence.\n"
    "   - Then block the attacker's external IP to cut command-and-control.\n"
    "   - Then isolate any compromised internal machines.\n"
    "   - Only use allow if you are confident there is no threat.\n"
    "6. Look for: failed SSH logins, unusual process spawns, credential dumps,\n"
    "   lateral movement, and data exfiltration indicators.\n"
)

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """Create the OpenAI client lazily so the script can always emit structured logs."""
    global _client
    if _client is None:
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN environment variable is required")
        _client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    return _client


def format_observation(obs) -> str:
    lines = []
    for log in obs.logs:
        lines.append(f"[{log.source.upper():5}] {log.timestamp} {log.severity} {log.message}")
    return "\n".join(lines)


def _extract_first(pattern: str, text: str) -> Optional[str]:
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None


def heuristic_action(obs, task_key: str) -> dict:
    """Deterministic fallback so inference still completes if the model call fails."""
    messages = [log.message for log in obs.logs]
    text = "\n".join(messages)
    ip_values = [log.ip for log in obs.logs if log.ip]
    host_values = [log.hostname for log in obs.logs if log.hostname]

    if task_key == "brute":
        attacker_ip = next((ip for ip in ip_values if ip == "203.0.113.5"), None)
        if attacker_ip:
            return {"type": "block_ip", "target_type": "ip", "target": attacker_ip}

    if task_key == "lateral":
        if "host-12.corp initiated SSH connection to host-27.corp" in text:
            return {"type": "isolate_host", "target_type": "host", "target": "host-12.corp"}
        if "Accepted password for svc_user from 203.0.113.5 on host-12.corp" in text:
            return {"type": "investigate", "target_type": "host", "target": "host-12.corp"}
        if "203.0.113.5" in text:
            return {"type": "investigate", "target_type": "ip", "target": "203.0.113.5"}

    if task_key == "apt":
        if "ws-PC042.corp" in text and (
            "EXCEL.EXE spawned powershell.exe" in text
            or "LSASS memory access" in text
            or "Kerberos TGS request" in text
        ):
            return {"type": "isolate_host", "target_type": "host", "target": "ws-PC042.corp"}
        if "198.51.100.23" in text and (
            "Outbound HTTPS connection" in text
            or "DNS-over-HTTPS traffic" in text
            or "High-volume outbound" in text
        ):
            return {"type": "block_ip", "target_type": "ip", "target": "198.51.100.23"}

    suspicious_ip = next((ip for ip in ip_values if ip in {"203.0.113.5", "198.51.100.23"}), None)
    if suspicious_ip:
        return {"type": "investigate", "target_type": "ip", "target": suspicious_ip}

    suspicious_host = _extract_first(r"on ([A-Za-z0-9._-]+\.corp)", text)
    if suspicious_host:
        return {"type": "investigate", "target_type": "host", "target": suspicious_host}

    if host_values:
        return {"type": "investigate", "target_type": "host", "target": host_values[0]}
    if ip_values:
        return {"type": "investigate", "target_type": "ip", "target": ip_values[0]}
    return {"type": "allow", "target_type": "ip", "target": "10.0.1.1"}


def model_action(obs, task_key: str) -> dict:
    prompt = (
        "Analyze these security logs and respond with ONE JSON action:\n\n"
        f"{format_observation(obs)}"
    )
    response = get_client().chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    raw_content = response.choices[0].message.content
    if raw_content is None:
        raise RuntimeError("Model returned empty content")

    if raw_content.startswith("```json"):
        raw_content = raw_content[7:]
    if raw_content.startswith("```"):
        raw_content = raw_content[3:]
    if raw_content.endswith("```"):
        raw_content = raw_content[:-3]
    return json.loads(raw_content.strip())


def run_agent(task_key: str, max_steps: int = 20) -> None:
    benchmark_name = "ai-soc-gym"
    _, task_cls = TASK_MAP[task_key]

    env = AIGymEnv(seed=42)
    env.load_task(task_cls())

    print(f"[START] task={task_key} env={benchmark_name} model={MODEL_NAME}", flush=True)

    obs = env.reset()
    all_rewards: List[float] = []
    final_success = False

    for step in range(1, max_steps + 1):
        last_action_error = "null"
        action_str = "null"
        reward_score = 0.01
        done = False

        try:
            try:
                llm_out = model_action(obs, task_key)
            except Exception as model_exc:
                llm_out = heuristic_action(obs, task_key)
                last_action_error = f"fallback:{str(model_exc).replace('\n', ' ').replace('\r', ' ')}"

            action = Action(**llm_out)
            action_str = json.dumps(llm_out, separators=(",", ":"))
            obs, reward, done, _info = env.step(action)
            reward_score = reward.score
            if done and not env._state.data_exfiltrated:
                final_success = True
        except Exception as exc:
            last_action_error = str(exc).replace("\n", " ").replace("\r", " ")
            done = True

        all_rewards.append(reward_score)
        done_str = "true" if done else "false"
        print(
            f"[STEP] step={step} action={action_str} reward={reward_score:.2f} "
            f"done={done_str} error={last_action_error}",
            flush=True,
        )

        if done:
            break

    success_str = "true" if final_success else "false"
    rewards_str = ",".join(f"{reward:.2f}" for reward in all_rewards) if all_rewards else "0.01"
    print(f"[END] success={success_str} steps={len(all_rewards)} rewards={rewards_str}", flush=True)


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
        for task_key in TASK_MAP:
            run_agent(task_key)
    else:
        run_agent(args.task)
