"""demo_task3.py – Real-time visual demo of the APT Multi-Stage (Hard) task.

Streams logs step-by-step with pauses between phases to simulate
a live SOC terminal feed.
"""

import sys
import time
import io

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from environment import AIGymEnv
from tasks import APTMultiStageTask
from models import Action, ActionType

SEPARATOR = "-" * 62

SEVERITY_LABEL = {
    "INFO":     " INFO    ",
    "WARN":     " WARN    ",
    "CRITICAL": " CRITICAL",
}

def print_log(log):
    sev = SEVERITY_LABEL.get(log.severity, log.severity)
    print(f"  [{log.source.upper():<8}]{sev}: {log.message}")


def heuristic_agent(step: int):
    """Optimal SOC playbook for APT mitigation."""
    if step == 1:
        return Action(type=ActionType.INVESTIGATE, target_type="ip",   target="198.51.100.23")
    elif step == 2:
        return Action(type=ActionType.BLOCK_IP,    target_type="ip",   target="198.51.100.23")
    elif step == 3:
        return Action(type=ActionType.INVESTIGATE, target_type="host", target="ws-PC042.corp")
    elif step == 4:
        return Action(type=ActionType.ISOLATE_HOST, target_type="host", target="ws-PC042.corp")
    return Action(type=ActionType.ALLOW, target_type="ip", target="0.0.0.0")


def run_demo():
    env = AIGymEnv(seed=789)
    env.load_task(APTMultiStageTask())
    obs = env.reset()

    print()
    print("=" * 62)
    print("  AI SOC GYM  |  APT MULTI-STAGE DEMO  |  HARD TASK")
    print("=" * 62)
    print()
    time.sleep(0.5)

    print("[*] EPISODE STARTED — waiting for first log batch...")
    time.sleep(1)
    print()
    print("=== INITIAL LOGS (RESET) ===")
    for log in obs.logs:
        print_log(log)
        time.sleep(0.07)

    time.sleep(1.2)

    for step in range(1, 6):
        act = heuristic_agent(step)
        obs, reward, done, info = env.step(act)

        print()
        print(SEPARATOR)
        print(f"  STEP {step}")
        print(SEPARATOR)
        time.sleep(0.3)

        print(f"  [>] ACTION  : {act.type.value.upper()}  |  {act.target_type} = {act.target}")
        time.sleep(0.2)
        print(f"  [>] REWARD  : {reward.score:.3f}")
        time.sleep(0.2)
        print(f"  [>] REASON  : {info.reason}")
        time.sleep(0.2)
        print(f"  [>] EFFECT  : {info.action_effect}")
        time.sleep(0.8)

        print()
        print("  [*] Streaming new log events...")
        time.sleep(0.5)

        for log in obs.logs:
            print_log(log)
            time.sleep(0.08)

        if done:
            time.sleep(0.5)
            print()
            print("=" * 62)
            print("  *** THREAT NEUTRALIZED — EPISODE FINISHED ***")
            print(f"  Total steps taken: {step}")
            print(f"  Final reward:      {reward.score:.3f}")
            print("=" * 62)
            print()
            break

        time.sleep(1.2)


if __name__ == "__main__":
    run_demo()
