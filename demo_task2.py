"""demo_task2.py – quick end‑to‑end run of the Lateral‑Movement scenario.

The script builds an `AIGymEnv`, loads the `LateralMovementTask`, and then
steps through a tiny deterministic agent that follows the optimal play‑
book:

1️⃣ Investigate the attacker IP (gives a small bonus).
2️⃣ Block the attacker IP (partial mitigation).
3️⃣ Isolate the pivot host (full mitigation, ends the episode).

The loop prints the action taken, the environment’s explanatory `info`,
the reward score, and the logs generated at each step.  It mirrors the
demo you saw for the easy scenario, but now showcases multi‑entity
reasoning and the delayed‑consequence mechanic.
"""

from environment import AIGymEnv
from tasks import LateralMovementTask
from models import Action, ActionType


def ideal_agent(obs):
    """Very simple heuristic agent used only for the demo.
    It looks for the attacker IP in the current logs; if it sees the
    pivot host name it isolates it; otherwise it falls back to blocking the
    IP.
    """
    ip_set = {log.ip for log in obs.logs if log.ip}
    if "203.0.113.5" in ip_set:
        # First step – investigate the IP to collect the bonus.
        return Action(
            type=ActionType.INVESTIGATE,
            target_type="ip",
            target="203.0.113.5",
        )
    # Once the IP is blocked we will start seeing the pivot host name.
    host_set = {log.hostname for log in obs.logs if log.hostname}
    if "host-12.corp" in host_set:
        return Action(
            type=ActionType.ISOLATE_HOST,
            target_type="host",
            target="host-12.corp",
        )
    # Default – block the malicious IP.
    return Action(
        type=ActionType.BLOCK_IP,
        target_type="ip",
        target="203.0.113.5",
    )


def run_demo():
    env = AIGymEnv(seed=123)  # deterministic for the demo
    env.load_task(LateralMovementTask())

    obs = env.reset()
    print("=== RESET ===")
    for log in obs.logs:
        print(f"[{log.source}] {log.message}")

    for step in range(1, 12):
        act = ideal_agent(obs)
        obs, reward, done, info = env.step(act)

        print(f"\n--- STEP {step} ---")
        print(
            f"Action: {act.type.value} ({act.target_type}={act.target})"
        )
        print(f"Info.reason:   {info.reason}")
        print(f"Info.effect:   {info.action_effect}")
        print(f"Reward.score: {reward.score:.3f}")
        for log in obs.logs:
            print(f"[{log.source}] {log.message}")

        if done:
            print("\n*** EPISODE FINISHED ***")
            break


if __name__ == "__main__":
    run_demo()
