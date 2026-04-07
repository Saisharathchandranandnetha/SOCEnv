"""metrics.py – batch evaluation utilities for AI SOC Gym.

The module runs many episodes (with a deterministic seed per episode) using a
simple baseline policy that blocks any IP seen in the latest logs.  It then
aggregates the following metrics:

* **avg_steps** – average number of steps taken before termination.
* **success_rate** – fraction of episodes where the attack succeeded
  (i.e., data exfiltrated).
* **mean_detection** – average `detection` component of the reward.
* **mean_false_positive** – average false‑positive penalty.

You can invoke the script directly:
```
python metrics.py
```
which will evaluate both the easy (Brute‑Force) and medium (Lateral‑Movement)
tasks and print a concise report.
"""

import random
from typing import List, Tuple, Dict

from environment import AIGymEnv
from tasks import BruteForceSSHTask, LateralMovementTask, BaseTask
from models import Action, ActionType

# ----------------------------------------------------------------------
# Simple deterministic policy – block any IP observed in the logs.
# ----------------------------------------------------------------------
def block_ip_policy(env) -> Action:
    ip_set = {log.ip for log in env._last_observation.logs if log.ip}
    if ip_set:
        return Action(
            type=ActionType.BLOCK_IP,
            target_type="ip",
            target=next(iter(ip_set)),
        )
    # fallback – allow a dummy IP (no effect)
    return Action(
        type=ActionType.ALLOW,
        target_type="ip",
        target="0.0.0.0",
    )

# ----------------------------------------------------------------------
def run_episode(env: AIGymEnv, policy) -> Tuple[int, bool, float, float]:
    """Run a single episode using *policy* and return metrics.

    Returns:
        steps_used, success_flag, mean_detection, mean_false_positive
    """
    obs = env.reset()
    env._last_observation = obs
    detections: List[float] = []
    false_positives: List[float] = []

    for step in range(1, env.MAX_STEPS + 1):
        action = policy(env)
        obs, reward, done, _ = env.step(action)
        env._last_observation = obs
        detections.append(reward.details.detection)
        false_positives.append(reward.details.false_positive_penalty)
        if done:
            break
    success = env._state.data_exfiltrated
    return step, success, sum(detections) / len(detections), sum(false_positives) / len(false_positives)

# ----------------------------------------------------------------------
def evaluate(task: BaseTask, episodes: int = 30) -> Dict[str, float]:
    seeds = [random.randint(0, 2 ** 31 - 1) for _ in range(episodes)]
    steps, successes, dets, fps = [], [], [], []
    for s in seeds:
        env = AIGymEnv(seed=s)
        env.load_task(task)
        st, succ, det_avg, fp_avg = run_episode(env, block_ip_policy)
        steps.append(st)
        successes.append(succ)
        dets.append(det_avg)
        fps.append(fp_avg)
    return {
        "avg_steps": sum(steps) / len(steps),
        "success_rate": sum(successes) / len(successes),
        "mean_detection": sum(dets) / len(dets),
        "mean_false_positive": sum(fps) / len(fps),
    }

# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Brute‑Force (Task 1) ===")
    print(evaluate(BruteForceSSHTask()))
    print("\n=== Lateral‑Movement (Task 2) ===")
    print(evaluate(LateralMovementTask()))
    # Future: add evaluation for the APT task here.
