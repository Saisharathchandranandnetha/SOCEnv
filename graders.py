"""graders.py – reusable reward computation for AI SOC Gym.

The environment itself already contains a private `_compute_reward` method, but the
public `compute_reward` function allows external scripts (e.g., batch
evaluation, custom policies) to obtain exactly the same reward semantics.
"""

from models import Reward, RewardDetails


def compute_reward(
    detection: float,
    false_positive: float,
    efficiency: float,
    investigation_bonus: float = 0.0,
) -> Reward:
    """Calculate a scalar reward (0‑1) with a detailed breakdown.

    Parameters
    ----------
    detection: float – 1 if the attack is mitigated on this step, else 0.
    false_positive: float – penalty for blocking/isolating benign traffic.
    efficiency: float – higher when fewer steps are used (scaled 0‑1).
    investigation_bonus: float – optional extra boost for a "investigate‑first"
        strategy (used by the medium task).
    """
    base = max(0.0, min(1.0, (detection - false_positive + efficiency) / 3.0))
    overall = min(1.0, base + investigation_bonus)
    details = RewardDetails(
        detection=detection,
        false_positive_penalty=false_positive,
        efficiency=efficiency,
    )
    return Reward(score=overall, details=details)
