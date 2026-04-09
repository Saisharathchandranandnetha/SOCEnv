"""Reusable reward computation for AI SOC Gym.

The public ``compute_reward`` function is the canonical place where task
scores are normalized so they always remain strictly inside ``(0, 1)``.
"""

from models import Reward, RewardDetails


MIN_SCORE = 0.01
MAX_SCORE = 0.99


def _strict_unit_interval(value: float) -> float:
    """Clamp numeric values to the reward contract required by the validator."""
    return max(MIN_SCORE, min(MAX_SCORE, float(value)))


def compute_reward(
    detection: float,
    false_positive: float,
    efficiency: float,
    investigation_bonus: float = 0.0,
) -> Reward:
    """Calculate a scalar reward with every component strictly inside ``(0, 1)``.

    Parameters
    ----------
    detection: float - 1 if the attack is mitigated on this step, else 0.
    false_positive: float - penalty for blocking/isolating benign traffic.
    efficiency: float - higher when fewer steps are used (scaled 0-1).
    investigation_bonus: float - optional extra boost for an investigate-first
        strategy.
    """
    detection = _strict_unit_interval(detection)
    false_positive = _strict_unit_interval(false_positive)
    efficiency = _strict_unit_interval(efficiency)
    investigation_bonus = max(0.0, float(investigation_bonus))

    base = _strict_unit_interval((detection - false_positive + efficiency) / 3.0)
    overall = _strict_unit_interval(base + investigation_bonus)
    details = RewardDetails(
        detection=detection,
        false_positive_penalty=false_positive,
        efficiency=efficiency,
    )
    return Reward(score=overall, details=details)
