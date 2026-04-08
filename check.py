#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verification checklist for AI SOC Gym."""
import sys, traceback, os

# Force UTF-8 output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

results = {}

# 1. models.py
try:
    from models import (
        LogEntry, ObservationMetadata, Observation,
        ActionType, Action, RewardDetails, Reward, StepInfo
    )
    results["models.py imports"] = "PASS"
except Exception as e:
    results["models.py imports"] = f"FAIL: {e}"
    traceback.print_exc()

# 2. environment import
try:
    from environment import AIGymEnv
    results["from environment import AIGymEnv"] = "PASS"
except Exception as e:
    results["from environment import AIGymEnv"] = f"FAIL: {e}"
    traceback.print_exc()

# 3. graders import
try:
    from graders import compute_reward
    results["graders.py import"] = "PASS"
except SyntaxError as e:
    results["graders.py import"] = f"SYNTAX ERROR: {e}"
    traceback.print_exc()
except Exception as e:
    results["graders.py import"] = f"FAIL: {e}"
    traceback.print_exc()

# 4. state() method (OpenEnv spec)
try:
    env_test = AIGymEnv(seed=0)
    from tasks import BruteForceSSHTask
    env_test.load_task(BruteForceSSHTask())
    env_test.reset()
    s = env_test.state()
    assert isinstance(s, dict), "state() must return a dict"
    assert "step" in s and "stage" in s, "state() missing keys"
    results["env.state() OpenEnv method"] = "PASS"
except Exception as e:
    results["env.state() OpenEnv method"] = f"FAIL: {e}"
    traceback.print_exc()

# 5. Task 1 - BruteForce (easy)
try:
    env = AIGymEnv(seed=123)
    env.load_task(BruteForceSSHTask())
    obs = env.reset()
    act = Action(type=ActionType.BLOCK_IP, target_type="ip", target="203.0.113.5")
    obs2, rew, done, info = env.step(act)
    results["Task 1: BruteForceSSH"] = f"PASS (reward={rew.score:.3f})"
except Exception as e:
    results["Task 1: BruteForceSSH"] = f"FAIL: {e}"
    traceback.print_exc()

# 6. Task 2 - Lateral Movement (medium)
try:
    from tasks import LateralMovementTask
    env = AIGymEnv(seed=456)
    env.load_task(LateralMovementTask())
    obs = env.reset()
    act = Action(type=ActionType.INVESTIGATE, target_type="ip", target="203.0.113.5")
    obs2, rew, done, info = env.step(act)
    results["Task 2: LateralMovement"] = f"PASS (reward={rew.score:.3f})"
except Exception as e:
    results["Task 2: LateralMovement"] = f"FAIL: {e}"
    traceback.print_exc()

# 7. Task 3 - APT (hard)
try:
    from tasks import APTMultiStageTask
    env = AIGymEnv(seed=789)
    env.load_task(APTMultiStageTask())
    obs = env.reset()
    act = Action(type=ActionType.INVESTIGATE, target_type="host", target="ws-PC042.corp")
    obs2, rew, done, info = env.step(act)
    results["Task 3: APTMultiStage"] = f"PASS (reward={rew.score:.3f})"
except Exception as e:
    results["Task 3: APTMultiStage"] = f"FAIL: {e}"
    traceback.print_exc()

# 8. graders
try:
    r = compute_reward(detection=1.0, false_positive=0.0, efficiency=0.75)
    results["compute_reward()"] = f"PASS (score={r.score:.4f})"
except Exception as e:
    results["compute_reward()"] = f"FAIL: {e}"
    traceback.print_exc()

print("\n" + "="*60)
print("VERIFICATION CHECKLIST RESULTS")
print("="*60)
for k, v in results.items():
    icon = "[OK]" if v.startswith("PASS") else "[!!]"
    print(f"  {icon}  {k}: {v}")

print("\nFile checks:")
for f in ["models.py", "requirements.txt", "openenv.yaml", "README.md",
          "TEAM_README.md", "Dockerfile", "inference.py"]:
    exists = os.path.isfile(f)
    icon = "[OK]" if exists else "[!!]"
    print(f"  {icon}  {f}")

print()
