import json
import os
import urllib.error
import urllib.request

from graders import compute_reward

BASE = os.getenv("SOC_GYM_BASE_URL", "http://127.0.0.1:7860")


def post(path, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        BASE + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, {"error": error.read().decode()}
    except urllib.error.URLError as error:
        return None, {"error": f"Unable to reach {BASE}: {error.reason}"}


def is_valid_score(value):
    return isinstance(value, float) and 0.049 < value < 0.951


print(f"=== Validator target: {BASE} ===")

print()
print("=== Test 1: UNWRAPPED action (what OpenEnv validator sends) ===")
reset_status, reset_result = post("/reset", {"task": "brute", "seed": 42})
if reset_status is None:
    print(reset_result["error"])
    raise SystemExit(1)

status, result = post(
    "/step",
    {"type": "block_ip", "target_type": "ip", "target": "203.0.113.5"},
)
score = result.get("reward", {}).get("score", "MISSING")
print(f"HTTP Status : {status}")
print(f"Score       : {score}")
print(f"Valid       : {is_valid_score(score) if isinstance(score, float) else 'FAIL - no score'}")

print()
print("=== Test 2: WRAPPED action (our original format) ===")
post("/reset", {"task": "brute", "seed": 42})
status2, result2 = post(
    "/step",
    {"action": {"type": "block_ip", "target_type": "ip", "target": "203.0.113.5"}},
)
score2 = result2.get("reward", {}).get("score", "MISSING")
print(f"HTTP Status : {status2}")
print(f"Score       : {score2}")
print(f"Valid       : {is_valid_score(score2) if isinstance(score2, float) else 'FAIL - no score'}")

print()
print("=== Test 3: ALL tasks, all steps ===")
tasks = [
    ("brute", [{"type": "block_ip", "target_type": "ip", "target": "203.0.113.5"}]),
    (
        "lateral",
        [
            {"type": "investigate", "target_type": "ip", "target": "203.0.113.5"},
            {"type": "isolate_host", "target_type": "host", "target": "host-12.corp"},
        ],
    ),
    (
        "apt",
        [
            {"type": "investigate", "target_type": "host", "target": "ws-PC042.corp"},
            {"type": "block_ip", "target_type": "ip", "target": "198.51.100.23"},
            {"type": "isolate_host", "target_type": "host", "target": "ws-PC042.corp"},
        ],
    ),
]

all_pass = True
for task_id, actions in tasks:
    post("/reset", {"task": task_id, "seed": 42})
    for i, act in enumerate(actions, 1):
        step_status, response = post("/step", act)
        score_value = response.get("reward", {}).get("score", None)
        ok = is_valid_score(score_value)
        print(
            f"  [{task_id}] step {i}: HTTP={step_status} score={score_value} -> "
            f"{'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            all_pass = False

print()
print("RESULT:", "ALL PASSED" if all_pass else "FAILED")

print()
print("=== Test 4: LOCAL grader boundary checks ===")
boundary_cases = [
    ("lower", compute_reward(0.05, 0.95, 0.05, investigation_bonus=-1.0).score),
    ("upper", compute_reward(0.95, 0.05, 0.95, investigation_bonus=1.0).score),
]
for label, boundary_score in boundary_cases:
    ok = is_valid_score(boundary_score)
    print(f"  [{label}] score={boundary_score} -> {'PASS' if ok else 'FAIL'}")
