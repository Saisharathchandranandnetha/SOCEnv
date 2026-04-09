import urllib.request, json

BASE = "https://netha01-socenv.hf.space"

def post(path, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        BASE + path, data=body,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()}

print("=== Test 1: UNWRAPPED action (what OpenEnv validator sends) ===")
post("/reset", {"task": "brute", "seed": 42})
status, result = post("/step", {"type": "block_ip", "target_type": "ip", "target": "203.0.113.5"})
score = result.get("reward", {}).get("score", "MISSING")
print(f"HTTP Status : {status}")
print(f"Score       : {score}")
print(f"Valid       : {0.0 < score < 1.0 if isinstance(score, float) else 'FAIL - no score'}")

print()
print("=== Test 2: WRAPPED action (our original format) ===")
post("/reset", {"task": "brute", "seed": 42})
status2, result2 = post("/step", {"action": {"type": "block_ip", "target_type": "ip", "target": "203.0.113.5"}})
score2 = result2.get("reward", {}).get("score", "MISSING")
print(f"HTTP Status : {status2}")
print(f"Score       : {score2}")
print(f"Valid       : {0.0 < score2 < 1.0 if isinstance(score2, float) else 'FAIL - no score'}")

print()
print("=== Test 3: ALL tasks, all steps ===")
tasks = [
    ("brute",   [{"type": "block_ip", "target_type": "ip", "target": "203.0.113.5"}]),
    ("lateral", [{"type": "investigate", "target_type": "ip", "target": "203.0.113.5"},
                 {"type": "isolate_host", "target_type": "host", "target": "host-12.corp"}]),
    ("apt",     [{"type": "investigate", "target_type": "host", "target": "ws-PC042.corp"},
                 {"type": "block_ip", "target_type": "ip", "target": "198.51.100.23"},
                 {"type": "isolate_host", "target_type": "host", "target": "ws-PC042.corp"}]),
]

all_pass = True
for task_id, actions in tasks:
    post("/reset", {"task": task_id, "seed": 42})
    for i, act in enumerate(actions, 1):
        s, r = post("/step", act)   # UNWRAPPED - exactly like the validator
        sc = r.get("reward", {}).get("score", None)
        ok = isinstance(sc, float) and 0.0 < sc < 1.0
        print(f"  [{task_id}] step {i}: HTTP={s} score={sc} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False

print()
print("RESULT:", "ALL PASSED" if all_pass else "FAILED")
