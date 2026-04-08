# AI SOC Gym - Team Internal Documentation

## What Problem Does This Actually Solve?

SOC automation is incredibly difficult because real security analysis requires:
- **Temporal reasoning**: Connecting events that happen hours/days apart
- **Causal inference**: Determining if symptom A caused symptom B or they're unrelated
- **Partial observability**: Never seeing the attacker's true state, only indirect effects
- **Cost-sensitive decisions**: Blocking an IP might stop an attack but also break payroll for 500 people

AI SOC Gym isolates the core cognitive challenge: interpreting noisy, partial log streams to make timely intervention decisions.

## How the Code is Structured

### environment.py - The Core Loop
- **Does**: Implements `reset()`, `step()`, and `state()` (OpenEnv spec), manages hidden `AttackState`, generates realistic logs, computes rewards
- **Key methods**:
  - `reset()` -> Observation
  - `step(action)` -> (Observation, Reward, bool, StepInfo)
  - `state()` -> dict (serializable snapshot for OpenEnv validation)
- **Watch for**: The `_state` variable contains the ground truth that agents must infer from logs

### tasks.py - Attack Scenarios
- **Does**: Defines `BaseTask` interface and three concrete scenarios:
  1. `BruteForceSSHTask` (Easy) - Single IP brute-force
  2. `LateralMovementTask` (Medium) - Pivot + internal hop
  3. `APTMultiStageTask` (Hard) - Full kill-chain: phishing -> cred theft -> lateral movement -> exfiltration
- **Watch for**: Each task's `advance()` method returns logs that reveal partial attack progress

### models.py - Type Safety Layer
- **Does**: Provides Pydantic schemas ensuring type safety between environment, agent, and graders
- **Watch for**: All inter-module communication uses these models; mismatch causes validation errors

### inference.py - LLM Bridge (OpenEnv Compliant)
- **Does**: Connects any OpenAI-compatible LLM to the environment using the `openai` Python client
- **Environment variables**:
  - `HF_TOKEN` (required) - Hugging Face API token
  - `API_BASE_URL` (default: `https://api.openai.com/v1`)
  - `MODEL_NAME` (default: `gpt-4.1-mini`)
- **Output format**: Strict `[START]` / `[STEP]` / `[END]` tags for automated grading
- **Watch for**: No debug prints allowed - only the structured output tags

### graders.py & metrics.py - Evaluation Tools
- **Does**: Provide reusable reward computation and batch evaluation across all 3 tasks
- **Watch for**: `metrics.py` uses a monkey-patch to access `_last_observation` for the baseline policy

## The Hidden State Mechanic

The `AttackState` in `environment.py` represents the complete attacker progression:
- `stage`: "initial_access" -> "lateral_movement" -> "exfiltration"
- `compromised_hosts`: List of systems under attacker control
- `attacker_ips`: External IPs used in the attack
- `data_exfiltrated`: Boolean indicating breach completion

**Crucially**: The agent NEVER sees this directly. Instead, they observe:
- Login attempts revealing `attacker_ips`
- New logins from compromised IPs showing lateral movement
- Service creation/logins on newly compromised hosts
- Unusual outbound connections suggesting exfiltration

## How a Full Episode Works

### Easy: Brute-Force SSH
1. Attacker IP hammers SSH with failed logins
2. Agent should `block_ip` the attacker
3. If not blocked, attacker gains root -> data exfiltrated

### Medium: Lateral Movement
1. External brute-force -> pivot host compromised
2. Pivot host SSH-hops to internal host
3. Admin account created on internal host
4. Agent should `investigate` then `isolate_host` the pivot

### Hard: APT Multi-Stage
1. Phishing email delivered -> user clicks macro -> C2 beacon established
2. LSASS credential dump -> service account harvested
3. Kerberos lateral movement: workstation -> DC -> file server
4. DNS tunneling exfiltrates 2.3 GB of staged data
5. Agent must `block_ip` the C2 AND `isolate_host` a compromised machine

## How the Reward Works - Concrete Numbers

Using the lateral movement example, if agent isolates pivot host on step 5 of 20 max steps:

```
detection = 1.0          (attack stopped this step)
false_positive = 0.0     (no benign IPs/hosts blocked)
efficiency = 0.75        = (1 - 5/20)
investigation_bonus = 0.04 (for early investigation)

base = max(0.0, min(1.0, (1.0 - 0.0 + 0.75) / 3.0))
     = 0.5833

overall = min(1.0, 0.5833 + 0.04) = 0.6233
```

## Local Setup & Execution

```bash
# Create and activate venv
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix/MacOS: source .venv/bin/activate

# Install deps
pip install -r requirements.txt

# Verify installation
python check.py

# Run demos
python demo_task2.py
python metrics.py

# LLM agent (requires API keys)
export HF_TOKEN="your-token"
python inference.py --task all
```

## Docker Testing

```bash
docker build -t ai-soc-gym .
docker run --rm ai-soc-gym python check.py
docker run --rm ai-soc-gym python metrics.py
docker run --rm -e HF_TOKEN="your-token" ai-soc-gym python inference.py
```