# AI SOC Gym - Team Internal Documentation

## What Problem Does This Actually Solve?

SOC automation is incredibly difficult because real security analysis requires:
- **Temporal reasoning**: Connecting events that happen hours/days apart
- **Causal inference**: Determining if symptom A caused symptom B or they're unrelated
- **Partial observability**: Never seeing the attacker's true state, only indirect effects
- **Cost-sensitive decisions**: Blocking an IP might stop an attack but also break payroll for 500 people

Existing RL environments like CybORG simulate full network stacks but are overly complex for hackathon use - they require understanding network protocols, OS internals, and complex attack graphs. AI SOC Gym isolates the core cognitive challenge: interpreting noisy, partial log streams to make timely intervention decisions.

## How the Code is Structured

### environment.py - The Core Loop
- **Does**: Implements Gymnasium-style `reset()` and `step()`, manages hidden `AttackState`, generates realistic logs, computes rewards
- **Doesn't**: Handle LLM communication or task variability (delegated to other modules)
- **Watch for**: The `_state` variable contains the ground truth that agents must infer from logs

### tasks.py - Attack Scenarios
- **Does**: Defines `BaseTask` interface and two concrete scenarios (BruteForceSSH, LateralMovement)
- **Doesn't**: Include the hard APT task yet (stubbed as NotImplementedError)
- **Watch for**: Each task's `advance()` method returns logs that reveal partial attack progress

### models.py - Type Safety Layer
- **Does**: Provides Pydantic schemas ensuring type safety between environment, agent, and graders
- **Doesn't**: Contain any logic - purely data contracts
- **Watch for**: All inter-module communication uses these models; mismatch causes validation errors

### inference.py - LLM Bridge
- **Does**: Converts log observations to text prompts, calls LLM APIs, parses JSON actions
- **Doesn't**: Implement agent logic - just connects external LLMs to the environment
- **Watch for**: Requires properly formatted LLM endpoint (Anthropic Messages API or OpenAI-compatible)

### graders.py & metrics.py - Evaluation Tools
- **Does**: Provide reusable reward computation and batch evaluation capabilities
- **Doesn't**: Modify environment behavior - pure utility functions
- **Watch for**: `metrics.py` uses a monkey-patch to access `_last_observation` (explained below)

## The Hidden State Mechanic

The `AttackState` in `environment.py` represents the complete attacker progression:
- `stage`: "initial_access" → "lateral_movement" → "exfiltration"
- `compromised_hosts`: List of systems under attacker control
- `attacker_ips`: External IPs used in the attack
- `data_exfiltrated`: Boolean indicating breach completion

**Crucially**: The agent NEVER sees this directly. Instead, they observe:
- Login attempts revealing `attacker_ips`
- New logins from compromised IPs showing lateral movement
- Service creation/logins on newly compromised hosts
- Unusual outbound connections suggesting exfiltration

This forces agents to develop real SOC skills: correlating distributed events over time to infer hidden state.

## How a Full Episode Works (Lateral Movement Example)

**Reset**: 
- Environment creates new `AttackState` (stage=initial_access)
- `BruteForceSSHTask` or `LateralMovementTask.initialize()` configures attacker parameters
- Initial logs generated: mostly noise + early brute force attempts

**Step 1-3** (Brute Force Phase):
- Logs show increasing failed SSH attempts from 203.0.113.5
- Agent observes: `[ssh] WARN: Failed password for invalid user admin from 203.0.113.5`
- Optimal action: `investigate` the IP to get bonus
- Environment: validates action, updates investigation flag, generates new logs

**Step 4-6** (Pivot Phase):
- Logs show successful login: `[ssh] INFO: Accepted password for svc_user from 203.0.113.5 on host-12.corp`
- Agent observes: Internal host `host-12.corp` now communicating
- Optimal action: `block_ip` 203.0.113.5 (stops new command & control)
- Environment: adds IP to blocked set, attack can't progress further

**Step 7-9** (Containment Phase):
- Logs show lateral movement attempt: `[ssh] INFO: host-12.corp initiated SSH connection to host-27.corp`
- Agent observes: Attacker using pivot to reach internal host
- Optimal action: `isolate_host` host-12.corp (cuts attack path)
- Environment: adds host to isolated set, detects mitigation via `is_mitigated()`
- Episode ends with high reward: detection + efficiency bonuses

## How the Reward Works - Concrete Numbers

Using the lateral movement example above, if agent isolates pivot host on step 5 of 20 max steps:

```
detection = 1.0          (attack stopped this step)
false_positive = 0.0     (no benign IPs/hosts blocked)
efficiency = 0.75        = (1 - 5/20) = 1 - 0.25
investigation_bonus = 0.04 (for early investigation)

base = max(0.0, min(1.0, (1.0 - 0.0 + 0.75) / 3.0))
   = max(0.0, min(1.0, 1.75 / 3.0))
   = max(0.0, 0.5833)
   = 0.5833

overall = min(1.0, 0.5833 + 0.04) = 0.6233

Final reward: 0.623
```

Breakdown: 
- 0.58 from stopping attack efficiently 
- +0.04 for investigating first
- Total: 0.62/1.0 = solid performance

## How to Add a New Task

1. **Create subclass**: In `tasks.py`, create `class MyNewTask(BaseTask):`
2. **Implement initialize(self, env)**: 
   - Set up `env._state` with initial conditions
   - Store any task-specific counters on env (e.g., `env._my_counter = 0`)
3. **Implement advance(self, env) -> List[LogEntry]**:
   - Return log entries that reveal partial attack progress
   - Update env counters as attack progresses
   - Return empty list when no new logs this step
4. **Implement is_mitigated(self, env) -> bool** (optional):
   - Return True when agent has successfully stopped attack
   - Default returns False (never mitigated)
5. **Register**: Add to imports and `__all__` list if needed
6. **Test**: Load in `demo_task2.py` or `inference.py` with your new task

Example structure:
```python
@dataclass
class PhishingTask(BaseTask):
    # ... fields ...
    
    def initialize(self, env: AIGymEnv) -> None:
        env._state.stage = "initial_access"
        env._state.attacker_ips = ["198.51.100.10"]
        env._phish_clicked = False
    
    def advance(self, env: AIGymEnv) -> List[LogEntry]:
        logs = []
        if not env._phish_clicked:
            # Return phishing email logs
            if some_condition:
                logs.append(LogEntry(...))
                env._phish_clicked = True
        else:
            # Return malware installation logs
            pass
        return logs
    
    def is_mitigated(self, env: AIGymEnv) -> bool:
        # Blocked sender IP or isolated clicked host?
        return "198.51.100.10" in env._blocked_ips
```

## Known Limitations & Stubs

1. **APT Task**: `tasks.py` raises `NotImplementedError` - intentional scaffold for future work
2. **metrics.py Hack**: Uses `env._last_observation = observation` monkey-patch because:
   - Original design didn't expose last observation
   - Needed for batch evaluation to show what agent saw
   - Clean fix would be adding `get_last_observation()` to environment
3. **Missing models.py**: Was absent from original zip - created from scratch based on usage
4. **LLM Specificity**: `inference.py` assumes Anthropic Messages API format:
   - Works with Claude via API or compatible proxies
   - For OpenAI, would need different response parsing
   - For local models, would need endpoint adapter

## Local Setup & Execution

### Environment Preparation
```bash
# Create and activate venv
python -m venv .venv
# Windows: .venv\Scripts\activate.bat
# Unix/MacOS: source .venv/bin/activate

# Install deps
pip install -r requirements.txt

# Verify installation
python -c "from environment import AIGymEnv; print('Import successful')"
```

### Running Demos
```bash
# Heuristic agent demo (lateral movement)
python demo_task2.py

# LLM agent requires API keys:
export LLM_ENDPOINT="https://api.anthropic.com/v1/messages"
export LLM_API_KEY="your-key-here"
python inference.py --task lateral
```

### Batch Evaluation
```bash
# Run 100 episodes with random policy for baseline
python metrics.py
```

## Extending inference.py for Other LLMs

To use different LLM providers:

1. **OpenAI Compatible Endpoints**:
   - Set `LLM_ENDPOINT` to your OpenAI-compatible URL
   - Keep `LLM_API_KEY` as bearer token
   - The current parsing works for OpenAI Chat Completions format

2. **Local Models (llama.cpp, TGI, etc.)**:
   - Modify `call_llm()` function in inference.py
   - Adjust payload format and response parsing to match your endpoint
   - Example for HuggingFace TGI:
     ```python
     payload = {
         "inputs": prompt,
         "parameters": {"max_new_tokens": 256, "temperature": 0.0}
     }
     # Response is direct text, not JSON
     raw = resp.json()[0]["generated_text"]
     ```

3. **Google Gemini / Other APIs**:
   - Adapt payload structure and auth headers
   - Maintain the core contract: prompt in, Action JSON out

The key requirement remains: LLM must output valid JSON matching the Action schema.