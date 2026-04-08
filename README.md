---
title: AI SOC Gym
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# AI SOC Gym - Reinforcement-Learning Environment for Cybersecurity Incident Response

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![HF Spaces](https://img.shields.io/badge/%F0%9F%A4%97-HF%20Spaces-orange)
![OpenEnv](https://img.shields.io/badge/OpenEnv-0.1.0-purple)

## One-line Pitch
A production-quality OpenEnv environment where an LLM agent acts as a SOC analyst, navigating multi-stage cyberattacks through partial observability, causal reasoning, and structured action spaces.

## Why This Environment Matters
Most RL environments focus on games or toy problems, but real-world SOC analysis requires temporal reasoning, causal inference, and decision-making under partial observability. AI SOC Gym bridges this gap by providing a standardized, hackathon-friendly environment that isolates the core reasoning challenges of SOC work: interpreting noisy logs, connecting distributed events over time, and taking precise mitigating actions.

## Architecture

| Component | File | Responsibility |
|-----------|------|----------------|
| Core simulation loop | `environment.py` | Gym-like `reset`/`step`/`state`, hidden `AttackState`, log generation, reward computation |
| Attack scenarios | `tasks.py` | `BaseTask` interface + 3 concrete tasks (Easy, Medium, Hard) |
| Data models | `models.py` | Pydantic schemas for type-safe logs, observations, actions, rewards |
| LLM agent | `inference.py` | OpenAI client connecting any LLM to the environment |
| Evaluation | `metrics.py` | Batch evaluation utilities for agent benchmarking |
| Grading | `graders.py` | Public reward computation matching environment internals |
| Demo | `demo_task2.py` | End-to-end run of lateral movement with heuristic agent |
| Configuration | `openenv.yaml` | OpenEnv registration metadata |
| Containerization | `Dockerfile` | Builds portable container |

## OpenEnv Interface

The environment implements the full OpenEnv specification:

```python
env = AIGymEnv(seed=42)
env.load_task(BruteForceSSHTask())

obs = env.reset()          # -> Observation
obs, reward, done, info = env.step(action)  # -> (Observation, Reward, bool, StepInfo)
state = env.state()        # -> dict (serializable snapshot)
```

## Observation Space

Realistic log streams from multiple sources:

| Source | Example |
|--------|---------|
| SSH | `Failed password for invalid user admin from 203.0.113.5` |
| HTTP | `GET /api/v1/resource HTTP/1.1 200 OK` |
| DNS | `Query A example.com` |
| Email | `Inbound email to jdoe@corp.local from hr-benefits@secure-docs.com` |
| Endpoint | `EXCEL.EXE spawned powershell.exe on ws-PC042.corp` |
| Firewall | `Outbound HTTPS connection from ws-PC042.corp to 198.51.100.23:443` |
| Auth | `Kerberos TGS request for host/dc-01.corp from ws-PC042.corp` |

Each observation includes metadata (step number, alerts triggered) and a list of `LogEntry` objects with timestamp, source, severity, message, IP, user, hostname, and event type.

## Action Space

```json
{
  "type": "block_ip | isolate_host | allow | investigate",
  "target_type": "ip | host | user",
  "target": "<identifier>"
}
```

| Action | Effect | Use Case |
|--------|--------|----------|
| `block_ip` | Adds IP to firewall deny list | Stop external C2 / brute-force |
| `isolate_host` | Removes host from network | Contain lateral movement |
| `allow` | Explicitly permit traffic | Reduce false positives |
| `investigate` | Gather more intel (bonus reward) | Build context before acting |

## Tasks

| ID | Name | Difficulty | Description | Optimal Score |
|----|------|------------|-------------|---------------|
| `brute` | Brute-Force SSH | Easy | Single IP performs repeated login attempts against gateway | ~0.58 |
| `lateral` | Lateral Movement | Medium | External brute-force -> pivot host -> internal hop -> admin creation | ~0.62 |
| `apt` | APT Multi-Stage | Hard | Phishing -> credential theft -> lateral movement (2 hops) -> DNS exfiltration | ~0.39 |

### Task 1: Brute-Force SSH (Easy)
A single attacker IP (`203.0.113.5`) performs repeated SSH login attempts against the gateway. After several failures, the attack succeeds. The agent must block the IP before the attacker gains root access.

### Task 2: Lateral Movement (Medium)
An attacker brute-forces entry, compromises a pivot host, and hops to an internal server. The agent must correlate logs across hosts and isolate the compromised machine to stop the attack chain.

### Task 3: APT Multi-Stage (Hard)
A full Advanced Persistent Threat kill-chain:
1. **Phishing** - Spear-phishing email delivers malicious macro document
2. **Credential Theft** - LSASS memory dump extracts service account credentials
3. **Lateral Movement** - Two internal hops: workstation -> domain controller -> file server
4. **Exfiltration** - DNS-over-HTTPS tunneling exfiltrates 2.3 GB of staged data

Mitigation requires blocking the C2 IP AND isolating a compromised host.

## Reward Function

The reward function balances three SOC objectives:

```
base = max(0.0, min(1.0, (detection - false_positive + efficiency) / 3.0))
overall = min(1.0, base + investigation_bonus)
```

| Component | Range | Description |
|-----------|-------|-------------|
| Detection | 0.0-1.0 | 1.0 if attack mitigated this step |
| False Positive | 0.0-1.0 | Penalty for blocking benign traffic |
| Efficiency | 0.0-1.0 | `1 - (steps_used / max_steps)` |
| Investigation Bonus | 0.0-0.06 | Reward for investigating before acting |

## Quickstart

### Local Development
```bash
# Clone and setup
git clone <repo-url>
cd cyber_ENV
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# Run verification
python check.py

# Run demo (lateral movement)
python demo_task2.py

# Run batch evaluation (all 3 tasks)
python metrics.py
```

### LLM Agent
```bash
# Set required environment variables
export HF_TOKEN="your-huggingface-token"
export API_BASE_URL="https://api.openai.com/v1"   # default
export MODEL_NAME="gpt-4.1-mini"                  # default

# Run all tasks
python inference.py

# Run specific task
python inference.py --task brute
python inference.py --task lateral
python inference.py --task apt
```

### Docker
```bash
docker build -t ai-soc-gym .
docker run --rm -e HF_TOKEN="your-token" ai-soc-gym
docker run --rm ai-soc-gym python check.py
docker run --rm ai-soc-gym python metrics.py
```

## Baseline Performance

Results from `python metrics.py` with the block-any-IP baseline policy (30 episodes each):

| Task | Avg Steps | Success Rate | Mean Detection | Mean False Positive |
|------|-----------|-------------|----------------|---------------------|
| Brute-Force SSH | ~4.4 | 1.00 | 0.00 | 1.00 |
| Lateral Movement | ~6.2 | 1.00 | 0.00 | 1.00 |
| APT Multi-Stage | ~8.5 | 1.00 | 0.00 | 1.00 |

> **Note**: The baseline policy is intentionally naive (blocks any IP it sees). A well-designed LLM agent should achieve significantly higher detection rates and lower false positive rates.

## HuggingFace Spaces Deployment

This environment is ready for HF Spaces deployment:
1. Create new Space with Docker template
2. Upload all project files
3. Set `HF_TOKEN` as a Space secret
4. Tag the space with `openenv`
5. The provided Dockerfile handles the rest

## License
MIT License