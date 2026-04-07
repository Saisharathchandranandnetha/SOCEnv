# AI SOC Gym – Reinforcement‑Learning Environment for Cybersecurity Incident Response

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![HF Spaces](https://img.shields.io/badge/%F0%9F%A4%97-HF%20Spaces-orange)
![OpenEnv](https://img.shields.io/badge/OpenEnv-0.1.0-purple)

## 📚 One-line Pitch
A production-quality OpenEnv environment where an LLM agent acts as a SOC analyst, navigating multi-stage cyberattacks through partial observability, causal reasoning, and structured action spaces.

## Why This Environment Matters
Most reinforcement learning environments focus on games or toy problems, but real-world SOC analysis requires sophisticated temporal reasoning, causal inference, and decision-making under partial observability. Current cybersecurity RL environments like CybORG are often overly complex for rapid experimentation. AI SOC Gym bridges this gap by providing a standardized, hackathon-friendly environment that isolates the core reasoning challenges of SOC work: interpreting noisy logs, connecting distributed events over time, and taking precise mitigating actions—all while optimizing for detection speed and minimizing disruption.

## 🏗️ Architecture & Core Concepts

| Component | File | Responsibility |
|-----------|------|----------------|
| Core simulation loop | `environment.py` | Gym-like `reset`/`step`, hidden `AttackState`, log generation, reward computation |
| Attack scenarios | `tasks.py` | `BaseTask` interface + `BruteForceSSHTask` (easy), `LateralMovementTask` (medium) |
| Data models | `models.py` | Pydantic schemas for type-safe logs, observations, actions, rewards |
| LLM agent | `inference.py` | Connects any OpenAI/Anthropic-compatible LLM to the environment |
| Evaluation | `metrics.py` | Batch evaluation utilities for agent benchmarking |
| Grading | `graders.py` | Public reward computation matching environment internals |
| Demo | `demo_task2.py` | End-to-end run of lateral movement with optimal heuristic agent |
| Configuration | `openenv.yaml` | OpenEnv registration metadata |
| Containerization | `Dockerfile` | Builds portable demo container |

## Environment Spec
- **Observation space**: Realistic log streams (SSH, HTTP, DNS) with configurable noise levels
- **Action space**: `block_ip`, `isolate_host`, `allow`, `investigate` (structured, typed actions)
- **Reward function**: 
  - Detection: +1.0 if attack mitigated this step
  - False Positive: -1.0 for blocking/isolating benign entities
  - Efficiency: (1 - steps_used/max_steps) scaled to [0,1]
  - Investigation Bonus: +0.04 for investigating before acting (lateral movement task)
  - Final: `min(1.0, (detection - false_positive + efficiency) / 3.0 + investigation_bonus)`
- **Episodes**: Maximum 20 steps per episode
- **Partial Observability**: Agent never sees hidden `AttackState`; only observes generated logs

## Tasks Table

| ID | Name | Difficulty | Description |
|----|------|------------|-------------|
| brute | Brute-Force SSH | Easy | Single IP performs repeated login attempts against public gateway |
| lateral | Lateral Movement | Medium | External brute-force → pivot host compromise → internal hop |
| apt | APT Multi-Stage | Hard | *Scaffolded*: Phishing → credential theft → lateral movement → exfiltration (future work) |

## 🚀 Quickstart

### Local Development
```bash
# 1. Clone (if not already done)
git clone <repo-url>
cd cyber_ENV

# 2. Create virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the demo (lateral movement task)
python demo_task2.py
```

### Docker Deployment
```bash
# Build container
docker build -t ai-soc-gym .

# Run demo
docker run --rm ai-soc-gym python demo_task2.py
```

## 🤖 LLM Agent Integration
The `inference.py` script connects any OpenAI/Anthropic-compatible LLM to the environment:

```bash
# Set required environment variables
export LLM_ENDPOINT="https://api.anthropic.com/v1/messages"  # or your endpoint
export LLM_API_KEY="your-api-key-here"

# Run with lateral movement task (default)
python inference.py --task lateral

# Or brute force task
python inference.py --task brute
```

The LLM must output JSON matching the `Action` schema:
```json
{
  "type": "block_ip|isolate_host|allow|investigate",
  "target_type": "ip|host|user",
  "target": "<identifier>"
}
```

## 🎯 Reward Design
The reward function balances three critical SOC objectives:

1. **Detection (Primary Objective)**: +1.0 for stopping the attack
   - Encourages agents to actually mitigate threats, not just observe

2. **False Positive Penalty (Cost Optimization)**: -1.0 for disruptive actions on benign entities
   - Models real-world SOC costs: blocked services, investigation overhead, user impact

3. **Efficiency (Speed Bonus)**: Higher reward for resolving in fewer steps
   - Reflects that faster containment = less damage

The division by 3.0 normalizes the raw sum to [0,1] range, making scores comparable across episodes. The investigation bonus (0.04) rewards proactive information gathering in the lateral movement task.

## 🤗 HuggingFace Spaces Deployment
This environment is ready for HF Spaces deployment:
1. Create new Space using Gradio template
2. Add `openenv.yaml` to root
3. Include all Python files and requirements
4. Set `LLM_ENDPOINT` and `LLM_API_KEY` as Space secrets
5. The provided `Dockerfile` ensures containerized consistency

## 🗺️ Roadmap
- [ ] **APT Task**: Implement full multi-stage attack with credential theft
- [ ] **IDS Alerts**: Add Snort/Suricata-style alert logs
- [ ] **Windows Event Logs**: Expand log sources beyond *nix systems
- [ ] **Multi-Agent**: Red vs Blue team training scenarios
- [ ] **Attention Visualizations**: Show which logs influenced decisions

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.