# environment.py – full implementation (see analysis for code)
import random
from typing import Any, Dict, List, Optional

from models import (
    Action,
    ActionType,
    LogEntry,
    Observation,
    ObservationMetadata,
    Reward,
    RewardDetails,
    StepInfo,
)
from tasks import BaseTask, BruteForceSSHTask, LateralMovementTask, APTMultiStageTask

# ----------------------------------------------------------------------
# Hidden attacker model (never exported)
# ----------------------------------------------------------------------
class AttackState:
    """
    Hidden truth the agent never sees.
    The fields below are deliberately mutable only inside the env.
    """

    stage: str  # "initial_access" | "lateral_movement" | "exfiltration"
    compromised_hosts: List[str]
    attacker_ips: List[str]
    data_exfiltrated: bool

    # NEW fields for lateral‑movement
    pivot_host: Optional[str] = None
    internal_host: Optional[str] = None

    def __init__(self) -> None:
        self.stage = "initial_access"
        self.compromised_hosts = []
        self.attacker_ips = []
        self.data_exfiltrated = False
        self.pivot_host = None
        self.internal_host = None

    # ------------------------------------------------------------------
    def advance_stage(self) -> None:
        if self.stage == "initial_access":
            self.stage = "lateral_movement"
        elif self.stage == "lateral_movement":
            self.stage = "exfiltration"

    def is_finished(self) -> bool:
        return self.data_exfiltrated

# ----------------------------------------------------------------------
# Core environment (Gym‑like)
# ----------------------------------------------------------------------
class AIGymEnv:
    MAX_STEPS = 20
    NOISE_SOURCES = ["ssh", "http", "dns"]

    def __init__(self, seed: int | None = None) -> None:
        self._rand = random.Random(seed)
        self._step_counter: int = 0
        self._state: AttackState = AttackState()
        self._blocked_ips: set[str] = set()
        self._isolated_hosts: set[str] = set()
        self._last_action_effect: Optional[str] = None
        self._task: Optional[BaseTask] = None
        self._ip_pool = [
            f"10.0.{i}.{j}" for i in range(1, 5) for j in range(1, 255)
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> Observation:
        self._step_counter = 0
        self._state = AttackState()
        self._blocked_ips.clear()
        self._isolated_hosts.clear()
        if self._task:
            self._task.initialize(self)
        init_logs = self._generate_logs(phase="reset") + self._task_advance()
        return Observation(
            logs=init_logs,
            metadata=ObservationMetadata(step=self._step_counter, alerts_triggered=0),
        )

    def step(self, action: Action) -> tuple[Observation, Reward, bool, StepInfo]:
        self._step_counter += 1
        valid, msg = self._validate_action(action)
        if not valid:
            reward = self._compute_reward(detection=0.0, false_positive=1.0, efficiency=0.0)
            info = StepInfo(reason=f"Invalid action: {msg}", confidence=0.0, action_effect="none")
            obs = Observation(
                logs=self._generate_logs() + self._task_advance(),
                metadata=ObservationMetadata(step=self._step_counter, alerts_triggered=0),
            )
            return obs, reward, False, info

        self._apply_action(action)
        mitigated = self._task_mitigated()
        if not mitigated:
            self._state.advance_stage()
        logs = self._generate_logs() + self._task_advance()
        observation = Observation(
            logs=logs,
            metadata=ObservationMetadata(
                step=self._step_counter,
                alerts_triggered=self._count_alerts(logs),
            ),
        )
        detection = 1.0 if mitigated else 0.0
        false_positive = 1.0 if self._action_was_false_positive(action) else 0.0
        efficiency = max(0.0, 1.0 - (self._step_counter / self.MAX_STEPS))
        reward = self._compute_reward(detection, false_positive, efficiency)
        done = (
            self._step_counter >= self.MAX_STEPS or self._state.is_finished()
        )
        info = StepInfo(
            reason=self._explain_reason(),
            confidence=self._rand.uniform(0.7, 0.99),
            action_effect=self._last_action_effect,
        )
        return observation, reward, done, info

    def state(self) -> Dict[str, Any]:
        """Return a serializable snapshot of the current environment state.

        Required by the OpenEnv specification.
        """
        return {
            "step": self._step_counter,
            "stage": self._state.stage,
            "attacker_ips": list(self._state.attacker_ips),
            "compromised_hosts": list(self._state.compromised_hosts),
            "pivot_host": self._state.pivot_host,
            "internal_host": self._state.internal_host,
            "data_exfiltrated": self._state.data_exfiltrated,
            "blocked_ips": list(self._blocked_ips),
            "isolated_hosts": list(self._isolated_hosts),
        }

    # ------------------------------------------------------------------
    # Task handling helpers
    # ------------------------------------------------------------------
    def load_task(self, task: BaseTask) -> None:
        self._task = task
        task.initialize(self)

    def _task_advance(self) -> List[LogEntry]:
        if self._task:
            return self._task.advance(self)  # type: ignore[arg-type]
        return []

    def _task_mitigated(self) -> bool:
        if self._task and hasattr(self._task, "is_mitigated"):
            return self._task.is_mitigated(self)  # type: ignore[arg-type]
        return False

    # ------------------------------------------------------------------
    # Validation & application
    # ------------------------------------------------------------------
    def _validate_action(self, action: Action) -> tuple[bool, str]:
        if action.type == ActionType.BLOCK_IP:
            if action.target_type != "ip":
                return False, "BLOCK_IP requires target_type='ip'"
            if action.target not in self._ip_pool and action.target not in self._state.attacker_ips:
                return False, "IP unknown to the system"
            if action.target in self._blocked_ips:
                return False, "IP already blocked"
        elif action.type == ActionType.ISOLATE_HOST:
            if action.target_type != "host":
                return False, "ISOLATE_HOST requires target_type='host'"
        elif action.type == ActionType.ALLOW:
            if action.target_type not in {"ip", "host"}:
                return False, "ALLOW must target ip or host"
        return True, ""

    def _apply_action(self, action: Action) -> None:
        if action.type == ActionType.BLOCK_IP:
            self._blocked_ips.add(action.target)
            self._last_action_effect = f"IP {action.target} blocked"
        elif action.type == ActionType.ISOLATE_HOST:
            self._isolated_hosts.add(action.target)
            self._last_action_effect = f"Host {action.target} isolated"
        elif action.type == ActionType.ALLOW:
            self._last_action_effect = (
                f"Allowed {action.target_type} {action.target}"
            )
        elif action.type == ActionType.INVESTIGATE:
            if (
                isinstance(getattr(self, "_task", None), BruteForceSSHTask)
                and action.target_type == "ip"
                and action.target == self._state.attacker_ips[0]
            ):
                self._task.investigation_done = True
            if (
                isinstance(getattr(self, "_task", None), LateralMovementTask)
                and action.target_type == "host"
                and action.target in {self._state.pivot_host, self._state.internal_host}
            ):
                self._task.hosts_investigated.add(action.target)
            if (
                isinstance(getattr(self, "_task", None), APTMultiStageTask)
                and action.target_type in {"host", "ip"}
            ):
                self._task.hosts_investigated.add(action.target)
            self._last_action_effect = f"Investigated {action.target_type} {action.target}"
        else:
            self._last_action_effect = "no effect"

    # ------------------------------------------------------------------
    def _action_was_false_positive(self, action: Action) -> bool:
        if action.type == ActionType.BLOCK_IP:
            if action.target not in self._state.attacker_ips:
                return True
        if action.type == ActionType.ISOLATE_HOST:
            if (
                action.target not in self._state.compromised_hosts
                and action.target != self._state.pivot_host
                and action.target != self._state.internal_host
            ):
                return True
        return False

    # ------------------------------------------------------------------
    def _generate_logs(self, phase: str = "step") -> List[LogEntry]:
        logs: List[LogEntry] = []
        for _ in range(self._rand.randint(2, 5)):
            src = self._rand.choice(self.NOISE_SOURCES)
            logs.append(
                LogEntry(
                    timestamp=self._rand_timestamp(),
                    source=src,
                    severity=self._rand.choice(["INFO", "WARN"]),
                    message=self._random_noise_message(src),
                    ip=self._rand.choice(self._ip_pool) if src != "dns" else None,
                    user=self._rand.choice(["alice", "bob", "carol"]),
                    hostname=f"host-{self._rand.randint(1,20)}.corp",
                    event_type=self._rand.choice(
                        ["login", "file_access", "network_conn"]
                    ),
                )
            )
        return logs

    def _rand_timestamp(self) -> str:
        base = 1672502400
        ts = base + self._step_counter * 60 + self._rand.randint(0, 30)
        return f"2026-04-07T{(ts % 86400)//3600:02d}:{(ts % 3600)//60:02d}:{ts % 60:02d}Z"

    def _random_noise_message(self, src: str) -> str:
        if src == "ssh":
            return (
                f"Accepted publickey for {self._rand.choice(['alice', 'bob'])} "
                f"from {self._rand.choice(self._ip_pool)}"
            )
        if src == "http":
            return "GET /api/v1/resource HTTP/1.1 200 OK"
        if src == "dns":
            return "Query A example.com"
        return "Generic log entry"

    def _count_alerts(self, logs: List[LogEntry]) -> int:
        return sum(1 for l in logs if l.severity in {"WARN", "ERROR", "CRITICAL"})

    # ------------------------------------------------------------------
    def _compute_reward(
        self,
        detection: float,
        false_positive: float,
        efficiency: float,
    ) -> Reward:
        from graders import compute_reward
        bonus = 0.0
        if isinstance(getattr(self, "_task", None), LateralMovementTask):
            task: LateralMovementTask = self._task  # type: ignore[assignment]
            if task.hosts_investigated:
                bonus = 0.04
        if isinstance(getattr(self, "_task", None), APTMultiStageTask):
            apt_task: APTMultiStageTask = self._task  # type: ignore[assignment]
            if apt_task.hosts_investigated:
                bonus = 0.06  # Higher bonus for harder task investigation
        return compute_reward(detection, false_positive, efficiency, investigation_bonus=bonus)

    # ------------------------------------------------------------------
    def _explain_reason(self) -> str:
        if self._state.stage == "initial_access":
            return "Brute-force attempts detected"
        if self._state.stage == "lateral_movement":
            return "Suspicious internal traffic observed"
        if self._state.stage == "exfiltration":
            return "Large outbound data transfer observed"
        return "Noise / normal operation"

    # ------------------------------------------------------------------
    def dump_internal_state(self) -> str:
        return (
            f"Step: {self._step_counter}\n"
            f"Stage: {self._state.stage}\n"
            f"Attacker IPs: {self._state.attacker_ips}\n"
            f"Compromised hosts: {self._state.compromised_hosts}\n"
            f"Pivot host: {self._state.pivot_host}\n"
            f"Internal host: {self._state.internal_host}\n"
            f"Blocked IPs: {list(self._blocked_ips)}\n"
            f"Isolated hosts: {list(self._isolated_hosts)}\n"
            f"Data exfiltrated: {self._state.data_exfiltrated}"
        )
