# tasks.py – scenario definitions for the AI SOC Gym
"""
This module defines the attack scenarios (tasks) that drive the hidden
`AttackState` inside `environment.py`.  Each task implements a tiny
interface (`BaseTask`) used by the environment to generate logs and to
report whether the analyst has successfully mitigated the attack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Set

from models import Action, ActionType, LogEntry

if TYPE_CHECKING:
    from environment import AIGymEnv

# ----------------------------------------------------------------------
# BaseTask – minimal contract for every scenario
# ----------------------------------------------------------------------
class BaseTask:
    def initialize(self, env: AIGymEnv) -> None:
        """Set up hidden state before the first reset."""
        raise NotImplementedError

    def advance(self, env: AIGymEnv) -> List[LogEntry]:
        """Return task‑specific logs for the current step."""
        raise NotImplementedError

    # Most tasks will override this; default = never mitigated.
    def is_mitigated(self, env: AIGymEnv) -> bool:  # pragma: no cover
        return False

# ----------------------------------------------------------------------
# Task 1 – Brute‑Force SSH (easy)
# ----------------------------------------------------------------------
@dataclass
class BruteForceSSHTask(BaseTask):
    attacker_ip: str = "203.0.113.5"
    max_failures: int = 4
    investigation_done: bool = False

    def initialize(self, env: AIGymEnv) -> None:
        env._state.stage = "initial_access"
        env._state.attacker_ips = [self.attacker_ip]
        env._bf_failed = 0
        env._bf_success = False

    def advance(self, env: AIGymEnv) -> List[LogEntry]:
        logs: List[LogEntry] = []
        if not env._bf_success:
            if env._bf_failed < self.max_failures:
                logs.append(
                    LogEntry(
                        timestamp=env._rand_timestamp(),
                        source="ssh",
                        severity="WARN",
                        message=f"Failed password for invalid user admin from {self.attacker_ip}",
                        ip=self.attacker_ip,
                        user="admin",
                        hostname="gateway.corp",
                        event_type="login",
                    )
                )
                env._bf_failed += 1
                return logs
            # success path – attacker gains foothold on gateway
            logs.append(
                LogEntry(
                    timestamp=env._rand_timestamp(),
                    source="ssh",
                    severity="INFO",
                    message=f"Accepted password for root from {self.attacker_ip}",
                    ip=self.attacker_ip,
                    user="root",
                    hostname="gateway.corp",
                    event_type="login",
                )
            )
            env._bf_success = True
            env._state.compromised_hosts.append("gateway.corp")
        return logs

    def is_mitigated(self, env: AIGymEnv) -> bool:
        return self.attacker_ip in env._blocked_ips

# ----------------------------------------------------------------------
# Task 2 – Lateral Movement (medium difficulty)
# ----------------------------------------------------------------------
@dataclass
class LateralMovementTask(BaseTask):
    """Medium scenario – external brute‑force, pivot host, internal hop.

    Steps:
    1️⃣ Brute‑force external login.
    2️⃣ Compromise **pivot_host** and SSH to **internal_host**.
    3️⃣ Create admin on **internal_host**.
    The analyst must eventually isolate a compromised host.
    """

    attacker_ip: str = "203.0.113.5"
    max_failures: int = 3
    pivot_host: str = "host-12.corp"
    internal_host: str = "host-27.corp"

    # internal counters – kept on the environment for simplicity
    _failed: int = field(default=0, init=False)
    _pivot_done: bool = field(default=False, init=False)
    _internal_done: bool = field(default=False, init=False)
    hosts_investigated: Set[str] = field(default_factory=set, init=False)

    def initialize(self, env: AIGymEnv) -> None:
        env._state.stage = "initial_access"
        env._state.attacker_ips = [self.attacker_ip]
        env._state.pivot_host = self.pivot_host
        env._state.internal_host = self.internal_host
        env._lm_failed = 0
        env._lm_pivot_done = False
        env._lm_internal_done = False

    def advance(self, env: AIGymEnv) -> List[LogEntry]:
        logs: List[LogEntry] = []
        # ----- Phase 1 – external brute‑force -----
        if not env._lm_pivot_done:
            if env._lm_failed < self.max_failures:
                logs.append(
                    LogEntry(
                        timestamp=env._rand_timestamp(),
                        source="ssh",
                        severity="WARN",
                        message=f"Failed password for invalid user admin from {self.attacker_ip}",
                        ip=self.attacker_ip,
                        user="admin",
                        hostname="gateway.corp",
                        event_type="login",
                    )
                )
                env._lm_failed += 1
                return logs
            # success – pivot host compromised
            logs.append(
                LogEntry(
                    timestamp=env._rand_timestamp(),
                    source="ssh",
                    severity="INFO",
                    message=f"Accepted password for svc_user from {self.attacker_ip} on {self.pivot_host}",
                    ip=self.attacker_ip,
                    user="svc_user",
                    hostname=self.pivot_host,
                    event_type="login",
                )
            )
            env._state.compromised_hosts.append(self.pivot_host)
            env._lm_pivot_done = True
            return logs

        # ----- Phase 2 – internal SSH hop -----
        if env._lm_pivot_done and not env._lm_internal_done:
            logs.append(
                LogEntry(
                    timestamp=env._rand_timestamp(),
                    source="ssh",
                    severity="INFO",
                    message=f"{self.pivot_host} initiated SSH connection to {self.internal_host}",
                    ip=self.pivot_host,
                    user="svc_user",
                    hostname=self.internal_host,
                    event_type="network_conn",
                )
            )
            # after a couple of steps the internal host gets admin rights
            if self._failed >= 2:
                logs.append(
                    LogEntry(
                        timestamp=env._rand_timestamp(),
                        source="ssh",
                        severity="INFO",
                        message=f"New admin account created on {self.internal_host} by svc_user",
                        ip=self.internal_host,
                        user="admin",
                        hostname=self.internal_host,
                        event_type="login",
                    )
                )
                env._state.compromised_hosts.append(self.internal_host)
                env._lm_internal_done = True
            else:
                self._failed += 1
            return logs

        # Phase 3 – attack already succeeded – nothing extra to emit
        return logs

    def is_mitigated(self, env: AIGymEnv) -> bool:
        compromised = set(env._state.compromised_hosts)
        isolated = set(env._isolated_hosts)
        return len(compromised & isolated) > 0

# ----------------------------------------------------------------------
# Exported symbols for `import *`
# ----------------------------------------------------------------------
__all__ = [
    "BaseTask",
    "BruteForceSSHTask",
    "LateralMovementTask",
]
