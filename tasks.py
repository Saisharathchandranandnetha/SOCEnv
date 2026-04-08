# tasks.py – scenario definitions for the AI SOC Gym
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
    _failed: int = field(default=0, init=False)
    _success: bool = field(default=False, init=False)

    def initialize(self, env: AIGymEnv) -> None:
        env._state.stage = "initial_access"
        env._state.attacker_ips = [self.attacker_ip]
        self._failed = 0
        self._success = False
        self.investigation_done = False

    def advance(self, env: AIGymEnv) -> List[LogEntry]:
        logs: List[LogEntry] = []
        if not self._success:
            if self._failed < self.max_failures:
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
                self._failed += 1
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
            self._success = True
            env._state.compromised_hosts.append("gateway.corp")
            env._state.data_exfiltrated = True
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

    # internal counters – kept on the task instance
    _external_failed: int = field(default=0, init=False)
    _internal_failed: int = field(default=0, init=False)
    _pivot_done: bool = field(default=False, init=False)
    _internal_done: bool = field(default=False, init=False)
    hosts_investigated: Set[str] = field(default_factory=set, init=False)

    def initialize(self, env: AIGymEnv) -> None:
        env._state.stage = "initial_access"
        env._state.attacker_ips = [self.attacker_ip]
        env._state.pivot_host = self.pivot_host
        env._state.internal_host = self.internal_host
        self._external_failed = 0
        self._internal_failed = 0
        self._pivot_done = False
        self._internal_done = False
        self.hosts_investigated = set()

    def advance(self, env: AIGymEnv) -> List[LogEntry]:
        logs: List[LogEntry] = []
        # ----- Phase 1 – external brute‑force -----
        if not self._pivot_done:
            if self._external_failed < self.max_failures:
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
                self._external_failed += 1
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
            self._pivot_done = True
            return logs

        # ----- Phase 2 – internal SSH hop -----
        if self._pivot_done and not self._internal_done:
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
            if self._internal_failed >= 2:
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
                self._internal_done = True
                env._state.data_exfiltrated = True
            else:
                self._internal_failed += 1
            return logs

        # Phase 3 – attack already succeeded – nothing extra to emit
        return logs

    def is_mitigated(self, env: AIGymEnv) -> bool:
        compromised = set(env._state.compromised_hosts)
        isolated = set(env._isolated_hosts)
        return len(compromised & isolated) > 0


# ----------------------------------------------------------------------
# Task 3 – APT Multi‑Stage (hard difficulty)
# ----------------------------------------------------------------------
@dataclass
class APTMultiStageTask(BaseTask):
    """Hard scenario — a full Advanced Persistent Threat kill-chain.

    Phases:
    1️⃣  **Phishing** — Spear‑phishing email delivers a malicious document
         to a user workstation.  The logs show an inbound email with a
         suspicious attachment, followed by an outbound C2 beacon.
    2️⃣  **Credential Theft** — The compromised workstation runs a
         credential‑dumping tool (Mimikatz‑style).  Logs show LSASS
         access and new logon events with harvested service accounts.
    3️⃣  **Lateral Movement** — Stolen creds are used to hop through
         two internal hosts.  Logs show WinRM / SSH connections between
         internal machines.
    4️⃣  **Exfiltration** — Attacker stages data on a staging host and
         tunnels it out via DNS over HTTPS.  Logs show large DNS‑TXT
         responses and unusual outbound traffic volume.

    Mitigation requires the analyst to:
      • Isolate the phished workstation  **AND**
      • Block the C2 IP
    Partial credit is given for blocking C2 alone or isolating any
    compromised host.
    """

    # ---------- configurable knobs ----------
    c2_ip: str = "198.51.100.23"
    phish_target_user: str = "jdoe"
    phished_host: str = "ws-PC042.corp"
    staging_host: str = "file-srv-01.corp"
    dc_host: str = "dc-01.corp"
    max_phish_steps: int = 2  # steps before the user clicks the link

    # ---------- internal counters ----------
    _phase: int = field(default=0, init=False)          # 0–4
    _phase_counter: int = field(default=0, init=False)   # steps within phase
    _user_clicked: bool = field(default=False, init=False)
    _creds_dumped: bool = field(default=False, init=False)
    _lateral_hops: int = field(default=0, init=False)
    _exfil_started: bool = field(default=False, init=False)
    hosts_investigated: Set[str] = field(default_factory=set, init=False)

    def initialize(self, env: AIGymEnv) -> None:
        env._state.stage = "initial_access"
        env._state.attacker_ips = [self.c2_ip]
        env._state.pivot_host = self.phished_host
        env._state.internal_host = self.staging_host
        self._phase = 0
        self._phase_counter = 0
        self._user_clicked = False
        self._creds_dumped = False
        self._lateral_hops = 0
        self._exfil_started = False
        self.hosts_investigated = set()

    # ------------------------------------------------------------------
    def advance(self, env: AIGymEnv) -> List[LogEntry]:
        logs: List[LogEntry] = []
        self._phase_counter += 1

        # ---- Phase 0 : Phishing delivery ----
        if self._phase == 0:
            logs.append(
                LogEntry(
                    timestamp=env._rand_timestamp(),
                    source="email",
                    severity="INFO",
                    message=(
                        f"Inbound email to {self.phish_target_user}@corp.local "
                        f"from hr-benefits@secure‑docs.com "
                        f"subject='Q1 Compensation Update' attachment=Q1_Update.xlsm"
                    ),
                    ip=self.c2_ip,
                    user=self.phish_target_user,
                    hostname=self.phished_host,
                    event_type="email",
                )
            )
            if self._phase_counter >= self.max_phish_steps:
                # User clicks the doc — macro fires
                logs.append(
                    LogEntry(
                        timestamp=env._rand_timestamp(),
                        source="endpoint",
                        severity="WARN",
                        message=(
                            f"EXCEL.EXE spawned powershell.exe on {self.phished_host} "
                            f"(user={self.phish_target_user})"
                        ),
                        ip=None,
                        user=self.phish_target_user,
                        hostname=self.phished_host,
                        event_type="process",
                    )
                )
                logs.append(
                    LogEntry(
                        timestamp=env._rand_timestamp(),
                        source="firewall",
                        severity="WARN",
                        message=(
                            f"Outbound HTTPS connection from {self.phished_host} "
                            f"to {self.c2_ip}:443 (TLS SNI: cdn‑assets.com)"
                        ),
                        ip=self.c2_ip,
                        user=self.phish_target_user,
                        hostname=self.phished_host,
                        event_type="network_conn",
                    )
                )
                self._user_clicked = True
                env._state.compromised_hosts.append(self.phished_host)
                self._phase = 1
                self._phase_counter = 0
            return logs

        # ---- Phase 1 : Credential theft ----
        if self._phase == 1:
            # First step in this phase — LSASS access
            if not self._creds_dumped:
                logs.append(
                    LogEntry(
                        timestamp=env._rand_timestamp(),
                        source="endpoint",
                        severity="CRITICAL",
                        message=(
                            f"Suspicious LSASS memory access on {self.phished_host} "
                            f"by process rundll32.exe (PID 4728)"
                        ),
                        ip=None,
                        user="SYSTEM",
                        hostname=self.phished_host,
                        event_type="process",
                    )
                )
                logs.append(
                    LogEntry(
                        timestamp=env._rand_timestamp(),
                        source="auth",
                        severity="WARN",
                        message=(
                            f"New logon: svc_backup@corp.local on {self.phished_host} "
                            f"(logon type 9 — NewCredentials)"
                        ),
                        ip=None,
                        user="svc_backup",
                        hostname=self.phished_host,
                        event_type="login",
                    )
                )
                self._creds_dumped = True
                env._state.stage = "lateral_movement"
                self._phase = 2
                self._phase_counter = 0
            return logs

        # ---- Phase 2 : Lateral movement (2 hops) ----
        if self._phase == 2:
            if self._lateral_hops == 0:
                # Hop 1: phished_host → dc_host
                logs.append(
                    LogEntry(
                        timestamp=env._rand_timestamp(),
                        source="auth",
                        severity="WARN",
                        message=(
                            f"Kerberos TGS request for host/{self.dc_host} "
                            f"from {self.phished_host} (user=svc_backup)"
                        ),
                        ip=self.phished_host,
                        user="svc_backup",
                        hostname=self.dc_host,
                        event_type="login",
                    )
                )
                logs.append(
                    LogEntry(
                        timestamp=env._rand_timestamp(),
                        source="ssh",
                        severity="INFO",
                        message=(
                            f"WinRM session opened on {self.dc_host} "
                            f"by svc_backup from {self.phished_host}"
                        ),
                        ip=self.phished_host,
                        user="svc_backup",
                        hostname=self.dc_host,
                        event_type="network_conn",
                    )
                )
                env._state.compromised_hosts.append(self.dc_host)
                self._lateral_hops = 1
                return logs

            if self._lateral_hops == 1:
                # Hop 2: dc_host → staging_host
                logs.append(
                    LogEntry(
                        timestamp=env._rand_timestamp(),
                        source="ssh",
                        severity="INFO",
                        message=(
                            f"{self.dc_host} initiated SMB connection to "
                            f"{self.staging_host} (share: \\\\{self.staging_host}\\data$)"
                        ),
                        ip=self.dc_host,
                        user="svc_backup",
                        hostname=self.staging_host,
                        event_type="network_conn",
                    )
                )
                logs.append(
                    LogEntry(
                        timestamp=env._rand_timestamp(),
                        source="endpoint",
                        severity="WARN",
                        message=(
                            f"Batch file copy: 847 files (2.3 GB) staged in "
                            f"C:\\Windows\\Temp\\~cache on {self.staging_host}"
                        ),
                        ip=None,
                        user="svc_backup",
                        hostname=self.staging_host,
                        event_type="file_access",
                    )
                )
                env._state.compromised_hosts.append(self.staging_host)
                self._lateral_hops = 2
                env._state.stage = "exfiltration"
                self._phase = 3
                self._phase_counter = 0
                return logs

            return logs

        # ---- Phase 3 : Exfiltration via DNS tunneling ----
        if self._phase == 3:
            if not self._exfil_started:
                logs.append(
                    LogEntry(
                        timestamp=env._rand_timestamp(),
                        source="dns",
                        severity="WARN",
                        message=(
                            f"Anomalous DNS-over-HTTPS traffic from {self.staging_host} "
                            f"to {self.c2_ip} — 340 TXT queries in 60 s "
                            f"(avg payload 220 bytes)"
                        ),
                        ip=self.c2_ip,
                        user=None,
                        hostname=self.staging_host,
                        event_type="network_conn",
                    )
                )
                logs.append(
                    LogEntry(
                        timestamp=env._rand_timestamp(),
                        source="firewall",
                        severity="CRITICAL",
                        message=(
                            f"High-volume outbound: {self.staging_host} → "
                            f"{self.c2_ip}:443  |  2.1 GB transferred in 180 s"
                        ),
                        ip=self.c2_ip,
                        user=None,
                        hostname=self.staging_host,
                        event_type="network_conn",
                    )
                )
                self._exfil_started = True

            if self._phase_counter >= 2:
                env._state.data_exfiltrated = True

            return logs

        return logs

    # ------------------------------------------------------------------
    def is_mitigated(self, env: AIGymEnv) -> bool:
        """Full mitigation requires BOTH:
        1. The C2 IP is blocked  OR  the phished host is isolated.
        2. At least one compromised host is isolated.
        """
        c2_blocked = self.c2_ip in env._blocked_ips
        phished_isolated = self.phished_host in env._isolated_hosts
        any_compromised_isolated = bool(
            set(env._state.compromised_hosts) & set(env._isolated_hosts)
        )
        return (c2_blocked or phished_isolated) and any_compromised_isolated


# ----------------------------------------------------------------------
# Exported symbols for `import *`
# ----------------------------------------------------------------------
__all__ = [
    "BaseTask",
    "BruteForceSSHTask",
    "LateralMovementTask",
    "APTMultiStageTask",
]
