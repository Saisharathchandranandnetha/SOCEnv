# models.py – Pydantic schemas for AI SOC Gym
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Supported actions for the SOC analyst."""
    BLOCK_IP = "block_ip"
    ISOLATE_HOST = "isolate_host"
    ALLOW = "allow"
    INVESTIGATE = "investigate"


class LogEntry(BaseModel):
    """A single log entry as seen by the SOC analyst."""
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    source: str = Field(..., description="Log source (ssh, http, dns, etc.)")
    severity: str = Field(..., description="Log severity (INFO, WARN, ERROR, CRITICAL)")
    message: str = Field(..., description="Log message text")
    ip: Optional[str] = Field(None, description="IP address if relevant")
    user: Optional[str] = Field(None, description="Username if relevant")
    hostname: Optional[str] = Field(None, description="Hostname if relevant")
    event_type: Optional[str] = Field(None, description="Event type (login, file_access, etc.)")


class ObservationMetadata(BaseModel):
    """Metadata accompanying an observation."""
    step: int = Field(..., description="Current step number")
    alerts_triggered: int = Field(..., description="Number of WARN/ERROR/CRITICAL logs")


class Observation(BaseModel):
    """Observation received by the agent at each step."""
    logs: List[LogEntry] = Field(..., description="List of log entries")
    metadata: ObservationMetadata = Field(..., description="Step metadata")


class Action(BaseModel):
    """Action to be taken by the SOC analyst."""
    type: ActionType = Field(..., description="Type of action to take")
    target_type: str = Field(..., description="Type of target (ip, host, or user)")
    target: str = Field(..., description="Target identifier (IP address, hostname, or username)")


class RewardDetails(BaseModel):
    """Components that make up the reward."""
    detection: float = Field(..., ge=0.05, le=0.95, description=">0.0 if attack mitigated, else tiny value")
    false_positive_penalty: float = Field(..., ge=0.05, le=0.95, description="Penalty for blocking benign traffic")
    efficiency: float = Field(..., ge=0.05, le=0.95, description="Higher when fewer steps used")


class Reward(BaseModel):
    """Reward signal for the agent."""
    score: float = Field(..., ge=0.05, le=0.95, description="Final reward score (0-1, strictly)")
    details: RewardDetails = Field(..., description="Breakdown of reward components")


class StepInfo(BaseModel):
    """Debugging information returned after each step."""
    reason: str = Field(..., description="Human-readable explanation of what happened")
    confidence: float = Field(..., ge=0.05, le=0.95, description="Confidence in the explanation")
    action_effect: str = Field(..., description="Description of what the action actually did")