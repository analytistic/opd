"""Agentic worker: orchestrates LLM + Env in a loop, records full trajectory."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch

from ..env import Env
from ..llm_backend import LLMBackend


@dataclass
class StepOutput:
    """One step in the agentic loop: LLM call → Env feedback."""
    response: str                                # LLM raw text output
    rollout_logits: torch.Tensor | None = None   # for KL training
    action_type: str = ""                        # e.g. "execute_code"
    action_args: dict = field(default_factory=dict)
    observation: str = ""                        # env feedback


@dataclass
class Trajectory:
    """Complete agentic rollout trajectory for one task."""
    messages: list[dict]              # Full conversation history
    steps: list[StepOutput]           # Ordered steps
    reward: float = 0.0               # Final reward / score
    info: dict = field(default_factory=dict)  # Extra metadata


class Worker(ABC):
    """Orchestrates LLM + Env in a loop to produce a Trajectory.

    Subclasses define the agentic protocol (ReAct, function calling, etc.).
    """

    @abstractmethod
    def run(self, task: dict, llm: LLMBackend, env: Env) -> Trajectory:
        """Run one episode: LLM acts, Env responds, repeat."""

    def reset(self):
        """Reset worker state between episodes if needed."""
