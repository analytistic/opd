"""Abstract base class for LLM inference backends.

A backend implements generate() to produce tokens + logits from prompts.
Backends handle their own initialization (device map, distributed config, etc.)
and expose only a uniform generate() interface to the worker.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class RolloutOutput:
    """Single response from a single LLM call.

    `tokens` = prompt + response concatenated, split by `response_length`.
    `rollout_logits` shape = ``[response_length, vocab_size]``.
    """
    tokens: list[int] | torch.Tensor          # [prompt || response]
    response_length: int                      # where response starts in tokens
    rollout_logits: torch.Tensor              # [response_length, vocab_size]
    rollout_hidden_states: tuple | None = None
    full_text: str = ""                       # Decoded text for reference


class LLMBackend(ABC):
    """Interface for LLM inference engines.

    Each backend (HF, vLLM, SGLang) implements this interface.
    Backends that need weight sync should override update().
    Backends that hold GPU resources should override shutdown().
    """

    @abstractmethod
    def generate(
        self, batch: list[dict[str, Any]], **kwargs
    ) -> list[RolloutOutput]:
        """Generate responses for a batch of prompts.

        Args:
            batch: Each sample has at least ``"messages"``.
            **kwargs: Backend-specific overrides.

        Returns:
            One RolloutOutput per sample × num_return_sequences.
        """
        ...

    def update(self, state_dict: dict[str, torch.Tensor] | None = None) -> None:
        """Sync weights from training into the inference engine.

        Only needed for multi-process setups (HF backend).
        """

    def shutdown(self) -> None:
        """Release GPU resources."""
