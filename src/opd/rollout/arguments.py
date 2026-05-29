"""Rollout component configurations.

These configs are consumed by individual rollout components
(LLM backend, Env, Worker). The top-level OPDArgs in
``src/opd/arguments.py`` composes them together.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class LLMEngineConfig:
    """LLM inference engine configuration."""
    # Model loading
    model_name_or_path: str = ""
    dtype: str = "bfloat16"
    device: str = "cuda:0"

    # Sampling
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = -1
    max_new_tokens: int = 1024
    num_return_sequences: int = 1
    do_sample: bool = True

    # Backend selection
    backend: Literal["hf", "vllm"] = "hf"


@dataclass
class EnvConfig:
    """Execution environment configuration."""
    type: str = "sandbox"          # environment type key
    timeout: int = 10              # max seconds per step
    max_memory: int = 268_435_456  # max memory in bytes (256 MB)


@dataclass
class WorkerConfig:
    """Agentic worker configuration."""
    max_steps: int = 10            # max agentic loop steps per episode
    strategy: str = "react"        # "react", "function_calling", etc.
