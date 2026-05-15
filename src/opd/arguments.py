"""OPD training arguments, organized by component.

Usage:
    args = OPDArgs(
        rollout=RolloutArgs(temperature=0.7),
        train=TrainArgs(kl_type="reverse"),
    )
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class RolloutArgs:
    """Sampling and generation parameters for student rollout."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = -1
    max_new_tokens: int = 1024
    num_return_sequences: int = 1
    do_sample: bool = True


@dataclass
class TrainArgs:
    """KL distillation and optimization parameters."""
    kl_type: Literal["forward", "reverse"] = "reverse"
    kl_coef: float = 1.0
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    gradient_accumulation_steps: int = 1


@dataclass
class BufferArgs:
    """Rollout buffer configuration."""
    max_size: int = 500
    train_batch_size: int = 4
    rollout_batch_size: int = 4


@dataclass
class TeacherArgs:
    """Teacher model loading."""
    model_name_or_path: str = ""
    dtype: str = "bfloat16"
    device: str = "cuda:0"


@dataclass
class OPDArgs:
    """Top-level OPD configuration."""
    rollout: RolloutArgs = field(default_factory=RolloutArgs)
    train: TrainArgs = field(default_factory=TrainArgs)
    buffer: BufferArgs = field(default_factory=BufferArgs)
    teacher: TeacherArgs = field(default_factory=TeacherArgs)
    output_dir: str = "outputs/opd"
    seed: int = 42
    total_steps: int = 1000
    log_interval: int = 10
    save_interval: int = 100
