"""HuggingFace-based rollout backend for LatentSeeker models."""

from typing import Any

import torch
from transformers import PreTrainedModel, ProcessorMixin

from ..arguments import RolloutArgs
from .base import RolloutBackend, RolloutOutput


class HFRolloutEngine:
    """HF model + processor wrapper.

    Owns its own model instance, fully separate from the training model.
    Weights are synced from training via backend.update().

    Usage:
        engine = HFRolloutEngine("path/to/model")
        engine = HFRolloutEngine("path", model_class=MyModel)
    """

    def __init__(
        self,
        model_name_or_path: str,
        model_class: type[PreTrainedModel] | None = None,
        processor_class: type[ProcessorMixin] | None = None,
        **kwargs,
    ):
        from transformers import AutoModelForCausalLM, AutoProcessor

        model_cls = model_class or AutoModelForCausalLM
        proc_cls = processor_class or AutoProcessor

        self.model = model_cls.from_pretrained(model_name_or_path, **kwargs)
        self.processor = proc_cls.from_pretrained(model_name_or_path)
        self.model.eval()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = False):
        self.model.load_state_dict(state_dict, strict=strict)


def _resolve_args(
    local: RolloutArgs | None,
    fallback: RolloutArgs | None,
    default: type[RolloutArgs],
) -> RolloutArgs:
    return local if local is not None else (fallback if fallback is not None else default())


class HFRolloutBackend(RolloutBackend):
    """Rollout backend using HF generate().

    Usage:
        engine = HFRolloutEngine("path/to/model")
        backend = HFRolloutBackend(engine, rollout_args=RolloutArgs(temperature=0.8))
        outputs = backend.generate(batch)
        backend.update(student.state_dict())  # sync weights
    """

    def __init__(
        self,
        engine: HFRolloutEngine,
        rollout_args: RolloutArgs | None = None,
    ):
        self.engine = engine
        self.rollout_args = rollout_args

    def generate(
        self,
        batch: list[dict[str, Any]],
        rollout_args: RolloutArgs | None = None,
        **kwargs,
    ) -> list[RolloutOutput]:
        args = _resolve_args(rollout_args, self.rollout_args, RolloutArgs)
        model = self.engine.model
        processor = self.engine.processor
        device = self.engine.device

        results: list[RolloutOutput] = []

        for item in batch:
            messages = item["messages"]
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

            inputs = processor(
                text=[prompt],
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

            input_len = inputs["input_ids"].shape[1]

            gen_output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                do_sample=args.do_sample,
                num_return_sequences=args.num_return_sequences,
                output_logits=True,
                return_dict_in_generate=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

            sequences = gen_output.sequences  # (num_return, total_len)
            logits = gen_output.logits         # tuple of (num_return, vocab_size)

            for seq_idx in range(sequences.shape[0]):
                seq = sequences[seq_idx]
                response_logits = torch.stack(
                    [l[seq_idx] for l in logits]
                )  # (response_len, vocab_size)

                full_text = processor.tokenizer.decode(seq, skip_special_tokens=True)

                results.append(RolloutOutput(
                    tokens=seq,
                    response_length=input_len,
                    rollout_logits=response_logits,
                    full_text=full_text,
                ))

        return results

    def update(self, state_dict: dict[str, torch.Tensor] | None = None) -> None:
        """Sync student weights from training into the rollout engine."""
        if state_dict is not None:
            self.engine.load_state_dict(state_dict)

    def shutdown(self) -> None:
        """Release GPU memory."""
        del self.engine.model
        torch.cuda.empty_cache()
