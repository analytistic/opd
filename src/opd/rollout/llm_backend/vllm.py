"""vLLM-based rollout backend."""

from typing import Any

import torch

from ...arguments import RolloutArgs
from ..llm_backend.base import LLMBackend, RolloutOutput

RolloutBackend = LLMBackend  # backward compat alias


class VLLMEngine:
    """vLLM model wrapper, fully independent from the training model.

    Usage:
        engine = VLLMEngine("path/to/model", tensor_parallel_size=1)
        backend = VLLMRolloutBackend(engine)
    """

    def __init__(self, model_name_or_path: str, **kwargs):
        from vllm import LLM

        self.model = LLM(model=model_name_or_path, **kwargs)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = False):
        """Not supported in vLLM — requires engine restart or separate weight sync."""
        raise NotImplementedError(
            "vLLM does not support live weight update. "
            "Use a multi-process setup: save weights to disk and reload, "
            "or use a separate vLLM server with --load-format."
        )


class VLLMRolloutBackend(RolloutBackend):
    """Rollout backend using vLLM generate().

    Usage:
        engine = VLLMEngine("path/to/model")
        backend = VLLMRolloutBackend(engine, rollout_args=RolloutArgs(temperature=0.8))
        outputs = backend.generate(batch)
    """

    def __init__(
        self,
        engine: VLLMEngine,
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
        from vllm import SamplingParams

        from ..arguments import RolloutArgs as _RA

        args = rollout_args or self.rollout_args or _RA()

        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k if args.top_k > 0 else -1,
            max_tokens=args.max_new_tokens,
            n=args.num_return_sequences,
            output_logits=True,
        )

        # Format prompts with chat template if processor is available
        prompts = []
        for item in batch:
            messages = item["messages"]
            prompt = item.get("prompt", None)
            if prompt is None and hasattr(self, "_format_prompt"):
                prompt = self._format_prompt(messages)
            elif prompt is None:
                # Fallback: use last user message content
                prompt = messages[-1]["content"] if isinstance(messages[-1], dict) else str(messages[-1])
            prompts.append(prompt)

        outputs = self.engine.model.generate(prompts, sampling_params)

        results: list[RolloutOutput] = []
        for output in outputs:
            prompt_len = len(output.prompt_token_ids)

            for out in output.outputs:
                tokens = output.prompt_token_ids + out.token_ids
                response_logits = torch.stack(out.logits) if out.logits else torch.empty(0, 0)

                full_text = (output.prompt + out.text) if hasattr(out, "text") else ""

                results.append(RolloutOutput(
                    tokens=tokens,
                    response_length=prompt_len,
                    rollout_logits=response_logits,
                    full_text=full_text,
                ))

        return results

    def update(self, state_dict: dict[str, torch.Tensor] | None = None) -> None:
        """vLLM does not support live weight update — no-op.

        Use multi-process setup: save checkpoint and restart vLLM server.
        """
        if state_dict is not None:
            raise RuntimeError(
                "vLLM does not support live weight sync. "
                "Use VLLMEngine with --load-format or save/reload."
            )

    def shutdown(self) -> None:
        """Release GPU memory."""
        del self.engine.model
