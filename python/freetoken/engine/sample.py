from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from freetoken.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from freetoken.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    from freetoken.kernel.backend import is_flashinfer_installed

    if is_flashinfer_installed():
        import flashinfer.sampling as sampling
    else:
        import freetoken.kernel.triton.sampling as sampling

    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)

    def apply_penalties(self, logits: torch.Tensor, batch: Batch | None) -> torch.Tensor:
        if batch is None:
            return logits
        for i, req in enumerate(batch.reqs):
            sp = getattr(req, "sampling_params", None)
            if sp is None:
                continue
            rep_pen = getattr(sp, "repetition_penalty", 1.0)
            freq_pen = getattr(sp, "frequency_penalty", 0.0)
            pres_pen = getattr(sp, "presence_penalty", 0.0)

            if rep_pen == 1.0 and freq_pen == 0.0 and pres_pen == 0.0:
                continue

            input_ids = getattr(req, "input_ids", None)
            if input_ids is None or input_ids.numel() == 0:
                continue

            prompt_len = getattr(req, "prompt_len", len(input_ids))
            out_tokens = input_ids[prompt_len:]

            # Multiplicative repetition penalty (Hugging Face / vLLM / llama.cpp standard)
            if rep_pen != 1.0:
                recent_prompt = input_ids[max(0, prompt_len - 128):prompt_len]
                target_tokens = (
                    torch.cat([recent_prompt, out_tokens])
                    if len(recent_prompt) > 0 and len(out_tokens) > 0
                    else (out_tokens if len(out_tokens) > 0 else recent_prompt)
                )
                if len(target_tokens) > 0:
                    tokens_dev = target_tokens.to(device=logits.device, dtype=torch.int64)
                    tokens_dev = tokens_dev[(tokens_dev >= 0) & (tokens_dev < self.vocab_size)]
                    if len(tokens_dev) > 0:
                        unique_rep = torch.unique(tokens_dev)
                        row_logits = logits[i, unique_rep]
                        logits[i, unique_rep] = torch.where(
                            row_logits > 0,
                            row_logits / rep_pen,
                            row_logits * rep_pen,
                        )

            # Subtractive presence & frequency penalties (OpenAI standard, applied to output tokens)
            if (freq_pen != 0.0 or pres_pen != 0.0) and len(out_tokens) > 0:
                out_dev = out_tokens.to(device=logits.device, dtype=torch.int64)
                out_dev = out_dev[(out_dev >= 0) & (out_dev < self.vocab_size)]
                if len(out_dev) > 0:
                    unique_out, counts = torch.unique(out_dev, return_counts=True)
                    penalty = counts.to(dtype=logits.dtype) * freq_pen + pres_pen
                    logits[i, unique_out] -= penalty

        return logits

    @nvtx_annotate("Sampler")
    def sample(
        self,
        logits: torch.Tensor,
        args: BatchSamplingArgs,
        batch: Batch | None = None,
    ) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            logits = self.apply_penalties(logits, batch)
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
