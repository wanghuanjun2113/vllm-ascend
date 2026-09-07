# SPDX-License-Identifier: Apache-2.0
"""NPU-side target-logit capture. Hooks execute outside model graph replay."""
import os
import time

import torch

from vllm.v1.sample.flight_recorder import enabled, transport

_active = None


def current():
    return _active


def processed(logits, metadata=None):
    if _active is not None:
        _active.processed(logits, metadata)


class Capture:
    def __init__(self, runner, logits, spec):
        self.runner = runner
        self.spec = spec
        self.req_ids = list(runner.input_batch.req_ids)
        self.k = int(os.getenv("VLLM_FLIGHT_RECORDER_TOPK", "128"))
        if not 1 <= self.k <= logits.shape[-1]:
            raise ValueError("capture Top-K outside vocabulary")
        # Preserve genuine pre-grammar values before downstream in-place edits.
        self.raw = logits.float().clone()
        self.final = torch.full_like(self.raw, float('nan'))
        self.filled = torch.zeros(self.raw.shape[0], dtype=torch.bool, device=logits.device)
        self.seen = set()
        self.rows = None
        self.greedy_rows = None
        self.probe_ids = [int(x) for x in os.environ.get(
            "VLLM_FLIGHT_RECORDER_PROBES", "248044,248046").split(",")]
        if any(x < 0 or x >= logits.shape[-1] for x in self.probe_ids):
            raise ValueError("probe token outside vocabulary")
        self.timestamp = time.time()

    def processed(self, logits, metadata=None):
        if isinstance(logits, tuple):
            logits, indices = logits
            if indices is not None:
                raise NotImplementedError("recorder needs global vocabulary, reduce_sample unsupported")
        rows = self.rows
        if rows is None:
            rows = torch.arange(logits.shape[0], device=logits.device)
        self.final[rows] = logits.float()
        self.filled[rows] = True
        self.seen.add("processed")
        # Save greedy rows before random-only temperature/top-p processing.
        if metadata is not None and not metadata.all_random:
            self.greedy_logits = logits.float().clone()
            self.greedy_rows = rows.clone()
            self.greedy_mask = (torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
                                if metadata.all_greedy else metadata.temperature < 1e-5)

    def finish(self, output):
        global _active
        _active = None
        if "processed" not in self.seen:
            raise RuntimeError("no processed-logit hook reached")
        if self.greedy_rows is not None:
            rows = self.greedy_rows[self.greedy_mask]
            self.final[rows] = self.greedy_logits[self.greedy_mask]
        selected = output.sampled_token_ids
        batch, width = selected.shape
        if self.spec is None:
            row_ids = torch.arange(batch, device=selected.device).unsqueeze(1)
            drafts = []
        else:
            drafts = list(self.spec.num_draft_tokens)
            starts = []
            start = 0
            for count in drafts:
                starts.append(start)
                start += count + 1
            row_ids = torch.tensor(starts, device=selected.device).unsqueeze(1)
            slots = torch.arange(width, device=selected.device).unsqueeze(0)
            limits = torch.tensor(drafts, device=selected.device).unsqueeze(1)
            row_ids = row_ids + torch.minimum(slots, limits)
        row_ids = row_ids.reshape(-1).long()
        chosen = selected.reshape(-1).long()
        safe_chosen = chosen.clamp_min(0)
        probe = torch.tensor(self.probe_ids, device=selected.device)
        gathered_ids = torch.cat([safe_chosen[:, None], probe.expand(chosen.numel(), -1)], dim=1)
        arrays = {"selected": chosen.to(torch.int32), "final_seen": self.filled[row_ids]}
        for stage, tensor in (("raw", self.raw), ("final", self.final)):
            # Top-K/reductions once per source row; only small summaries copied to host.
            vals, ids = torch.topk(tensor, self.k, dim=-1)
            arrays[stage + "_ids"] = ids[row_ids].to(torch.int32)
            arrays[stage + "_values"] = vals[row_ids]
            arrays[stage + "_lse"] = torch.logsumexp(tensor, dim=-1)[row_ids]
            arrays[stage + "_counts"] = torch.stack([
                torch.isfinite(tensor).sum(-1), torch.isnan(tensor).sum(-1),
                torch.isposinf(tensor).sum(-1), torch.isneginf(tensor).sum(-1)
            ], dim=-1)[row_ids].to(torch.int32)
            actual = tensor[row_ids].gather(1, gathered_ids)
            arrays[stage + "_probe_values"] = actual
            # Exact global ranks; ties share rank. Additional O(V) work, measured explicitly.
            ranks = []
            for col in range(gathered_ids.shape[1]):
                ranks.append((tensor[row_ids] > actual[:, col:col+1]).sum(-1) + 1)
            arrays[stage + "_probe_ranks"] = torch.stack(ranks, dim=-1).to(torch.int32)
        if self.spec is not None:
            arrays["draft_ids"] = self.spec.draft_token_ids.to(torch.int32)
        cpu = {}
        for key, value in arrays.items():
            buf = torch.empty(value.shape, dtype=value.dtype, device="cpu", pin_memory=True)
            buf.copy_(value, non_blocking=True)
            cpu[key] = buf
        event = torch.npu.Event()
        event.record()
        metadata = self.runner.input_batch.sampling_metadata
        from vllm_ascend.ascend_config import get_ascend_config
        rejection = get_ascend_config().rejection_sampler_config
        transport().submit({
            "kind": "batch", "req_ids": self.req_ids, "width": width,
            "discard_rows": self.runner.discard_request_indices.np[
                :self.runner.num_discarded_requests].tolist(),
            "num_draft_tokens": drafts, "top_k": self.k,
            "probe_ids": self.probe_ids, "timestamp": self.timestamp,
            "all_greedy": metadata.all_greedy, "all_random": metadata.all_random,
            "block_verify": rejection.enable_block_verify,
            "entropy_verify": rejection.enable_entropy_verify,
            "posterior_threshold": rejection.posterior_threshold,
            "posterior_alpha": rejection.posterior_alpha,
            "stage_semantics": ["model_pre_grammar", "target_post_processing"],
        }, cpu, event)


def begin(runner, logits, spec):
    global _active
    if not enabled():
        return None
    from vllm.distributed.parallel_state import get_tp_group
    from vllm_ascend.ascend_config import get_ascend_config
    if get_ascend_config().enable_reduce_sample:
        raise NotImplementedError("flight recorder requires full-vocabulary sampling")
    if get_tp_group().rank_in_group != 0:
        return None
    transport().check()
    from vllm.v1.sample import flight_recorder
    flight_recorder.capture_callback = processed
    _active = Capture(runner, logits, spec)
    return _active
