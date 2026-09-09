"""Opt-in synchronous capture of actual Full Attention inputs (eager, one request)."""
import json
import os
import re
from pathlib import Path

import torch

_CONTROL = os.environ.get("QKV_CAPTURE_CONTROL")


def capture_inputs(module, positions, q, k, v):
    if not _CONTROL or not Path(_CONTROL).is_file():
        return None
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    from vllm.forward_context import get_forward_context

    control = json.loads(Path(_CONTROL).read_text())
    layer_index = int(re.search(r"layers\.(\d+)\.", module.attn.layer_name).group(1))
    if "layers" in control and layer_index not in control["layers"]:
        return None
    context = get_forward_context()
    metadata = context.attn_metadata
    if metadata is None:  # engine profile / warmup is never a dataset sample
        return None
    if isinstance(metadata, dict):
        metadata = metadata[module.attn.layer_name]
    n = int(metadata.num_actual_tokens)
    expected = int(control["seq_len"])
    if n != expected or int(metadata.num_prefills) != 1 or int(metadata.num_decodes) != 0:
        raise RuntimeError(f"Capture requires one unchunked prefill: actual={n}, expected={expected}")
    lengths = metadata.seq_lens_list
    if len(lengths) != 1 or int(lengths[0]) != n:
        raise RuntimeError("Capture requires matching complete Q/K lengths")
    pos = positions[..., :n].detach().cpu().clone()
    expected_positions = torch.arange(n)
    if not bool((pos == expected_positions).all()):
        raise RuntimeError("Capture positions are not a complete zero-based prefill")
    rank = get_tensor_model_parallel_rank()
    out = Path(control["output_dir"]) / control["sample_id"]
    out.mkdir(parents=True, exist_ok=True)
    path = out / (module.attn.layer_name + f".tp{rank}.pt")
    if path.exists():
        raise RuntimeError(f"Refusing duplicate capture: {path}")
    rows = torch.linspace(0, n-1, min(64, n)).round().long().unique()
    payload = {
        "schema_version": 1, "sample_id": control["sample_id"],
        "token_ids_sha256": control["token_ids_sha256"],
        "layer_name": module.attn.layer_name, "seq_len": n,
        "tp_rank": rank, "tp_size": get_tensor_model_parallel_world_size(),
        "q_heads_local": module.num_heads, "kv_heads_local": module.num_kv_heads,
        "head_dim": module.head_dim, "scaling": module.scaling,
        "capture_stage": "after_qk_rmsnorm_and_rope_before_attention",
        "positions": pos, "output_query_rows": rows,
        "q": q[:n].detach().cpu().clone().contiguous(),
        "k": k[:n].detach().cpu().clone().contiguous(),
        "v": v[:n].detach().cpu().clone().contiguous(),
    }
    return path, payload


def finish_capture(capture, attn_output):
    if capture is None:
        return
    path, payload = capture
    rows = payload["output_query_rows"].to(attn_output.device)
    payload["attention_output_rows"] = attn_output.index_select(0, rows).detach().cpu().clone().contiguous()
    temp = path.with_suffix(".pt.partial")
    torch.save(payload, temp)
    os.replace(temp, path)
