"""Opt-in snapshots of actual unquantized GEMM operands and local outputs."""
import json
import os
import re
from pathlib import Path

import torch
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

_CONTROL = os.environ.get("LINEAR_CAPTURE_CONTROL")
_ORIGINAL = AscendUnquantizedLinearMethod.apply
_WEIGHTS = set()


def capture_before(layer, x, bias):
    if not _CONTROL or not Path(_CONTROL).is_file():
        return None
    control = json.loads(Path(_CONTROL).read_text())
    match = re.search(r"layers\.\d+\..+", layer.prefix)
    if match is None or match[0] not in control["modules"]:
        return None
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    from vllm.forward_context import get_forward_context

    metadata = get_forward_context().attn_metadata
    if metadata is None:
        return None
    values = metadata.values() if isinstance(metadata, dict) else [metadata]
    full = [m for m in values if getattr(m, "seq_lens_list", None) is not None]
    n = int(control["seq_len"])
    if not full or any(int(m.num_actual_tokens) != n or list(m.seq_lens_list) != [n] for m in full):
        raise RuntimeError("Linear capture requires one complete prefill")
    if x.ndim != 2 or x.shape[0] < n or bias is not None:
        raise RuntimeError("Unexpected Linear input shape or bias")
    rank = get_tensor_model_parallel_rank()
    if get_tensor_model_parallel_world_size() != 2:
        raise RuntimeError("This capture contract requires TP2")
    suffix = match[0]
    root = Path(control["output_dir"])
    weight_path = root / "weights" / f"{suffix}.tp{rank}.pt"
    key = str(weight_path)
    if key not in _WEIGHTS:
        if weight_path.exists():
            raise RuntimeError(f"Existing weight snapshot: {weight_path}")
        weight_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"prefix": layer.prefix, "suffix": suffix, "rank": rank, "tp_size": 2,
                    "weight": layer.weight.detach().cpu().clone().contiguous(),
                    "output_sizes": list(getattr(layer, "output_sizes", [])),
                    "input_size": int(layer.input_size), "output_size": int(layer.output_size),
                    "class": type(layer).__name__}, weight_path)
        _WEIGHTS.add(key)
    target = root / "activations" / control["sample_id"] / f"{suffix}.tp{rank}.pt"
    if target.exists():
        raise RuntimeError(f"Repeated GEMM in one sample: {target}")
    rows = torch.linspace(0, n-1, 32).round().long().unique()
    return target, {"sample_id": control["sample_id"], "seq_len": n, "suffix": suffix,
                    "rank": rank, "tp_size": 2, "token_ids_sha256": control["token_ids_sha256"],
                    "x": x[:n].detach().cpu().clone().contiguous(), "output_rows": rows}


def capture_after(snapshot, output):
    if snapshot is None:
        return
    target, payload = snapshot
    rows = payload["output_rows"].to(output.device)
    payload["y_local_rows"] = output.index_select(0, rows).detach().cpu().clone().contiguous()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".partial")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def capture_apply(self, layer, x, bias=None):
    snapshot = capture_before(layer, x, bias)
    output = _ORIGINAL(self, layer, x, bias)
    capture_after(snapshot, output)
    return output


if _CONTROL:
    AscendUnquantizedLinearMethod.apply = capture_apply
