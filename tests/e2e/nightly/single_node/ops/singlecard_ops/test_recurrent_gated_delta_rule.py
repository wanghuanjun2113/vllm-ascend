import gc
import random

import numpy as np
import pytest
import torch
import torch_npu

torch_npu.npu.set_compile_mode(jit_compile=False)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


@pytest.mark.parametrize(
    "lengths,accepted",
    [
        ([4, 4], [1, 1]),
        ([3, 4], [1, 1]),
        ([4, 2, 4], [1, 1, 1]),
        ([3, 0, 4], [1, 1, 1]),
        ([3, 4], [4, 1]),
    ],
)
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_recurrent_gated_delta_rule_fixed_state_rows(lengths, accepted, state_dtype):
    """Short queries must not shift subsequent requests' state reads or writes."""
    row_width = 4
    num_tokens = sum(lengths)
    num_key_heads, num_value_heads, head_dim = 4, 8, 128
    state_indices = torch.arange(1, len(lengths) * row_width + 1, dtype=torch.int32).view(-1, row_width)
    num_blocks = state_indices.numel() + 2
    # Distinct block values expose both foreign writes and untouched stale slots.
    state = (
        torch.arange(num_blocks, dtype=state_dtype)
        .view(-1, 1, 1, 1)
        .expand(-1, num_value_heads, head_dim, head_dim)
        .clone()
    )
    query = torch.zeros(num_tokens, num_key_heads, head_dim, dtype=torch.bfloat16)
    query[..., 0] = 1
    value = torch.zeros(num_tokens, num_value_heads, head_dim, dtype=torch.bfloat16)
    expected_output = torch.empty_like(value)
    expected_state = state.clone()
    offset = 0
    for row, (length, count) in enumerate(zip(lengths, accepted)):
        current = state[state_indices[row, count - 1]].clone()
        for token in range(length):
            # q=k=e0, v=g=0, beta=0.5: only state column zero halves per token.
            current[..., 0] *= 0.5
            expected_output[offset + token] = current[..., 0]
            expected_state[state_indices[row, token]] = current
        offset += length

    state_npu = state.npu()
    output = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
        query=query.npu(),
        key=query.npu(),
        value=value.npu(),
        state=state_npu,
        beta=torch.full((num_tokens, num_value_heads), 0.5, dtype=torch.bfloat16).npu(),
        g=torch.zeros(num_tokens, num_value_heads, dtype=torch.float32).npu(),
        scale=1.0,
        actual_seq_lengths=torch.tensor([0, *lengths], dtype=torch.int32).npu(),
        ssm_state_indices=state_indices.npu(),
        num_accepted_tokens=torch.tensor(accepted, dtype=torch.int32).npu(),
    )
    torch.testing.assert_close(output.cpu(), expected_output, rtol=3e-3, atol=1e-2)
    torch.testing.assert_close(state_npu.cpu(), expected_state, rtol=3e-3, atol=1e-2)


def golden_recurrent_gated_delta_rule(
    query,
    key,
    value,
    state,
    beta,
    scale,
    actual_seq_lengths,
    ssm_state_indices,
    g,
    num_accepted_tokens,
):
    """Pure torch/CPU golden implementation of recurrent gated delta rule.

    Args:
        query: [T, nk, dk]
        key: [T, nk, dk]
        value: [T, nv, dv]
        state: [S, nv, dv, dk]
        beta: [T, nv]
        scale: float
        actual_seq_lengths: [batch_size] per-sequence lengths
        ssm_state_indices: [T] per-token state block index
        g: [T, nv] or None
        num_accepted_tokens: [batch_size] or None

    Returns:
        (output [T, nv, dv], updated_state [S, nv, dv, dk])
    """
    q = query.to(torch.float32)
    k = key.to(torch.float32)
    v = value.to(torch.float32)
    initial_state = state.clone().to(torch.float32)
    T, n_heads_v, Dv = v.shape
    n_heads_k = q.shape[-2]
    g = torch.ones(T, n_heads_v).to(torch.float32) if g is None else g.to(torch.float32).exp()
    beta = torch.ones(T, n_heads_v).to(torch.float32) if beta is None else beta.to(torch.float32)
    o = torch.empty_like(v).to(torch.float32)
    if scale is None:
        scale = k.shape[-1] ** -0.5
    q = q * scale

    seq_start = 0
    for i in range(len(actual_seq_lengths)):
        if num_accepted_tokens is None:
            init_state = initial_state[ssm_state_indices[seq_start]]
        else:
            init_state = initial_state[ssm_state_indices[seq_start + num_accepted_tokens[i] - 1]]
        for head_id in range(n_heads_v):
            S = init_state[head_id]
            for slot_id in range(seq_start, seq_start + actual_seq_lengths[i]):
                q_i = q[slot_id][head_id // (n_heads_v // n_heads_k)]
                k_i = k[slot_id][head_id // (n_heads_v // n_heads_k)]
                v_i = v[slot_id][head_id]
                alpha_i = g[slot_id][head_id]
                beta_i = beta[slot_id][head_id]
                S = S * alpha_i
                x = (S * k_i.unsqueeze(-2)).sum(dim=-1)
                y = (v_i - x) * beta_i
                S_ = y[:, None] * k_i[None, :]
                S = S + S_
                initial_state[ssm_state_indices[slot_id]][head_id] = S
                o[slot_id][head_id] = (S * q_i.unsqueeze(-2)).sum(dim=-1)
        seq_start += actual_seq_lengths[i]

    return o.to(query.dtype), initial_state.to(query.dtype)


@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("mtp", [1, 2])
@pytest.mark.parametrize("headnum", [(4, 8), (8, 16), (16, 32)])
@pytest.mark.parametrize("headdim_k", [128])
@pytest.mark.parametrize("headdim_v", [128])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_recurrent_gated_delta_rule(
    batch_size,
    mtp,
    headnum,
    headdim_k,
    headdim_v,
    state_dtype,
):
    torch.manual_seed(seed)
    dtype = torch.bfloat16
    headnum_k, headnum_v = headnum
    seq_lengths = torch.ones(batch_size, dtype=torch.int32) * mtp
    T = int(torch.sum(seq_lengths))

    state = torch.rand((T, headnum_v, headdim_v, headdim_k)).to(state_dtype)
    query = torch.nn.functional.normalize(
        torch.rand((T, headnum_k, headdim_k)),
        p=2,
        dim=-1,
    ).to(dtype)
    key = torch.nn.functional.normalize(
        torch.rand((T, headnum_k, headdim_k)),
        p=2,
        dim=-1,
    ).to(dtype)
    value = torch.rand((T, headnum_v, headdim_v)).to(dtype)
    g = torch.rand((T, headnum_v), dtype=torch.float32)
    beta = torch.rand((T, headnum_v)).to(dtype)
    ssm_state_indices = torch.arange(T, dtype=torch.int32)
    num_accepted_tokens = torch.randint(1, mtp + 1, (batch_size,), dtype=torch.int32)
    scale = headdim_k**-0.5

    out_golden, state_golden = golden_recurrent_gated_delta_rule(
        query,
        key,
        value,
        state,
        beta,
        scale,
        seq_lengths,
        ssm_state_indices,
        g,
        num_accepted_tokens,
    )
    out_golden = out_golden.to(torch.float32)
    state_golden = state_golden.to(torch.float32)

    # torch_npu op expects actual_seq_lengths = [start_pos, len1, len2, ..., lenB]
    actual_seq_lengths_npu = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            seq_lengths,
        ]
    )

    state_npu = state.npu()
    npu_out = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
        query=query.npu(),
        key=key.npu(),
        value=value.npu(),
        g=g.npu(),
        beta=beta.npu(),
        state=state_npu,
        scale=scale,
        actual_seq_lengths=actual_seq_lengths_npu.npu(),
        ssm_state_indices=ssm_state_indices.npu(),
        num_accepted_tokens=num_accepted_tokens.npu(),
    )

    torch.testing.assert_close(
        npu_out.to(torch.float32).cpu(),
        out_golden,
        rtol=3e-3,
        atol=1e-2,
        equal_nan=True,
    )
    torch.testing.assert_close(
        state_npu.to(torch.float32).cpu(),
        state_golden,
        rtol=3e-3,
        atol=1e-2,
        equal_nan=True,
    )

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("mtp", [1, 2])
@pytest.mark.parametrize("headnum", [(4, 8), (8, 16)])
@pytest.mark.parametrize("headdim_k", [128])
@pytest.mark.parametrize("headdim_v", [128])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_recurrent_gated_delta_rule_no_accepted(
    batch_size,
    mtp,
    headnum,
    headdim_k,
    headdim_v,
    state_dtype,
):
    torch.manual_seed(seed)
    dtype = torch.bfloat16
    headnum_k, headnum_v = headnum
    seq_lengths = torch.ones(batch_size, dtype=torch.int32) * mtp
    T = int(torch.sum(seq_lengths))

    state = torch.rand((T, headnum_v, headdim_v, headdim_k)).to(state_dtype)
    query = torch.nn.functional.normalize(
        torch.rand((T, headnum_k, headdim_k)),
        p=2,
        dim=-1,
    ).to(dtype)
    key = torch.nn.functional.normalize(
        torch.rand((T, headnum_k, headdim_k)),
        p=2,
        dim=-1,
    ).to(dtype)
    value = torch.rand((T, headnum_v, headdim_v)).to(dtype)
    g = torch.rand((T, headnum_v), dtype=torch.float32)
    beta = torch.rand((T, headnum_v)).to(dtype)
    ssm_state_indices = torch.arange(T, dtype=torch.int32)
    scale = headdim_k**-0.5

    out_golden, state_golden = golden_recurrent_gated_delta_rule(
        query,
        key,
        value,
        state,
        beta,
        scale,
        seq_lengths,
        ssm_state_indices,
        g,
        None,
    )
    out_golden = out_golden.to(torch.float32)
    state_golden = state_golden.to(torch.float32)

    actual_seq_lengths_npu = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            seq_lengths,
        ]
    )

    state_npu = state.npu()
    npu_out = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
        query=query.npu(),
        key=key.npu(),
        value=value.npu(),
        g=g.npu(),
        beta=beta.npu(),
        state=state_npu,
        scale=scale,
        actual_seq_lengths=actual_seq_lengths_npu.npu(),
        ssm_state_indices=ssm_state_indices.npu(),
    )

    torch.testing.assert_close(
        npu_out.to(torch.float32).cpu(),
        out_golden,
        rtol=3e-3,
        atol=1e-2,
        equal_nan=True,
    )
    torch.testing.assert_close(
        state_npu.to(torch.float32).cpu(),
        state_golden,
        rtol=3e-3,
        atol=1e-2,
        equal_nan=True,
    )

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
