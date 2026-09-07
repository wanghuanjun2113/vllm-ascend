import importlib.util
import json
import os
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from analyze import TraceStore

REPO = HERE.parents[2]
WRITER = REPO / "vllm/vllm/v1/sample/flight_recorder.py"


def load_writer():
    spec = importlib.util.spec_from_file_location("flight_writer_test", WRITER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def payload():
    meta = {"kind": "batch", "req_ids": ["a", "b"], "width": 2,
            "discard_rows": [], "num_draft_tokens": [1, 1], "top_k": 2,
            "probe_ids": [3], "all_greedy": True, "all_random": False,
            "block_verify": False, "entropy_verify": False}
    a = {"selected": np.array([1,2,0,-1], np.int32),
         "draft_ids": np.array([1,2], np.int32), "final_seen": np.ones(4, dtype=bool)}
    for stage in ("raw", "final"):
        a[stage+"_ids"] = np.array([[1,2]]*4,np.int32)
        a[stage+"_values"] = np.array([[3.,2.]]*4,np.float32)
        a[stage+"_lse"] = np.array([3.44]*4,np.float32)
        a[stage+"_counts"] = np.array([[4,0,0,0]]*4,np.int32)
        a[stage+"_probe_values"] = np.array([[3.,-1.]]*4,np.float32)
        a[stage+"_probe_ranks"] = np.array([[1,4]]*4,np.int32)
    return meta,a


def make_trace(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_FLIGHT_RECORDER_DIR",str(tmp_path))
    m = load_writer()
    t = m.Transport()
    for rid, ids in (("a",[1,2]),("b",[0])):
        t.submit({"kind":"request","request_id":rid,"prompt_token_ids":[9],
                  "sampling_params":{}, "external_req_id":rid})
        t.submit({"kind":"output","request_id":rid,"token_ids":ids,
                  "finish_reason":"stop","stop_reason":3})
    meta,arrays=payload()
    t.submit(meta,arrays)
    t.close()
    return TraceStore(tmp_path)


def test_roundtrip_mtp_alignment_and_durable_close(tmp_path,monkeypatch):
    store=make_trace(tmp_path,monkeypatch)
    assert store.audit()["complete"]
    assert len(list(tmp_path.glob("*.closed")))==1
    assert store.token(store.rows("a")[0])["source"]=="accepted_draft"
    assert store.token(store.rows("a")[1])["source"]=="bonus"
    assert store.token(store.rows("b")[0])["source"]=="replacement"
    assert store.token(store.rows("a")[0])["raw"]["probe_ranks"]==[1,4]


def test_missing_output_and_failure_markers(tmp_path,monkeypatch):
    store=make_trace(tmp_path,monkeypatch)
    (tmp_path/"FAILED-test").write_text("disk full")
    assert not store.audit()["complete"]


def test_corruption_is_detected(tmp_path,monkeypatch):
    store=make_trace(tmp_path,monkeypatch)
    data=next(tmp_path.glob("*.bin"))
    with data.open("r+b") as f:f.write(b"BROKEN")
    with pytest.raises(ValueError,match="checksum"):store.token(store.rows("a")[0])


def test_writer_error_surfaces_without_hang(tmp_path,monkeypatch):
    monkeypatch.setenv("VLLM_FLIGHT_RECORDER_DIR",str(tmp_path))
    m=load_writer();t=m.Transport()
    t.submit({"kind":"batch"})  # malformed batch models a failed writer.
    with pytest.raises(RuntimeError):t.flush()
    t.closed=True
    t.proc.wait(timeout=10)
    assert list(tmp_path.glob("FAILED-*"))


@pytest.mark.parametrize("k",[128,1024])
def test_npu_capture_matches_full_vocab(k,tmp_path,monkeypatch):
    if os.getenv("RUN_NPU_TESTS")!="1":
        pytest.skip("set RUN_NPU_TESTS=1 on a task-owned NPU")
    import torch
    import torch_npu
    from vllm_ascend.sample.flight_recorder import Capture
    import vllm_ascend.ascend_config as cfg
    import vllm_ascend.sample.flight_recorder as mod
    monkeypatch.setenv("VLLM_FLIGHT_RECORDER_TOPK",str(k))
    monkeypatch.setenv("VLLM_FLIGHT_RECORDER_PROBES","2046,2047")
    config=SimpleNamespace(rejection_sampler_config=SimpleNamespace(
        enable_block_verify=False,enable_entropy_verify=False,
        posterior_threshold=0.0,posterior_alpha=0.0))
    monkeypatch.setattr(cfg,"get_ascend_config",lambda:config)
    transport=SimpleNamespace()
    def submit(meta, arrays, event):
        event.synchronize()
        transport.meta=meta
        transport.arrays={key:t.numpy().copy() for key,t in arrays.items()}
    transport.submit=submit
    monkeypatch.setattr(mod,"transport",lambda:transport)
    runner=SimpleNamespace(input_batch=SimpleNamespace(
        req_ids=["test"],sampling_metadata=SimpleNamespace(all_greedy=True,all_random=False)),
        discard_request_indices=SimpleNamespace(np=np.array([],np.int64)),num_discarded_requests=0)
    raw=torch.arange(2048,dtype=torch.float32).view(1,-1)/1000
    cap=Capture(runner,raw.npu(),None)
    final=raw.clone();final[0,2047]=-float("inf")
    cap.processed(final.npu())
    cap.finish(SimpleNamespace(sampled_token_ids=torch.tensor([[1]],device="npu")))
    for stage,full in (("raw",raw),("final",final)):
        a=transport.arrays
        vals,ids=torch.topk(full,k,dim=-1)
        np.testing.assert_array_equal(a[stage+"_ids"],ids.numpy())
        np.testing.assert_allclose(a[stage+"_values"],vals.numpy(),rtol=0,atol=0)
        np.testing.assert_allclose(a[stage+"_lse"],torch.logsumexp(full,-1).numpy(),rtol=1e-6)
        assert a[stage+"_probe_ranks"][0,0]>k
        assert a[stage+"_probe_values"][0,0]==raw[0,1].item()
    assert transport.arrays["final_counts"][0,3]==1


def test_index_payload_mismatch(tmp_path,monkeypatch):
    store=make_trace(tmp_path,monkeypatch)
    db=sqlite3.connect(next(tmp_path.glob("*.sqlite")))
    db.execute("UPDATE tokens SET token_id=99 WHERE request_id='a' AND position=0")
    db.commit()
    with pytest.raises(ValueError,match="disagree"):
        store.token(store.rows("a")[0])


def test_missing_processed_stage(tmp_path,monkeypatch):
    monkeypatch.setenv("VLLM_FLIGHT_RECORDER_DIR",str(tmp_path))
    m=load_writer();t=m.Transport()
    t.submit({"kind":"request","request_id":"a","prompt_token_ids":[],"sampling_params":{}})
    t.submit({"kind":"output","request_id":"a","token_ids":[1,2],"finish_reason":"stop","stop_reason":3})
    meta,arrays=payload();arrays["final_seen"][0]=False
    meta["discard_rows"]=[1]
    t.submit(meta,arrays);t.close()
    assert not TraceStore(tmp_path).audit()["complete"]


@pytest.mark.parametrize("k",[128,1024])
def test_npu_mtp_mixed_lengths_full_qwen_vocab(k,monkeypatch):
    if os.getenv("RUN_NPU_TESTS")!="1":
        pytest.skip("requires a task-owned NPU")
    import torch
    import torch_npu
    import vllm_ascend.sample.flight_recorder as mod
    import vllm_ascend.ascend_config as cfg
    monkeypatch.setenv("VLLM_FLIGHT_RECORDER_TOPK",str(k))
    monkeypatch.setenv("VLLM_FLIGHT_RECORDER_PROBES","248044,248046")
    config=SimpleNamespace(rejection_sampler_config=SimpleNamespace(
        enable_block_verify=False,enable_entropy_verify=False,
        posterior_threshold=0.,posterior_alpha=0.))
    monkeypatch.setattr(cfg,"get_ascend_config",lambda:config)
    sink={}
    def submit(meta,arrays,event):
        event.synchronize()
        sink.update({key:t.numpy().copy() for key,t in arrays.items()})
    monkeypatch.setattr(mod,"transport",lambda:SimpleNamespace(submit=submit))
    runner=SimpleNamespace(input_batch=SimpleNamespace(
        req_ids=["a","b"],sampling_metadata=SimpleNamespace(all_greedy=True,all_random=False)),
        discard_request_indices=SimpleNamespace(np=np.array([],np.int64)),num_discarded_requests=0)
    raw=torch.arange(248320,dtype=torch.float32).repeat(6,1)/10000
    raw+=torch.arange(6,dtype=torch.float32).unsqueeze(1)
    final=raw.clone();final[:,248046]=-float("inf")
    spec=SimpleNamespace(num_draft_tokens=[3,1],
        draft_token_ids=torch.tensor([1,100,101,3],device="npu"))
    cap=mod.Capture(runner,raw.npu(),spec)
    cap.rows=torch.tensor([3,5],device="npu")
    cap.processed(final[[3,5]].npu(),runner.input_batch.sampling_metadata)
    cap.rows=torch.tensor([0,1,2,4],device="npu")
    cap.processed(final[[0,1,2,4]].npu())
    cap.finish(SimpleNamespace(sampled_token_ids=torch.tensor(
        [[1,2,-1,-1],[3,4,-1,-1]],device="npu")))
    slots=[0,1,4,5];source_rows=[0,1,4,5];chosen=[1,2,3,4]
    for stage,tensor in (("raw",raw),("final",final)):
        vals,ids=torch.topk(tensor,k,dim=-1)
        np.testing.assert_array_equal(sink[stage+"_ids"][slots],ids[source_rows].numpy())
        np.testing.assert_allclose(sink[stage+"_values"][slots],vals[source_rows].numpy(),rtol=0,atol=0)
        np.testing.assert_allclose(sink[stage+"_lse"][slots],
            torch.logsumexp(tensor,-1)[source_rows].numpy(),rtol=1e-6)
        np.testing.assert_array_equal(sink[stage+"_probe_values"][slots,0],
            tensor[source_rows,chosen].numpy())
    assert sink["final_seen"][slots].all()


def test_request_metadata_preserves_default_fields(monkeypatch,tmp_path):
    import msgspec
    monkeypatch.setenv("VLLM_FLIGHT_RECORDER_DIR",str(tmp_path))
    class Params(msgspec.Struct, omit_defaults=True):
        temperature: float = 0.0
        top_p: float = 1.0
        seed: int | None = None
    m=load_writer();events=[]
    monkeypatch.setattr(m,"record_event",lambda kind,rid,**values:events.append(values))
    m.request_start(SimpleNamespace(request_id="r",external_req_id="r",
                    prompt_token_ids=[1],sampling_params=Params()))
    assert events[0]["sampling_params"]=={"temperature":0.0,"top_p":1.0,"seed":None}


@pytest.mark.parametrize("k",[128,1024])
def test_pooled_snapshot_survives_source_reuse_and_backpressure(k,tmp_path,monkeypatch):
    if os.getenv("RUN_NPU_TESTS")!="1":
        pytest.skip("requires a task-owned NPU")
    import torch
    import torch_npu
    import vllm_ascend.sample.flight_recorder_pool as poolmod
    import vllm_ascend.ascend_config as cfg
    monkeypatch.setenv("VLLM_FLIGHT_RECORDER_DIR",str(tmp_path))
    monkeypatch.setenv("VLLM_FLIGHT_RECORDER_TOPK",str(k))
    monkeypatch.setenv("VLLM_FLIGHT_RECORDER_PROBES","2046,2047")
    monkeypatch.setenv("VLLM_FLIGHT_RECORDER_QUEUE","2")
    m=load_writer();writer=m.Transport()
    monkeypatch.setattr(poolmod,"transport",lambda:writer)
    config=SimpleNamespace(rejection_sampler_config=SimpleNamespace(
        enable_block_verify=False,enable_entropy_verify=False,
        posterior_threshold=0.,posterior_alpha=0.))
    monkeypatch.setattr(cfg,"get_ascend_config",lambda:config)
    runner=SimpleNamespace(max_num_reqs=2,num_spec_tokens=3,
        input_batch=SimpleNamespace(req_ids=["a","b"],
            sampling_metadata=SimpleNamespace(all_greedy=True,all_random=False)),
        discard_request_indices=SimpleNamespace(np=np.array([],np.int64)),num_discarded_requests=0)
    base=torch.arange(6*2048,dtype=torch.float32,device="npu").reshape(6,2048)
    source=base.clone()
    spec=SimpleNamespace(num_draft_tokens=[3,1],
        draft_token_ids=torch.tensor([1,100,101,3],device="npu"))
    selected=torch.tensor([[1,2,-1,-1],[3,4,-1,-1]],device="npu")
    pool=poolmod.SnapshotPool(runner,source)
    for rid in ("a","b"):
        writer.submit({"kind":"request","request_id":rid,"prompt_token_ids":[],
                       "sampling_params":{},"sampling_params_complete":True})
    for step in range(12):
        runner.input_batch.req_ids=["a","b"]
        source.copy_(base+step*20)
        frame=pool.capture(runner,source,spec)
        final=source.clone();final[:,2047]=-float("inf")
        frame.rows=torch.tensor([3,5],device="npu")
        frame.processed(final[[3,5]],runner.input_batch.sampling_metadata)
        frame.rows=torch.tensor([0,1,2,4],device="npu")
        frame.processed(final[[0,1,2,4]])
        frame.finish(SimpleNamespace(sampled_token_ids=selected))
        source.fill_(-999)  # Reused immediately, before the consumer finishes.
        runner.input_batch.req_ids=["wrong","request"]
    for rid,ids in (("a",[1,2]*12),("b",[3,4]*12)):
        writer.submit({"kind":"output","request_id":rid,"token_ids":ids,
                       "finish_reason":"stop","stop_reason":2047})
    pool.close()
    assert not Path(pool.path).exists()
    store=TraceStore(tmp_path)
    assert store.audit()["complete"]
    for rid,source_rows,tokens in (("a",[0,1],[1,2]),("b",[4,5],[3,4])):
        rows=store.rows(rid)
        assert len(rows)==24
        for i,item in enumerate(rows):
            token=store.token(item)
            expected=source_rows[i%2]*2048+tokens[i%2]+(i//2)*20
            assert token["raw"]["probe_logits"][0]==expected
            assert token["final"]["counts"][3]==1
    info=json.loads(next(tmp_path.glob("pool-*.json")).read_text())
    assert info["completed"]==info["submitted"]==12
    assert info["max_pending"]<=2
