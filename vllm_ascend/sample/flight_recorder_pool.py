# SPDX-License-Identifier: Apache-2.0
"""Pool short-lived full snapshots; CPU workers persist only Top-K summaries."""
import atexit
import json
import os
import queue
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch

from vllm.v1.sample.flight_recorder import transport


class Slot:
    def __init__(self, owner, index):
        self.index = index
        r, v, b, w = owner.rows, owner.vocab, owner.batch, owner.width
        self.raw = torch.empty((r,v),dtype=torch.float32,device=owner.device)
        self.final = torch.empty_like(self.raw)
        self.selected = torch.empty((b,w),dtype=torch.int32,device=owner.device)
        self.drafts = torch.empty((b*(w-1),),dtype=torch.int32,device=owner.device)
        self.filled = torch.empty((r,),dtype=torch.bool,device=owner.device)
        self.host_raw = torch.empty((r,v),dtype=torch.float32,pin_memory=True)
        self.host_final = torch.empty_like(self.host_raw,pin_memory=True)
        self.host_selected = torch.empty((b,w),dtype=torch.int32,pin_memory=True)
        self.host_drafts = torch.empty((b*(w-1),),dtype=torch.int32,pin_memory=True)
        self.host_filled = torch.empty((r,),dtype=torch.bool,pin_memory=True)
        self.ready = torch.npu.Event()
        self.copied = torch.npu.Event()
        self.mixed = None
        self.mixed_rows = torch.empty(b,dtype=torch.int64,device=owner.device)
        self.mixed_mask = torch.empty(b,dtype=torch.bool,device=owner.device)


class SnapshotPool:
    def __init__(self, runner, logits):
        started = time.perf_counter()
        self.device = logits.device
        self.batch = runner.max_num_reqs
        self.width = runner.num_spec_tokens+1
        self.rows = max(self.batch*self.width, logits.shape[0])
        self.vocab = logits.shape[-1]
        self.k = int(os.getenv("VLLM_FLIGHT_RECORDER_TOPK","128"))
        self.probes = [int(x) for x in os.getenv(
            "VLLM_FLIGHT_RECORDER_PROBES","248044,248046").split(",")]
        if not 1 <= self.k <= self.vocab or any(x<0 or x>=self.vocab for x in self.probes):
            raise ValueError("capture K/probe outside vocabulary")
        self.size = int(os.getenv("VLLM_FLIGHT_RECORDER_QUEUE","8"))
        if self.size < 1:
            raise ValueError("pool must contain at least one slot")
        fd, self.path = tempfile.mkstemp(prefix=f"vllm-logits-{os.getpid()}-",dir="/dev/shm")
        os.close(fd)
        self.shared = np.memmap(self.path,dtype=np.float32,mode="w+",
                                shape=(self.size,2,self.rows,self.vocab))
        self.slots = [Slot(self,i) for i in range(self.size)]
        self.free = queue.Queue(self.size)
        for slot in self.slots:self.free.put(slot)
        self.jobs = queue.Queue(self.size)
        self.row_ids = torch.arange(self.rows,device=self.device)
        self.copy_stream = torch.npu.Stream(device=self.device)
        self.writer = transport()
        self.error = None
        self.closed = False
        self.submitted = self.completed = self.max_pending = 0
        self.wait_ns = 0
        self.thread = threading.Thread(target=self._copy_loop,daemon=True,name="trace-pool-copy")
        self.thread.start()
        self.info = {
            "slots":self.size,"capacity_rows":self.rows,"vocab":self.vocab,
            "npu_snapshot_bytes":self.size*2*self.rows*self.vocab*4,
            "pinned_snapshot_bytes":self.size*2*self.rows*self.vocab*4,
            "shared_snapshot_bytes":self.size*2*self.rows*self.vocab*4,
            "init_ms":(time.perf_counter()-started)*1000}
        atexit.register(self.close)

    def check(self):
        self.writer.check()
        if self.error:
            raise RuntimeError("snapshot copy failed; trace incomplete") from self.error

    def capture(self, runner, logits, spec):
        if self.closed:
            raise RuntimeError("snapshot pool is closed")
        self.check()
        start = time.perf_counter_ns()
        while True:
            try:
                slot = self.free.get(timeout=0.05)
                break
            except queue.Empty:
                self.check()
        waited = time.perf_counter_ns()-start
        self.wait_ns += waited
        return PooledCapture(self,slot,runner,logits,spec,waited)

    def release(self, slot):
        self.completed += 1
        self.free.put(slot)

    def _copy_loop(self):
        try:
            torch.npu.set_device(self.device)
            while True:
                item = self.jobs.get()
                try:
                    if item is None:
                        return
                    frame, meta = item
                    slot, r = frame.slot, frame.nrows
                    with torch.npu.stream(self.copy_stream):
                        self.copy_stream.wait_event(slot.ready)
                        if frame.mixed_n:
                            n = frame.mixed_n
                            indices = slot.mixed_rows[:n]
                            restored = torch.where(slot.mixed_mask[:n,None],
                                                   slot.mixed[:n],slot.final[indices])
                            slot.final.index_copy_(0,indices,restored)
                        slot.host_raw[:r].copy_(slot.raw[:r],non_blocking=True)
                        slot.host_final[:r].copy_(slot.final[:r],non_blocking=True)
                        slot.host_selected.copy_(slot.selected,non_blocking=True)
                        slot.host_filled[:r].copy_(slot.filled[:r],non_blocking=True)
                        if frame.ndrafts:
                            slot.host_drafts[:frame.ndrafts].copy_(
                                slot.drafts[:frame.ndrafts],non_blocking=True)
                        slot.copied.record(self.copy_stream)
                    slot.copied.synchronize()
                    np.copyto(self.shared[slot.index,0,:r],slot.host_raw[:r].numpy())
                    np.copyto(self.shared[slot.index,1,:r],slot.host_final[:r].numpy())
                    arrays = {
                        "selected":slot.host_selected[:frame.batch,:frame.width].reshape(-1),
                        "row_filled":slot.host_filled[:r]}
                    if frame.spec is not None:
                        arrays["draft_ids"] = slot.host_drafts[:frame.ndrafts]
                    self.writer.submit(meta,arrays,on_done=lambda s=slot:self.release(s))
                finally:
                    self.jobs.task_done()
        except BaseException as exc:
            self.error = exc
            (self.writer.root/f"FAILED-pool-{os.getpid()}").write_text(repr(exc))

    def flush(self):
        while self.jobs.unfinished_tasks:
            self.check()
            time.sleep(0.01)
        self.writer.flush()
        self.check()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.flush()
            self.jobs.put(None)
            self.thread.join()
            self.writer.close()
            info = {**self.info,"submitted":self.submitted,"completed":self.completed,
                    "max_pending":self.max_pending,"acquire_total_ns":self.wait_ns}
            (self.writer.root/f"pool-{os.getpid()}.json").write_text(json.dumps(info,indent=2))
            self.shared._mmap.close()
        finally:
            # Unlinking does not invalidate mappings still owned by a failed worker.
            # It prevents tmpfs files surviving an aborted recorder process.
            if os.path.exists(self.path):
                os.unlink(self.path)



class PooledCapture:
    def __init__(self,pool,slot,runner,logits,spec,wait_ns):
        start=time.perf_counter_ns()
        self.pool,self.slot,self.runner,self.spec=pool,slot,runner,spec
        self.nrows=logits.shape[0]
        if self.nrows>pool.rows or logits.shape[1]!=pool.vocab:
            raise ValueError("snapshot shape exceeds fixed pool capacity")
        self.req_ids=list(runner.input_batch.req_ids)
        self.counts=list(spec.num_draft_tokens) if spec is not None else []
        self.ndrafts=sum(self.counts)
        self.rows=None
        self.mixed_n=0
        self.wait_ns=wait_ns
        self.timestamp=time.time()
        slot.raw[:self.nrows].copy_(logits)
        slot.filled[:self.nrows].zero_()
        self.cpu_ns=time.perf_counter_ns()-start

    def processed(self,logits,metadata=None):
        start=time.perf_counter_ns()
        if isinstance(logits,tuple):
            logits,indices=logits
            if indices is not None:
                raise NotImplementedError("reduce-sample snapshot unsupported")
        if metadata is not None and metadata.all_random:
            return  # The post-truncation hook will capture the actual distribution.
        rows=self.rows
        count=logits.shape[0]
        if rows is None:rows=self.pool.row_ids[:count]
        if metadata is not None and not metadata.all_greedy:
            if self.slot.mixed is None:
                self.slot.mixed=torch.empty((self.pool.batch,self.pool.vocab),
                                             dtype=torch.float32,device=self.pool.device)
            self.slot.mixed[:count].copy_(logits)
            self.slot.mixed_rows[:count].copy_(rows)
            self.slot.mixed_mask[:count].copy_(metadata.temperature<1e-5)
            self.mixed_n=count
        else:
            self.slot.final.index_copy_(0,rows,logits.to(torch.float32))
            self.slot.filled.index_fill_(0,rows,True)
        self.cpu_ns+=time.perf_counter_ns()-start

    def finish(self,output):
        from vllm_ascend.sample import flight_recorder
        if flight_recorder.current() is self:
            flight_recorder._active=None
        start=time.perf_counter_ns()
        selected=output.sampled_token_ids
        self.batch,self.width=selected.shape
        self.slot.selected[:self.batch,:self.width].copy_(selected)
        if self.ndrafts:
            self.slot.drafts[:self.ndrafts].copy_(self.spec.draft_token_ids)
        self.slot.ready.record()
        metadata=self.runner.input_batch.sampling_metadata
        from vllm_ascend.ascend_config import get_ascend_config
        rejection=get_ascend_config().rejection_sampler_config
        meta={
            "kind":"snapshot","req_ids":self.req_ids,"width":self.width,
            "graph_mode":getattr(self.runner,"_flight_graph_mode","unknown"),
            "discard_rows":self.runner.discard_request_indices.np[
                :self.runner.num_discarded_requests].tolist(),
            "num_draft_tokens":self.counts,"top_k":self.pool.k,
            "probe_ids":self.pool.probes,"timestamp":self.timestamp,
            "all_greedy":metadata.all_greedy,"all_random":metadata.all_random,
            "block_verify":rejection.enable_block_verify,
            "entropy_verify":rejection.enable_entropy_verify,
            "posterior_threshold":rejection.posterior_threshold,
            "posterior_alpha":rejection.posterior_alpha,
            "stage_semantics":["model_pre_grammar","target_post_processing"],
            "shared_path":self.pool.path,"shared_slot":self.slot.index,
            "capacity_rows":self.pool.rows,"vocab":self.pool.vocab,
            "snapshot_rows":self.nrows,"pool_wait_ns":self.wait_ns,
            "capture_cpu_ns":self.cpu_ns+time.perf_counter_ns()-start}
        self.pool.submitted+=1
        self.pool.max_pending=max(self.pool.max_pending,self.pool.submitted-self.pool.completed)
        self.pool.jobs.put((self,meta))
