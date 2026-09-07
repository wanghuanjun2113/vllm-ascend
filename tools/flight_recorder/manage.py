#!/usr/bin/env python3
"""Manage only this task's server process; never manages Docker lifecycle."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ART = Path("/home/w00498770/dev/artifacts/vllm-023-logits-save")
BASE = Path("/home/w00498770/dev/worktrees/vllm-023-logits-save")


def group_members(pgid):
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry/"stat").read_text().rsplit(")",1)[1].split()
            if fields[0] != "Z" and int(fields[2]) == pgid and int(fields[3]) == pgid:
                members.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return members


def cleanup_snapshot_files(owners, server_pid):
    removed = []
    for owner in owners:
        proc = Path(f"/proc/{owner}/stat")
        if proc.exists():
            try:
                if proc.read_text().rsplit(")",1)[1].split()[0] != "Z":
                    continue
            except FileNotFoundError:
                pass
        for path in Path("/dev/shm").glob(f"vllm-logits-{owner}-*"):
            if path.is_file() and path.stat().st_uid == os.getuid():
                removed.append({"path":str(path),"bytes":path.stat().st_size})
                path.unlink()
    if removed:
        (ART/f"pool-cleanup-{server_pid}.json").write_text(json.dumps(
            {"ownership":"verified task server session, after all members exited",
             "removed":removed},indent=2))


def stop():
    path = ART / "server.pid"
    if not path.exists():
        return
    pid = int(path.read_text())
    proc = Path(f"/proc/{pid}/cmdline")
    if not proc.exists() or not proc.read_bytes():
        if group_members(pid):
            raise RuntimeError("API exited with orphan workers; verify task ownership before cleanup")
        return
    command = proc.read_bytes()
    if b"vllm.entrypoints.openai.api_server" not in command or b"18327" not in command:
        raise RuntimeError("PID is not the task-owned server")
    if os.getpgid(pid) != pid:
        raise RuntimeError("server is not in its task-owned session")
    owners = set(group_members(pid))
    os.kill(pid, signal.SIGTERM)
    for _ in range(150):
        current = group_members(pid)
        owners.update(current)
        if not current:
            cleanup_snapshot_files(owners, pid)
            return
        time.sleep(0.1)
    # Ownership was verified while the original session leader was alive.
    if group_members(pid):
        os.killpg(pid, signal.SIGKILL)
    for _ in range(50):
        current = group_members(pid)
        owners.update(current)
        if not current:
            cleanup_snapshot_files(owners, pid)
            return
        time.sleep(0.1)
    raise RuntimeError("task worker group did not exit")


def start(name, k):
    config = json.loads((ART / "launch.json").read_text())
    env = os.environ.copy()
    env.update(config["env"])
    env.pop("VLLM_FLIGHT_RECORDER_DIR", None)
    trace = ART / name / "trace"
    run = ART / name
    run.mkdir(exist_ok=False)
    if k:
        env["VLLM_FLIGHT_RECORDER_DIR"] = str(trace)
        env["VLLM_FLIGHT_RECORDER_TOPK"] = str(k)
        env["VLLM_FLIGHT_RECORDER_PROBES"] = "248044,248046"
    manifest = {"schema_version": 1, "name": name, "capture_top_k": k,
                "launch": config, "timestamp": time.time(), "repos": {}}
    for repo in ("vllm","vllm-ascend"):
        def git(*args):
            return subprocess.check_output(["git","-C",str(BASE/repo),*args])
        manifest["repos"][repo] = {
            "head":git("rev-parse","HEAD").decode().strip(),
            "diff_sha256":hashlib.sha256(git("diff","HEAD")).hexdigest()}
    model = Path("/mnt/weights/Qwen3.6-27B-w8a8")
    manifest["model_files"] = {
        p.name: {"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in model.iterdir() if p.suffix in (".json",".jinja")}
    (run/"manifest.json").write_text(json.dumps(manifest,indent=2))
    with (run/"server.log").open("w") as f:
        proc = subprocess.Popen(config["args"],env=env,cwd=BASE/"vllm-ascend",
                                stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    (ART/"server.pid").write_text(str(proc.pid))
    print(proc.pid,run,flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=["start","stop"])
    p.add_argument("--name")
    p.add_argument("--top-k",type=int,default=0)
    args=p.parse_args()
    if args.action=="stop":
        stop()
    else:
        if not args.name:
            p.error("--name required")
        start(args.name,args.top_k)
