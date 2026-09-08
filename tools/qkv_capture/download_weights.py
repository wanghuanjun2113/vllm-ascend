#!/usr/bin/env python3
"""Download a ModelScope snapshot with bounded parallel ranges and SHA256 checks."""
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

ROOT = Path("/home/w00498770/dev/artifacts/vllm-023-qwen35-qkv/weights/Qwen3.5-35B-A3B")
BASE = "https://modelscope.cn/models/Qwen/Qwen3.5-35B-A3B/resolve/"
API = "https://modelscope.cn/api/v1/models/Qwen/Qwen3.5-35B-A3B/repo/files?Revision=master&Recursive=true"
CHUNK = 32 * 1024 * 1024

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def fetch(url, start=None, end=None):
    headers = {} if start is None else {"Range": f"bytes={start}-{end}"}
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as r:
                if start is not None:
                    expected = f"bytes {start}-{end}/"
                    if r.status != 206 or not r.headers.get("Content-Range", "").startswith(expected):
                        raise RuntimeError("Invalid HTTP byte range response")
                data = r.read()
                if start is not None and len(data) != end-start+1:
                    raise RuntimeError("Short range")
                return data
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2**attempt)

def part(job):
    row, start, end = job
    path = ROOT / row["Path"]
    marker = ROOT / ".parts" / (row["Name"] + "." + str(start))
    if marker.exists():
        return 0
    url = BASE + row["Revision"] + "/" + row["Path"] + f"?qkv_range={start}-{end}"
    data = fetch(url, start, end)
    fd = os.open(str(path) + ".partial", os.O_WRONLY)
    try:
        view = memoryview(data)
        offset = start
        while view:
            n = os.pwrite(fd, view, offset)
            offset += n
            view = view[n:]
    finally:
        os.close(fd)
    marker.touch()
    return len(data)

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / ".parts").mkdir(exist_ok=True)
    manifest = ROOT / "download_manifest.json"
    if manifest.exists():
        entries = json.loads(manifest.read_text())["files"]
    else:
        entries = json.loads(fetch(API))["Data"]["Files"]
        entries = [r for r in entries if r["Type"] == "blob"]
        manifest.write_text(json.dumps({"source": API, "files": entries}, indent=2))
    jobs = []
    for r in entries:
        p = ROOT / r["Path"]
        if p.exists() and p.stat().st_size == r["Size"] and sha(p) == r["Sha256"]:
            continue
        if r["Size"] <= CHUNK:
            data = fetch(BASE+r["Revision"]+"/"+r["Path"])
            if len(data) != r["Size"] or hashlib.sha256(data).hexdigest() != r["Sha256"]:
                raise RuntimeError("Small file checksum mismatch: "+r["Path"])
            p.write_bytes(data)
        else:
            partial = Path(str(p)+".partial")
            if not partial.exists():
                with partial.open("wb") as f:
                    f.truncate(r["Size"])
            jobs.extend((r, start, min(start+CHUNK, r["Size"])-1)
                        for start in range(0, r["Size"], CHUNK))
    started = time.monotonic()
    total = 0
    with cf.ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(part, j) for j in jobs]
        for i, future in enumerate(cf.as_completed(futures), 1):
            total += future.result()
            if i % 64 == 0 or i == len(futures):
                print(json.dumps({"parts": i, "total_parts": len(jobs), "downloaded_GiB": total/2**30,
                                  "MiB_s": total/2**20/(time.monotonic()-started)}), flush=True)
    verified = []
    for r in entries:
        p = ROOT/r["Path"]
        partial = Path(str(p)+".partial")
        check = partial if partial.exists() else p
        actual = sha(check)
        if check.stat().st_size != r["Size"] or actual != r["Sha256"]:
            raise RuntimeError("Checksum mismatch: "+r["Path"])
        if check == partial:
            os.replace(partial, p)
        verified.append({"path": r["Path"], "bytes": r["Size"], "sha256": actual})
        print("VERIFIED "+r["Path"], flush=True)
    (ROOT/"VERIFIED.json").write_text(json.dumps({"files": verified, "elapsed_s": time.monotonic()-started}, indent=2))
    print("DOWNLOAD_COMPLETE", flush=True)

if __name__ == "__main__":
    main()
