#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Apply the pinned GDN state-row fix, rebuild native code, and verify on an idle NPU."""

import argparse
import datetime
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

FIX_COMMIT = "7b909dbdf4789b4c25eb2c9f00e447334c056a72"
FIX_REMOTE = "https://github.com/wanghuanjun2113/vllm-ascend.git"
OP_NAME = "recurrent_gated_delta_rule"
TEST_FILE = "tests/e2e/nightly/single_node/ops/singlecard_ops/test_recurrent_gated_delta_rule.py"
TEST_NAME = "test_recurrent_gated_delta_rule_fixed_state_rows"


class Runner:
    def __init__(self, directory, env):
        self.directory = directory
        self.env = env
        self.log = (directory / "install.log").open("w")

    def run(self, command, cwd, check=True):
        command = [str(x) for x in command]
        message = "$ " + shlex.join(command) + "\n"
        print(message, end="", flush=True)
        self.log.write(message)
        self.log.flush()
        output = []
        with subprocess.Popen(
            command, cwd=cwd, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                self.log.write(line)
                self.log.flush()
                output.append(line)
            code = process.wait()
        text = "".join(output)
        if check and code:
            raise RuntimeError(f"Command failed ({code}); see {self.directory / 'install.log'}")
        return code, text


def cann_environment(directory):
    env = os.environ.copy()
    setup = Path("/usr/local/Ascend/ascend-toolkit/set_env.sh")
    atb = Path("/usr/local/Ascend/nnal/atb/set_env.sh")
    if setup.is_file():
        command = 'set -e; source "$1" >/dev/null; if [ -f "$2" ]; then source "$2" --cxx_abi=1 >/dev/null; fi; env -0'
        raw = subprocess.check_output(["bash", "-c", command, "--", str(setup), str(atb)], env=env)
        env = dict(item.decode().split("=", 1) for item in raw.split(b"\0") if item)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    for key, name in [
        ("TMPDIR", "tmp"),
        ("ASCEND_CACHE_PATH", "ascend-cache"),
        ("ASCEND_WORK_PATH", "ascend"),
        ("PIP_CACHE_DIR", "pip-cache"),
        ("XDG_CACHE_HOME", "cache"),
    ]:
        path = directory / name
        path.mkdir()
        env[key] = str(path)
    env.setdefault("MAX_JOBS", "32")
    env.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", env["MAX_JOBS"])
    env.setdefault("VLLM_BATCH_INVARIANT", "0")
    env["COMPILE_CUSTOM_KERNELS"] = "1"
    return env


def require_idle_device(output, device):
    idle = f"No running processes found in NPU {device}".split()
    if not any(line.strip(" |").split() == idle for line in output.splitlines()):
        raise RuntimeError(f"NPU {device} is occupied or npu-smi could not establish that it is idle")


def require_unused_source(repo, proc=Path("/proc")):
    prefix = str(repo / "vllm_ascend") + "/"
    for maps in proc.glob("[0-9]*/maps"):
        try:
            text = maps.read_text()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if prefix in text and ("vllm_ascend_C" in text or "_cann_ops_custom" in text):
            raise RuntimeError(f"PID {maps.parent.name} is using this source's native libraries; stop it first")


def obtain_patch(repo, runner):
    # Fetch the parent too: a depth-one commit would look like a root commit to git diff.
    source = repo
    for candidate in dict.fromkeys([repo, Path(__file__).resolve().parents[1]]):
        if not (candidate / ".git").exists():
            continue
        available, _ = runner.run(["git", "cat-file", "-e", f"{FIX_COMMIT}^{{commit}}"], candidate, check=False)
        parent, _ = runner.run(["git", "cat-file", "-e", f"{FIX_COMMIT}^"], candidate, check=False)
        if available == parent == 0:
            source = candidate
            break
    else:
        runner.run(["git", "fetch", "--no-tags", "--depth=2", FIX_REMOTE, FIX_COMMIT], repo)
    path = runner.directory / "fix.patch"
    # Let Git write bytes directly: universal-newline decoding would corrupt CRLF hunks.
    runner.run(["git", "diff", "--binary", f"--output={path}", f"{FIX_COMMIT}^", FIX_COMMIT], source)
    return path


def apply_fix(repo, patch, runner):
    reverse, _ = runner.run(["git", "apply", "--reverse", "--check", patch], repo, check=False)
    if reverse == 0:
        return "already-applied"
    runner.run(["git", "apply", "--check", patch], repo)
    runner.run(["git", "apply", patch], repo)
    return "applied"


def clear_gdn_cache(repo):
    build = repo / "csrc/build"
    if not (build / "binary").resolve().is_relative_to(repo):
        raise RuntimeError("Refusing to clean a build directory outside the target repository")
    removed = []
    for soc in (build / "binary").glob("*"):
        if not soc.is_dir():
            continue
        if not soc.resolve().is_relative_to(repo):
            raise RuntimeError(f"Refusing to follow build-cache symlink: {soc}")
        for name in ["src", "bin", "gen"]:
            if not (soc / name).resolve().is_relative_to(repo):
                raise RuntimeError(f"Refusing to follow build-cache symlink: {soc / name}")
        for kind in ["src", "bin"]:
            path = soc / kind / OP_NAME
            if path.is_symlink():
                path.unlink()
                removed.append(str(path))
            elif path.exists():
                if not path.resolve().is_relative_to(repo):
                    raise RuntimeError(f"Refusing to clean external cache: {path}")
                shutil.rmtree(path)
                removed.append(str(path))
        for path in (soc / "gen").glob(f"{OP_NAME}_*.done"):
            path.unlink()
            removed.append(str(path))
    return removed


def binary_hashes(repo):
    kernel = repo / "vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/kernel"
    return {
        str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in kernel.glob(f"*/{OP_NAME}/*.o")
        if p.is_file()
    }


def verify_install(repo, runner, device):
    # Run outside the source directory so cwd cannot hide an incorrect editable installation.
    runner.env["ASCEND_RT_VISIBLE_DEVICES"] = str(device)
    runner.env.pop("PYTEST_ADDOPTS", None)
    verification = runner.directory / "verification.json"
    junit = runner.directory / "regressions.xml"
    code = """
import importlib.metadata as m, json, sys
from pathlib import Path
import torch, torch_npu, vllm_ascend
from vllm_ascend.utils import enable_custom_op
import pytest
repo, verification, junit, target = map(str, sys.argv[1:])
assert Path(vllm_ascend.__file__).resolve().parent == Path(repo) / 'vllm_ascend', vllm_ascend.__file__
assert torch.npu.device_count() == 1
assert enable_custom_op()
from vllm_ascend import vllm_ascend_C
assert Path(vllm_ascend_C.__file__).resolve().parent == Path(repo) / 'vllm_ascend'
result = pytest.main(['-q', '-o', 'addopts=', target, '--junitxml=' + junit])
if result:
    raise SystemExit(result)
Path(verification).write_text(json.dumps({
    'python': sys.executable, 'source': vllm_ascend.__file__, 'native': vllm_ascend_C.__file__,
    'versions': {p: m.version(p) for p in ['vllm', 'vllm-ascend', 'torch', 'torch-npu']}
}, indent=2))
"""
    target = str(repo / TEST_FILE) + "::" + TEST_NAME
    runner.run([sys.executable, "-c", code, repo, verification, junit, target], runner.directory)
    suites = ET.parse(junit).getroot().iter("testsuite")
    counts = {key: 0 for key in ["tests", "failures", "errors", "skipped"]}
    for suite in suites:
        for key in counts:
            counts[key] += int(suite.get(key, "0"))
    if counts != {"tests": 10, "failures": 0, "errors": 0, "skipped": 0}:
        raise RuntimeError(f"Expected all 10 NPU regressions to run and pass, got {counts}")
    return counts


def install(args, directory):
    repo = args.repo.resolve(strict=True)
    env = cann_environment(directory)
    env["SOC_VERSION"] = "ascend910b1"
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device)
    runner = Runner(directory, env)
    for command in ["git", "cmake", "g++", "npu-smi"]:
        if not shutil.which(command, path=env["PATH"]):
            raise RuntimeError(f"Missing build prerequisite: {command}")
    if not (repo / "csrc/build_aclnn.sh").is_file() or not (repo / "vllm_ascend/ops/gdn.py").is_file():
        raise RuntimeError("--repo must point to a vllm-ascend source checkout")
    if not (repo / "vllm_ascend/_cann_ops_custom").resolve().is_relative_to(repo):
        raise RuntimeError("The target custom OPP directory must remain inside the checkout")
    require_unused_source(repo)
    _, smi = runner.run(["npu-smi", "info"], directory)
    require_idle_device(smi, args.device)
    preflight = (
        "import setuptools, setuptools_scm, pybind11; import torch, torch_npu, pytest; "
        "name = torch.npu.get_device_name(0); print('Selected device:', name); "
        "assert '910B' in name.upper(), 'This installer supports Ascend 910B only'"
    )
    runner.run([sys.executable, "-c", preflight], directory)
    _, head = runner.run(["git", "rev-parse", "HEAD"], repo)
    patch = obtain_patch(repo, runner)
    before = binary_hashes(repo)
    state = apply_fix(repo, patch, runner)
    runner.run(["git", "submodule", "update", "--init", "--recursive"], repo)
    removed = clear_gdn_cache(repo)
    (directory / "cache-cleanup.json").write_text(json.dumps(removed, indent=2))
    runner.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-v", "-e", repo], directory
    )
    after = binary_hashes(repo)
    if not after:
        raise RuntimeError("No installed GDN device binaries found after compilation")
    if state == "applied" and any(before[k] == after[k] for k in before.keys() & after.keys()):
        raise RuntimeError("A GDN device binary is unchanged after applying the fix; refusing a stale-kernel install")
    runner.run(["git", "apply", "--reverse", "--check", patch], repo)
    _, installed_head = runner.run(["git", "rev-parse", "HEAD"], repo)
    if installed_head.strip() != head.strip():
        raise RuntimeError("The target HEAD changed during installation")
    _, smi = runner.run(["npu-smi", "info"], directory)
    require_idle_device(smi, args.device)
    tests = verify_install(repo, runner, args.device)
    receipt = {
        "status": "verified",
        "fix_commit": FIX_COMMIT,
        "source_head": head.strip(),
        "repo": str(repo),
        "patch_state": state,
        "device": args.device,
        "python": sys.executable,
        "soc": env["SOC_VERSION"],
        "binary_hashes_before": before,
        "binary_hashes_after": after,
        "tests": tests,
    }
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(f"VERIFIED: 10/10 NPU regressions passed. Receipt: {directory / 'receipt.json'}")
    print("Restart the service using this Python environment and source checkout.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="Existing compatible vllm-ascend Git checkout")
    parser.add_argument("--work-dir", type=Path, required=True, help="Task directory for logs, caches and receipts")
    parser.add_argument("--device", type=int, required=True, help="Idle physical 910B NPU ID used for validation")
    args = parser.parse_args()
    if sys.platform != "linux" or args.device < 0:
        parser.error("Run inside the Linux Ascend environment with a nonnegative physical NPU ID")
    directory = args.work_dir.resolve() / datetime.datetime.now().strftime("run-%Y%m%d-%H%M%S-%f")
    directory.mkdir(parents=True)
    try:
        install(args, directory)
    except Exception as error:
        (directory / "failure.txt").write_text(str(error) + "\n")
        print(
            f"INSTALLATION FAILED: {error}\nEvidence: {directory}\n"
            "Fix the error before restarting; rerun the same command.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
