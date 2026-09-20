# SPDX-License-Identifier: Apache-2.0
"""Exercise installer mutations with real Git repositories and cache layouts."""

import hashlib
import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[3] / "tools/install_gdn_state_rows.py"
SPEC = importlib.util.spec_from_file_location("gdn_installer", SCRIPT)
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.logs = self.root / "logs"
        self.logs.mkdir()
        self.runner = installer.Runner(self.logs, os.environ.copy())

    def tearDown(self):
        self.runner.log.close()
        self.temp.cleanup()

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.repo, stderr=subprocess.STDOUT).decode().strip()

    def patch_fixture(self):
        self.git("init", "-q")
        self.git("config", "user.name", "Installer Test")
        self.git("config", "user.email", "installer-test@example.invalid")
        source = self.repo / "state.cpp"
        source.write_bytes(b"before\r\nold-index\r\nafter\r\n")
        self.git("add", ".")
        self.git("commit", "-qm", "baseline")
        base = self.git("rev-parse", "HEAD")
        source.write_bytes(b"before\r\nrow-index\r\nafter\r\n")
        self.git("commit", "-qam", "fix")
        fix = self.git("rev-parse", "HEAD")
        self.fix_commit = fix
        self.git("checkout", "--detach", base)
        with patch.object(installer, "FIX_COMMIT", fix):
            path = installer.obtain_patch(self.repo, self.runner)
        return source, path, base

    def test_patch_preserves_crlf_and_is_idempotent(self):
        source, path, head = self.patch_fixture()
        self.assertIn(b"+row-index\r\n", path.read_bytes())
        self.assertEqual(installer.apply_fix(self.repo, path, self.runner), "applied")
        self.assertEqual(source.read_bytes(), b"before\r\nrow-index\r\nafter\r\n")
        self.assertEqual(installer.apply_fix(self.repo, path, self.runner), "already-applied")
        self.assertEqual(self.git("rev-parse", "HEAD"), head)

    def test_conflict_does_not_overwrite_source(self):
        source, path, _ = self.patch_fixture()
        source.write_bytes(b"user change\r\n")
        with self.assertRaises(RuntimeError):
            installer.apply_fix(self.repo, path, self.runner)
        self.assertEqual(source.read_bytes(), b"user change\r\n")

    def test_patch_can_come_from_tool_checkout_without_network(self):
        _, path, _ = self.patch_fixture()
        expected = path.read_bytes()
        target = self.root / "target"
        target.mkdir()
        subprocess.run(["git", "init", "-q", str(target)], check=True)
        with (
            patch.object(installer, "FIX_COMMIT", self.fix_commit),
            patch.object(installer, "__file__", str(self.repo / "tools/install_gdn_state_rows.py")),
        ):
            installer.obtain_patch(target, self.runner)
        self.assertEqual(path.read_bytes(), expected)
        self.runner.log.flush()
        self.assertNotIn("git fetch", (self.logs / "install.log").read_text())

    def test_cleans_gdn_only_in_every_cached_soc(self):
        for soc in ["ascend910b", "ascend950"]:
            root = self.repo / "csrc/build/binary" / soc
            for name in ["src/recurrent_gated_delta_rule", "bin/recurrent_gated_delta_rule", "bin/other_op", "gen"]:
                (root / name).mkdir(parents=True)
            (root / "src/recurrent_gated_delta_rule/old.h").write_text("stale source")
            (root / "bin/recurrent_gated_delta_rule/old.o").write_text("stale binary")
            (root / "bin/other_op/keep.o").write_text("keep")
            (root / "gen/recurrent_gated_delta_rule_0.done").touch()
            (root / "gen/other_op.done").touch()
        removed = installer.clear_gdn_cache(self.repo)
        self.assertEqual(len(removed), 6)
        for soc in ["ascend910b", "ascend950"]:
            root = self.repo / "csrc/build/binary" / soc
            self.assertFalse((root / "src/recurrent_gated_delta_rule").exists())
            self.assertFalse((root / "bin/recurrent_gated_delta_rule").exists())
            self.assertFalse((root / "gen/recurrent_gated_delta_rule_0.done").exists())
            self.assertEqual((root / "bin/other_op/keep.o").read_text(), "keep")
            self.assertTrue((root / "gen/other_op.done").exists())
        self.assertEqual(installer.clear_gdn_cache(self.repo), [])

    def test_external_cache_symlink_is_not_followed(self):
        outside = self.root / "outside"
        outside.mkdir()
        marker = outside / "recurrent_gated_delta_rule_0.done"
        marker.touch()
        soc = self.repo / "csrc/build/binary/ascend910b"
        soc.mkdir(parents=True)
        (soc / "gen").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(RuntimeError):
            installer.clear_gdn_cache(self.repo)
        self.assertTrue(marker.exists())

    def test_live_source_is_rejected(self):
        proc = self.root / "proc/123"
        proc.mkdir(parents=True)
        maps = proc / "maps"
        maps.write_text(f"0000 r-xp {self.repo}/vllm_ascend/vllm_ascend_C.so\n")
        with self.assertRaisesRegex(RuntimeError, "PID 123"):
            installer.require_unused_source(self.repo, proc.parent)
        maps.write_text("0000 r-xp /another/source/vllm_ascend_C.so\n")
        installer.require_unused_source(self.repo, proc.parent)

    def test_device_guard_requires_the_selected_idle_device(self):
        installer.require_idle_device("| No running processes found in NPU 4 |", 4)
        for text in ["", "| No running processes found in NPU 40 |", "| 4 | 1234 | VLLMWorker |"]:
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                installer.require_idle_device(text, 4)

    def test_binary_receipt_ignores_other_operators(self):
        root = (
            self.repo / "vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910b"
        )
        for name in ["recurrent_gated_delta_rule", "other_op"]:
            (root / name).mkdir(parents=True)
            (root / name / "kernel.o").write_bytes(b"device-code")
        hashes = installer.binary_hashes(self.repo)
        self.assertEqual(len(hashes), 1)
        self.assertEqual(next(iter(hashes.values())), hashlib.sha256(b"device-code").hexdigest())


if __name__ == "__main__":
    unittest.main()
