from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from harness.agents.agent_context_compiler import (
  AgentContextCompilationError,
  compile_agent_context_packet,
)
from harness.repo_snapshot.repo_snapshot_compiler import (
  RepoSnapshotCompilationError,
  compile_repo_snapshot_packet,
)
from harness.repo_snapshot.repo_snapshot_packet import RepoSnapshotPacket
from harness.runtime.api_call_packet_builder import build_api_call_packet
from harness.runtime.supplementary_context import SupplementaryContextEntry
from harness.runtime.task import task_from_cli
from tests.agent_test_support import create_test_project_manager_agent


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = REPO_ROOT / "harness"
MANIFEST_PATH = HARNESS_ROOT / "project_spec" / "static_context_packet.manifest.json"
LIVE_AGENT_PATH = HARNESS_ROOT / "agents" / "project_manager.agent.json"
AGENT_PATH = create_test_project_manager_agent(LIVE_AGENT_PATH)
REPO_SNAPSHOT_SCHEMA_PATH = (
  HARNESS_ROOT / "repo_snapshot" / "RepoSnapshotPacket.schema.json"
)


def load_json(path: Path) -> dict:
  return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def init_git_repo(repo_root: Path) -> None:
  subprocess.run(
    ["git", "init"],
    cwd=repo_root,
    capture_output=True,
    text=True,
    check=True,
  )


def build_repo_fixture(repo_root: Path) -> None:
  init_git_repo(repo_root)
  (repo_root / "README.md").write_text("# Example Repo\n", encoding="utf-8")
  (repo_root / "src").mkdir()
  (repo_root / "src" / "app.py").write_text("print('hello')\n", encoding="utf-8")
  (repo_root / ".gitignore").write_text("ignored.txt\nruns/\n", encoding="utf-8")
  (repo_root / "ignored.txt").write_text("ignore me\n", encoding="utf-8")
  (repo_root / "runs").mkdir()
  (repo_root / "runs" / "ignored_artifact.json").write_text(
    '{"generated": true}\n',
    encoding="utf-8",
  )
  (repo_root / "binary.bin").write_bytes(b"\x00\x01\x02\xff")
  (repo_root / "large.txt").write_text("x" * 256, encoding="utf-8")
  (repo_root / "harness" / "runs").mkdir(parents=True)
  (repo_root / "harness" / "runs" / "generated.json").write_text(
    '{"generated": true}\n',
    encoding="utf-8",
  )
  (repo_root / "harness" / "state" / "ledgers").mkdir(parents=True)
  (repo_root / "harness" / "state" / "ledgers" / "api_call_ledger.jsonl").write_text(
    '{"ledger_version":"0.1","route":"plan"}\n',
    encoding="utf-8",
  )
  (repo_root / "harness" / "runtime").mkdir(parents=True)
  (repo_root / "harness" / "runtime" / "core.py").write_text(
    "def core():\n  return 'ok'\n",
    encoding="utf-8",
  )
  (repo_root / "harness" / "runtime" / "Generated.schema.json").write_text(
    '{"$schema":"https://json-schema.org/draft/2020-12/schema"}\n',
    encoding="utf-8",
  )


class RepoSnapshotCompilerTests(unittest.TestCase):
  def test_repo_snapshot_compiler_includes_single_explicit_path(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      repo_root = Path(temp_directory) / "repo"
      repo_root.mkdir()
      build_repo_fixture(repo_root)
      output_path = repo_root / "snapshot.json"

      packet = compile_repo_snapshot_packet(
        repo_root=repo_root,
        output_path=output_path,
        mode="paths",
        requested_paths=["README.md"],
      )

      self.assertEqual(packet.summary.included_count, 1)
      self.assertEqual(packet.files[0].path, "README.md")

  def test_repo_snapshot_compiler_includes_explicit_harness_state_ledger_path(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      repo_root = Path(temp_directory) / "repo"
      repo_root.mkdir()
      build_repo_fixture(repo_root)
      output_path = repo_root / "snapshot.json"

      packet = compile_repo_snapshot_packet(
        repo_root=repo_root,
        output_path=output_path,
        mode="paths",
        requested_paths=["harness/state/ledgers/api_call_ledger.jsonl"],
      )

      self.assertEqual(
        [file.path for file in packet.files],
        ["harness/state/ledgers/api_call_ledger.jsonl"],
      )
      self.assertTrue(packet.selection.explicit_path_overrides_default_exclusions)
      self.assertTrue(packet.files[0].explicit_requested_path)

  def test_repo_snapshot_compiler_includes_explicit_gitignored_runs_path(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      repo_root = Path(temp_directory) / "repo"
      repo_root.mkdir()
      build_repo_fixture(repo_root)
      output_path = repo_root / "snapshot.json"

      packet = compile_repo_snapshot_packet(
        repo_root=repo_root,
        output_path=output_path,
        mode="paths",
        requested_paths=["runs/ignored_artifact.json"],
      )

      self.assertEqual([file.path for file in packet.files], ["runs/ignored_artifact.json"])
      self.assertTrue(packet.selection.explicit_path_overrides_default_exclusions)
      self.assertTrue(packet.files[0].explicit_requested_path)

  def test_repo_snapshot_compiler_includes_files_matched_by_glob(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      repo_root = Path(temp_directory) / "repo"
      repo_root.mkdir()
      build_repo_fixture(repo_root)
      output_path = repo_root / "snapshot.json"

      packet = compile_repo_snapshot_packet(
        repo_root=repo_root,
        output_path=output_path,
        mode="globs",
        requested_globs=["src/*.py"],
      )

      self.assertEqual([file.path for file in packet.files], ["src/app.py"])
      self.assertTrue(packet.selection.harness_excluded)

  def test_repo_snapshot_compiler_default_traversal_still_excludes_harness(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      repo_root = Path(temp_directory) / "repo"
      repo_root.mkdir()
      build_repo_fixture(repo_root)
      output_path = repo_root / "snapshot.json"

      packet = compile_repo_snapshot_packet(
        repo_root=repo_root,
        output_path=output_path,
        mode="all_admissible",
      )

      self.assertFalse(
        any(file.path.startswith("harness/") for file in packet.files)
      )
      self.assertTrue(packet.selection.harness_excluded)

  def test_repo_snapshot_compiler_includes_explicit_harness_path_without_include_harness(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      repo_root = Path(temp_directory) / "repo"
      repo_root.mkdir()
      build_repo_fixture(repo_root)
      output_path = repo_root / "snapshot.json"

      packet = compile_repo_snapshot_packet(
        repo_root=repo_root,
        output_path=output_path,
        mode="paths",
        requested_paths=["harness/runtime/core.py"],
      )

      self.assertEqual(packet.summary.included_count, 1)
      self.assertEqual([file.path for file in packet.files], ["harness/runtime/core.py"])
      self.assertFalse(packet.selection.include_harness)
      self.assertTrue(packet.selection.harness_excluded)
      self.assertTrue(packet.selection.explicit_path_overrides_default_exclusions)
      self.assertTrue(packet.files[0].explicit_requested_path)

  def test_repo_snapshot_compiler_requested_glob_does_not_broaden_harness_inclusion(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      repo_root = Path(temp_directory) / "repo"
      repo_root.mkdir()
      build_repo_fixture(repo_root)
      output_path = repo_root / "snapshot.json"

      packet = compile_repo_snapshot_packet(
        repo_root=repo_root,
        output_path=output_path,
        mode="globs",
        requested_globs=["harness/**/*.json"],
      )

      self.assertEqual(packet.summary.included_count, 0)
      self.assertTrue(
        all(omitted.reason == "harness_excluded" for omitted in packet.omitted_files)
      )

  def test_repo_snapshot_compiler_omits_harness_glob_by_default(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      repo_root = Path(temp_directory) / "repo"
      repo_root.mkdir()
      build_repo_fixture(repo_root)
      output_path = repo_root / "snapshot.json"

      packet = compile_repo_snapshot_packet(
        repo_root=repo_root,
        output_path=output_path,
        mode="globs",
        requested_globs=["harness/**/*.py"],
      )

      self.assertEqual(packet.summary.included_count, 0)
      self.assertFalse(packet.selection.include_harness)
      self.assertTrue(packet.selection.harness_excluded)
      self.assertTrue(
        all(omitted.reason == "harness_excluded" for omitted in packet.omitted_files)
      )

  def test_repo_snapshot_compiler_supports_all_admissible_mode(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      repo_root = Path(temp_directory) / "repo"
      repo_root.mkdir()
      build_repo_fixture(repo_root)
      output_path = repo_root / "snapshot.json"

      packet = compile_repo_snapshot_packet(
        repo_root=repo_root,
        output_path=output_path,
        mode="all_admissible",
      )

      included_paths = {file.path for file in packet.files}
      self.assertIn("README.md", included_paths)
      self.assertIn("src/app.py", included_paths)
      self.assertNotIn("harness/runtime/core.py", included_paths)
      self.assertFalse(packet.selection.include_harness)
      self.assertTrue(packet.selection.harness_excluded)

  def test_repo_snapshot_compiler_path_may_explicitly_include_harness(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      repo_root = Path(temp_directory) / "repo"
      repo_root.mkdir()
      build_repo_fixture(repo_root)
      output_path = repo_root / "snapshot.json"

      packet = compile_repo_snapshot_packet(
        repo_root=repo_root,
        output_path=output_path,
        mode="paths",
        include_harness=True,
        requested_paths=["harness/runtime/core.py"],
      )

      self.assertEqual([file.path for file in packet.files], ["harness/runtime/core.py"])
      self.assertTrue(packet.selection.include_harness)
      self.assertFalse(packet.selection.harness_excluded)

  def test_repo_snapshot_compiler_all_admissible_may_explicitly_include_harness(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      repo_root = Path(temp_directory) / "repo"
      repo_root.mkdir()
      build_repo_fixture(repo_root)
      output_path = repo_root / "snapshot.json"

      packet = compile_repo_snapshot_packet(
        repo_root=repo_root,
        output_path=output_path,
        mode="all_admissible",
        include_harness=True,
      )

      included_paths = {file.path for file in packet.files}
      self.assertIn("harness/runtime/core.py", included_paths)
      self.assertTrue(packet.selection.include_harness)
      self.assertFalse(packet.selection.harness_excluded)

  def test_repo_snapshot_compiler_includes_explicit_gitignored_file(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      repo_root = Path(temp_directory) / "repo"
      repo_root.mkdir()
      build_repo_fixture(repo_root)
      output_path = repo_root / "snapshot.json"

      packet = compile_repo_snapshot_packet(
        repo_root=repo_root,
        output_path=output_path,
        mode="paths",
        requested_paths=["ignored.txt"],
      )

      self.assertEqual(packet.summary.included_count, 1)
      self.assertEqual([file.path for file in packet.files], ["ignored.txt"])
      self.assertTrue(packet.selection.explicit_path_overrides_default_exclusions)
      self.assertTrue(packet.files[0].explicit_requested_path)

  def test_repo_snapshot_compiler_excludes_dot_git(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      repo_root = Path(temp_directory) / "repo"
      repo_root.mkdir()
      build_repo_fixture(repo_root)
      output_path = repo_root / "snapshot.json"

      packet = compile_repo_snapshot_packet(
        repo_root=repo_root,
        output_path=output_path,
        mode="all_admissible",
      )

      self.assertFalse(any(file.path.startswith(".git/") for file in packet.files))

  def test_repo_snapshot_compiler_includes_explicit_harness_runs_file(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      repo_root = Path(temp_directory) / "repo"
      repo_root.mkdir()
      build_repo_fixture(repo_root)
      output_path = repo_root / "snapshot.json"

      packet = compile_repo_snapshot_packet(
        repo_root=repo_root,
        output_path=output_path,
        mode="paths",
        requested_paths=["harness/runs/generated.json"],
      )

      self.assertEqual(packet.summary.included_count, 1)
      self.assertEqual([file.path for file in packet.files], ["harness/runs/generated.json"])
      self.assertTrue(packet.selection.explicit_path_overrides_default_exclusions)
      self.assertTrue(packet.files[0].explicit_requested_path)

  def test_repo_snapshot_compiler_explicit_binary_file_fails(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      repo_root = Path(temp_directory) / "repo"
      repo_root.mkdir()
      build_repo_fixture(repo_root)
      with self.assertRaises(RepoSnapshotCompilationError) as error:
        compile_repo_snapshot_packet(
          repo_root=repo_root,
          output_path=repo_root / "snapshot.json",
          mode="paths",
          requested_paths=["binary.bin"],
        )

      self.assertIn("binary.bin", str(error.exception))
      self.assertIn("not UTF-8 text", str(error.exception))

  def test_repo_snapshot_compiler_explicit_dot_git_path_fails(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      repo_root = Path(temp_directory) / "repo"
      repo_root.mkdir()
      build_repo_fixture(repo_root)

      with self.assertRaises(RepoSnapshotCompilationError) as error:
        compile_repo_snapshot_packet(
          repo_root=repo_root,
          output_path=repo_root / "snapshot.json",
          mode="paths",
          requested_paths=[".git/config"],
        )

      self.assertIn(".git/config", str(error.exception))
      self.assertIn("hard-denied", str(error.exception))

  def test_repo_snapshot_compiler_explicit_outside_repo_path_fails(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      repo_root = Path(temp_directory) / "repo"
      repo_root.mkdir()
      build_repo_fixture(repo_root)

      with self.assertRaises(RepoSnapshotCompilationError) as error:
        compile_repo_snapshot_packet(
          repo_root=repo_root,
          output_path=repo_root / "snapshot.json",
          mode="paths",
          requested_paths=["../outside.txt"],
        )

      self.assertIn("outside repo root", str(error.exception))

  def test_repo_snapshot_compiler_explicit_missing_path_fails_clearly(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      repo_root = Path(temp_directory) / "repo"
      repo_root.mkdir()
      build_repo_fixture(repo_root)

      with self.assertRaises(RepoSnapshotCompilationError) as error:
        compile_repo_snapshot_packet(
          repo_root=repo_root,
          output_path=repo_root / "snapshot.json",
          mode="paths",
          requested_paths=["missing.txt"],
        )

      self.assertIn("missing.txt", str(error.exception))
      self.assertIn("was not found", str(error.exception))

  def test_repo_snapshot_compiler_explicit_directory_path_fails(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      repo_root = Path(temp_directory) / "repo"
      repo_root.mkdir()
      build_repo_fixture(repo_root)

      with self.assertRaises(RepoSnapshotCompilationError) as error:
        compile_repo_snapshot_packet(
          repo_root=repo_root,
          output_path=repo_root / "snapshot.json",
          mode="paths",
          requested_paths=["src"],
        )

      self.assertIn("directory", str(error.exception))

  def test_repo_snapshot_compiler_explicit_oversized_file_fails(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      repo_root = Path(temp_directory) / "repo"
      repo_root.mkdir()
      build_repo_fixture(repo_root)
      with self.assertRaises(RepoSnapshotCompilationError) as error:
        compile_repo_snapshot_packet(
          repo_root=repo_root,
          output_path=repo_root / "snapshot.json",
          mode="paths",
          requested_paths=["large.txt"],
          max_file_bytes=32,
        )

      self.assertIn("large.txt", str(error.exception))
      self.assertIn("exceeds max_file_bytes", str(error.exception))

  def test_repo_snapshot_packet_validates_against_generated_schema(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      repo_root = Path(temp_directory) / "repo"
      repo_root.mkdir()
      build_repo_fixture(repo_root)
      output_path = repo_root / "snapshot.json"

      packet = compile_repo_snapshot_packet(
        repo_root=repo_root,
        output_path=output_path,
        mode="paths",
        requested_paths=["README.md"],
      )

      schema = load_json(REPO_SNAPSHOT_SCHEMA_PATH)
      Draft202012Validator.check_schema(schema)
      Draft202012Validator(schema).validate(load_json(output_path))
      self.assertIsInstance(packet, RepoSnapshotPacket)

  def test_agent_input_policy_with_repo_snapshot_resolves_into_agent_context_packet(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      repo_root = temp_root / "repo"
      repo_root.mkdir()
      build_repo_fixture(repo_root)
      agent_data = load_json(AGENT_PATH)
      agent_data["agent_input_policy"] = [
        entry
        for entry in agent_data["agent_input_policy"]
        if entry["input_id"] != "repo_snapshot_packet"
      ]
      agent_data["agent_input_policy"].append(
        {
          "input_id": "repo_snapshot_packet",
          "required": True,
          "schema_ref": "../repo_snapshot/RepoSnapshotPacket.schema.json",
          "resolution": {
            "mode": "paths",
            "paths": ["README.md"],
          },
        }
      )
      agent_path = temp_root / "pm_with_snapshot.agent.json"
      write_json(agent_path, agent_data)
      output_path = temp_root / "agent_context_packet.json"

      packet = compile_agent_context_packet(
        agent_path=agent_path,
        output_path=output_path,
        manifest_path=MANIFEST_PATH,
        repo_root=repo_root,
        harness_root=REPO_ROOT,
        target_repo_root=REPO_ROOT,
        static_context_output_path=temp_root / "static_context_packet.json",
        repo_snapshot_output_path=temp_root / "repo_snapshot_packet.json",
      )

      self.assertIsNotNone(packet.resolved_inputs.repo_snapshot_packet)
      repo_snapshot_packet = packet.resolved_inputs.repo_snapshot_packet
      self.assertIsNotNone(repo_snapshot_packet)
      self.assertEqual(repo_snapshot_packet.files[0].path, "README.md")
      self.assertTrue(repo_snapshot_packet.selection.explicit_path_overrides_default_exclusions)
      self.assertTrue(repo_snapshot_packet.files[0].explicit_requested_path)
      self.assertEqual(packet.input_coverage[1].status, "included")

  def test_agent_input_policy_repo_snapshot_omits_harness_by_default(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      repo_root = temp_root / "repo"
      repo_root.mkdir()
      build_repo_fixture(repo_root)
      agent_data = load_json(AGENT_PATH)
      agent_data["agent_input_policy"] = [
        entry
        for entry in agent_data["agent_input_policy"]
        if entry["input_id"] != "repo_snapshot_packet"
      ]
      agent_data["agent_input_policy"].append(
        {
          "input_id": "repo_snapshot_packet",
          "required": True,
          "schema_ref": "../repo_snapshot/RepoSnapshotPacket.schema.json",
          "resolution": {
            "mode": "paths",
            "paths": ["harness/runtime/core.py"],
          },
        }
      )
      agent_path = temp_root / "pm_harness_omitted.agent.json"
      write_json(agent_path, agent_data)

      packet = compile_agent_context_packet(
        agent_path=agent_path,
        output_path=temp_root / "agent_context_packet.json",
        manifest_path=MANIFEST_PATH,
        repo_root=repo_root,
        harness_root=REPO_ROOT,
        target_repo_root=REPO_ROOT,
        static_context_output_path=temp_root / "static_context_packet.json",
        repo_snapshot_output_path=temp_root / "repo_snapshot_packet.json",
      )

      self.assertEqual(packet.input_coverage[1].status, "included")
      self.assertIsNotNone(packet.resolved_inputs.repo_snapshot_packet)
      repo_snapshot_packet = packet.resolved_inputs.repo_snapshot_packet
      self.assertEqual(
        [file.path for file in repo_snapshot_packet.files],
        ["harness/runtime/core.py"],
      )
      self.assertFalse(repo_snapshot_packet.selection.include_harness)
      self.assertTrue(repo_snapshot_packet.selection.harness_excluded)
      self.assertTrue(repo_snapshot_packet.selection.explicit_path_overrides_default_exclusions)
      self.assertTrue(repo_snapshot_packet.files[0].explicit_requested_path)

  def test_agent_input_policy_repo_snapshot_may_explicitly_include_harness(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      repo_root = temp_root / "repo"
      repo_root.mkdir()
      build_repo_fixture(repo_root)
      agent_data = load_json(AGENT_PATH)
      agent_data["agent_input_policy"] = [
        entry
        for entry in agent_data["agent_input_policy"]
        if entry["input_id"] != "repo_snapshot_packet"
      ]
      agent_data["agent_input_policy"].append(
        {
          "input_id": "repo_snapshot_packet",
          "required": True,
          "schema_ref": "../repo_snapshot/RepoSnapshotPacket.schema.json",
          "resolution": {
            "mode": "paths",
            "paths": ["harness/runtime/core.py"],
            "include_harness": True,
          },
        }
      )
      agent_path = temp_root / "pm_with_harness_snapshot.agent.json"
      write_json(agent_path, agent_data)

      packet = compile_agent_context_packet(
        agent_path=agent_path,
        output_path=temp_root / "agent_context_packet.json",
        manifest_path=MANIFEST_PATH,
        repo_root=repo_root,
        harness_root=REPO_ROOT,
        target_repo_root=REPO_ROOT,
        static_context_output_path=temp_root / "static_context_packet.json",
        repo_snapshot_output_path=temp_root / "repo_snapshot_packet.json",
      )

      included_paths = {
        file.path for file in packet.resolved_inputs.repo_snapshot_packet.files
      }
      self.assertIn("harness/runtime/core.py", included_paths)
      self.assertTrue(
        packet.resolved_inputs.repo_snapshot_packet.selection.include_harness
      )

  def test_agent_input_policy_without_repo_snapshot_does_not_include_it(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      agent_data = load_json(AGENT_PATH)
      agent_data["agent_input_policy"] = [
        entry
        for entry in agent_data["agent_input_policy"]
        if entry["input_id"] != "repo_snapshot_packet"
      ]
      agent_path = temp_root / "pm_without_snapshot.agent.json"
      write_json(agent_path, agent_data)
      output_path = temp_root / "agent_context_packet.json"

      packet = compile_agent_context_packet(
        agent_path=agent_path,
        output_path=output_path,
        manifest_path=MANIFEST_PATH,
        repo_root=REPO_ROOT,
        harness_root=REPO_ROOT,
        target_repo_root=REPO_ROOT,
        static_context_output_path=temp_root / "static_context_packet.json",
        repo_snapshot_output_path=temp_root / "repo_snapshot_packet.json",
      )

      self.assertIsNone(packet.resolved_inputs.repo_snapshot_packet)

  def test_required_repo_snapshot_without_resolution_fails(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      agent_data = load_json(AGENT_PATH)
      agent_data["agent_input_policy"].append(
        {
          "input_id": "repo_snapshot_packet",
          "required": True,
          "schema_ref": "../repo_snapshot/RepoSnapshotPacket.schema.json",
        }
      )
      agent_path = temp_root / "pm_missing_resolution.agent.json"
      write_json(agent_path, agent_data)

      with self.assertRaises(AgentContextCompilationError) as error:
        compile_agent_context_packet(
          agent_path=agent_path,
          output_path=temp_root / "agent_context_packet.json",
          manifest_path=MANIFEST_PATH,
          repo_root=REPO_ROOT,
          harness_root=REPO_ROOT,
          target_repo_root=REPO_ROOT,
          static_context_output_path=temp_root / "static_context_packet.json",
          repo_snapshot_output_path=temp_root / "repo_snapshot_packet.json",
        )

      self.assertEqual(error.exception.input_coverage[-1].status, "missing")

  def test_optional_repo_snapshot_without_resolution_is_recorded_missing(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      repo_root = temp_root / "repo"
      repo_root.mkdir()
      build_repo_fixture(repo_root)
      agent_data = load_json(AGENT_PATH)
      agent_data["agent_input_policy"] = [
        entry
        for entry in agent_data["agent_input_policy"]
        if entry["input_id"] != "repo_snapshot_packet"
      ]
      agent_data["agent_input_policy"].append(
        {
          "input_id": "repo_snapshot_packet",
          "required": False,
          "schema_ref": "../repo_snapshot/RepoSnapshotPacket.schema.json",
        }
      )
      agent_path = temp_root / "pm_optional_missing_resolution.agent.json"
      write_json(agent_path, agent_data)

      packet = compile_agent_context_packet(
        agent_path=agent_path,
        output_path=temp_root / "agent_context_packet.json",
        manifest_path=MANIFEST_PATH,
        repo_root=repo_root,
        harness_root=REPO_ROOT,
        target_repo_root=REPO_ROOT,
        static_context_output_path=temp_root / "static_context_packet.json",
        repo_snapshot_output_path=temp_root / "repo_snapshot_packet.json",
      )

      self.assertIsNone(packet.resolved_inputs.repo_snapshot_packet)
      self.assertEqual(packet.input_coverage[-1].status, "missing")

  def test_direct_api_call_can_attach_compiled_repo_snapshot_as_supplementary_context(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      repo_root = temp_root / "repo"
      repo_root.mkdir()
      build_repo_fixture(repo_root)
      repo_snapshot_path = temp_root / "repo_snapshot_packet.json"
      compile_repo_snapshot_packet(
        repo_root=repo_root,
        output_path=repo_snapshot_path,
        mode="paths",
        requested_paths=["README.md"],
      )
      output_path = temp_root / "api_call_packet.json"
      script_path = HARNESS_ROOT / "runtime" / "api_call_packet_builder.py"

      completed = subprocess.run(
        [
          sys.executable,
          str(script_path),
          "--task",
          "Review these files.",
          "--direct",
          "--repo-snapshot",
          str(repo_snapshot_path),
          "--output",
          str(output_path),
        ],
        cwd=script_path.parent,
        capture_output=True,
        text=True,
        check=False,
      )

      self.assertEqual(completed.returncode, 0, completed.stderr)
      emitted = load_json(output_path)
      self.assertEqual(
        emitted["supplementary_context"][0]["source_type"],
        "repo_snapshot_packet",
      )

  def test_agent_routed_api_call_rejects_duplicate_repo_snapshot_in_both_lanes(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      repo_root = temp_root / "repo"
      repo_root.mkdir()
      build_repo_fixture(repo_root)
      agent_data = load_json(AGENT_PATH)
      agent_data["agent_input_policy"] = [
        entry
        for entry in agent_data["agent_input_policy"]
        if entry["input_id"] != "repo_snapshot_packet"
      ]
      agent_data["agent_input_policy"].append(
        {
          "input_id": "repo_snapshot_packet",
          "required": True,
          "schema_ref": "../repo_snapshot/RepoSnapshotPacket.schema.json",
          "resolution": {
            "mode": "paths",
            "paths": ["README.md"],
          },
        }
      )
      agent_path = temp_root / "pm_with_snapshot.agent.json"
      write_json(agent_path, agent_data)
      agent_context_packet = compile_agent_context_packet(
        agent_path=agent_path,
        output_path=temp_root / "agent_context_packet.json",
        manifest_path=MANIFEST_PATH,
        repo_root=repo_root,
        harness_root=REPO_ROOT,
        target_repo_root=REPO_ROOT,
        static_context_output_path=temp_root / "static_context_packet.json",
        repo_snapshot_output_path=temp_root / "repo_snapshot_packet.json",
      )

      with self.assertRaises(ValueError):
        build_api_call_packet(
          task=task_from_cli("Duplicate repo snapshot."),
          call_mode="agent_routed",
          agent_context_packet=agent_context_packet,
          supplementary_context=[
            SupplementaryContextEntry(
              source_id="repo_snapshot_packet",
              source_type="repo_snapshot_packet",
              content=agent_context_packet.resolved_inputs.repo_snapshot_packet.model_dump(mode="json"),
              basis=["Duplicate fixture."],
            )
          ],
          output_path=temp_root / "api_call_packet.json",
        )

  def test_repo_snapshot_compiler_supports_direct_script_execution_without_provider_rendering(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      repo_root = temp_root / "repo"
      repo_root.mkdir()
      build_repo_fixture(repo_root)
      output_path = temp_root / "repo_snapshot_packet.json"
      script_path = HARNESS_ROOT / "repo_snapshot" / "repo_snapshot_compiler.py"
      openai_loaded_before = "openai" in sys.modules

      completed = subprocess.run(
        [
          sys.executable,
          str(script_path),
          "--repo-root",
          str(repo_root),
          "--path",
          "README.md",
          "--output",
          str(output_path),
        ],
        cwd=script_path.parent,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
      )

      self.assertEqual(completed.returncode, 0, completed.stderr)
      self.assertTrue(output_path.is_file())
      self.assertFalse((temp_root / "provider_payload.json").exists())
      self.assertEqual("openai" in sys.modules, openai_loaded_before)


if __name__ == "__main__":
  unittest.main()
