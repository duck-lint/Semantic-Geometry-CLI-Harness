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
from harness.agents.agent_context_packet import AgentContextPacket
from harness.agents.agent_runtime_registry import validate_agent_contract
from harness.agents.archive_manager import ArchiveManager
from harness.agents.project_manager_agent import ProjectManagerAgent
from harness.project_spec.static_context_packet_compiler import (
  compile_static_context_packet,
)
from harness.runtime.api_call_packet import ApiCallPacket
from harness.runtime.api_call_packet_builder import build_api_call_packet
from harness.runtime.api_call_ledger import DEFAULT_RUNTIME_CALL_LEDGER_PATH
from harness.runtime.git_context import (
  GitContext,
  GitChangedFile,
  GitDeltaContext,
  collect_git_context,
  collect_git_delta_context,
)
from harness.runtime.runtime_budget_policy import RuntimeBudgetPolicy
from harness.runtime.supplementary_context import SupplementaryContextEntry
from harness.runtime.orchestrator import build_pre_call_artifacts
from harness.runtime.task import Task, task_from_cli
from tests.agent_test_support import (
  create_test_archive_manager_agent,
  create_test_project_manager_agent,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = REPO_ROOT / "harness"
MANIFEST_PATH = HARNESS_ROOT / "project_spec" / "static_context_packet.manifest.json"
LIVE_AGENT_PATH = HARNESS_ROOT / "agents" / "project_manager.agent.json"
LIVE_AM_AGENT_PATH = HARNESS_ROOT / "agents" / "archive_manager.agent.json"
AGENT_PATH = create_test_project_manager_agent(LIVE_AGENT_PATH)
AM_AGENT_PATH = create_test_archive_manager_agent(LIVE_AM_AGENT_PATH)
AGENT_CONTEXT_SCHEMA_PATH = HARNESS_ROOT / "agents" / "AgentContextPacket.schema.json"
API_CALL_SCHEMA_PATH = HARNESS_ROOT / "runtime" / "ApiCallPacket.schema.json"
RUNTIME_BUDGET_PATH = HARNESS_ROOT / "runtime" / "runtime_budget.policy.json"


def load_json(path: Path) -> dict:
  return json.loads(path.read_text(encoding="utf-8"))


def compile_static_packet(output_path: Path):
  return compile_static_context_packet(
    manifest_path=MANIFEST_PATH,
    harness_root=REPO_ROOT,
    target_repo_root=REPO_ROOT,
    output_path=output_path,
  )


def run_git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
    ["git", *args],
    cwd=repo_root,
    capture_output=True,
    text=True,
    check=False,
  )


def run_git_checked(repo_root: Path, *args: str) -> str:
  completed = run_git(repo_root, *args)
  if completed.returncode != 0:
    raise AssertionError(completed.stderr)
  return completed.stdout.strip()


def git_commit_all(repo_root: Path, message: str) -> str:
  run_git_checked(repo_root, "add", ".")
  run_git_checked(
    repo_root,
    "-c",
    "user.name=Harness Test",
    "-c",
    "user.email=harness-test@example.test",
    "commit",
    "-m",
    message,
  )
  return run_git_checked(repo_root, "rev-parse", "HEAD")


def init_git_repo(repo_root: Path) -> None:
  run_git_checked(repo_root, "init")
  run_git_checked(repo_root, "checkout", "-b", "main")


def setUpModule() -> None:
  ledger_path = DEFAULT_RUNTIME_CALL_LEDGER_PATH
  ledger_path.parent.mkdir(parents=True, exist_ok=True)
  ledger_path.write_text("", encoding="utf-8")


def tearDownModule() -> None:
  ledger_path = DEFAULT_RUNTIME_CALL_LEDGER_PATH
  if ledger_path.exists():
    ledger_path.unlink()
  try:
    ledger_path.parent.rmdir()
  except OSError:
    pass


class PreCallPacketAssemblyTests(unittest.TestCase):
  def test_cli_task_text_validates_as_task(self) -> None:
    task = task_from_cli("Review the current proof frontier.")

    self.assertIsInstance(task, Task)
    self.assertEqual(task.task_text, "Review the current proof frontier.")
    self.assertEqual(task.source, "cli")

    with self.assertRaises(ValueError):
      task_from_cli("")

  def test_current_pm_agent_file_validates(self) -> None:
    agent = ProjectManagerAgent.model_validate(load_json(LIVE_AGENT_PATH))

    self.assertEqual(agent.metadata.id, "project_manager.agent.json")
    self.assertEqual(agent.provider, "openai")
    self.assertEqual(
      [entry.input_id for entry in agent.agent_input_policy],
      ["static_context_packet", "repo_snapshot_packet"],
    )

  def test_current_archive_manager_agent_file_validates(self) -> None:
    agent_data = load_json(LIVE_AM_AGENT_PATH)
    agent = ArchiveManager.model_validate(agent_data)
    generic_agent = validate_agent_contract(agent_data)

    self.assertEqual(agent.metadata.id, "archive_manager.agent.json")
    self.assertEqual(agent.provider, "openai")
    self.assertEqual(
      [entry.input_id for entry in agent.agent_input_policy],
      ["static_context_packet", "repo_snapshot_packet"],
    )
    self.assertEqual(agent.agent_output_policy[0].output_id, "archive_manager_report")
    self.assertEqual(generic_agent.metadata.id, agent.metadata.id)

  def test_agent_context_compiles_from_pm_input_policy(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      output_path = temp_root / "agent_context_packet.json"
      static_output_path = temp_root / "static_context_packet.json"

      packet = compile_agent_context_packet(
        agent_path=AGENT_PATH,
        output_path=output_path,
        manifest_path=MANIFEST_PATH,
        harness_root=REPO_ROOT,
        target_repo_root=REPO_ROOT,
        static_context_output_path=static_output_path,
      )

      self.assertTrue(output_path.is_file())
      self.assertTrue(static_output_path.is_file())
      self.assertIsInstance(packet, AgentContextPacket)
      self.assertEqual(
        packet.agent_contract.metadata.id,
        "project_manager.agent.json",
      )
      self.assertEqual(packet.agent_contract.provider, "openai")
      self.assertEqual(len(packet.input_coverage), 2)
      self.assertEqual(
        [entry.input_id for entry in packet.input_coverage],
        ["static_context_packet", "repo_snapshot_packet"],
      )
      self.assertTrue(all(entry.status == "included" for entry in packet.input_coverage))
      self.assertEqual(
        packet.resolved_inputs.static_context_packet.metadata.document_id,
        "static_context_packet.json",
      )
      self.assertIsNotNone(packet.resolved_inputs.repo_snapshot_packet)
      self.assertNotIn("task", packet.model_dump(mode="json"))
      self.assertNotIn("git_context", packet.model_dump(mode="json"))
      self.assertNotIn("supplementary_context", packet.model_dump(mode="json"))
      self.assertNotIn("runtime_budget", packet.model_dump(mode="json"))

  def test_agent_context_compiles_from_archive_manager_input_policy(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      output_path = temp_root / "agent_context_packet.json"
      static_output_path = temp_root / "static_context_packet.json"

      packet = compile_agent_context_packet(
        agent_path=AM_AGENT_PATH,
        output_path=output_path,
        manifest_path=MANIFEST_PATH,
        harness_root=REPO_ROOT,
        target_repo_root=REPO_ROOT,
        static_context_output_path=static_output_path,
      )

      self.assertTrue(output_path.is_file())
      self.assertTrue(static_output_path.is_file())
      self.assertIsInstance(packet, AgentContextPacket)
      self.assertEqual(packet.agent_contract.metadata.id, "archive_manager.agent.json")
      self.assertEqual(packet.agent_contract.provider, "openai")
      self.assertEqual(len(packet.input_coverage), 2)
      self.assertTrue(all(entry.status == "included" for entry in packet.input_coverage))
      self.assertIsNotNone(packet.resolved_inputs.repo_snapshot_packet)

      repo_snapshot_packet = packet.resolved_inputs.repo_snapshot_packet
      self.assertIsNotNone(repo_snapshot_packet)
      self.assertEqual(
        [file.path for file in repo_snapshot_packet.files],
        [
          "harness/contracts/archive_manager_report.example.json",
        ],
      )

  def test_agent_context_fails_for_invalid_static_context(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      output_path = Path(temp_directory) / "agent_context_packet.json"

      with self.assertRaises(AgentContextCompilationError) as error:
        compile_agent_context_packet(
          agent_path=AGENT_PATH,
          static_context_packet={},
          output_path=output_path,
        )

      self.assertFalse(output_path.exists())
      self.assertEqual(error.exception.input_coverage[0].status, "invalid")

  def test_agent_context_override_still_works(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      static_packet = compile_static_packet(
        temp_root / "custom_static_context_packet.json"
      )
      output_path = temp_root / "agent_context_packet.json"

      packet = compile_agent_context_packet(
        agent_path=AGENT_PATH,
        static_context_packet=static_packet.model_dump(mode="json"),
        static_context_override_path=temp_root / "custom_static_context_packet.json",
        output_path=output_path,
      )

      self.assertEqual(packet.input_coverage[0].status, "included")
      self.assertIn("override", packet.input_coverage[0].basis[0].lower())

  def test_agent_context_compiles_for_agent_with_no_inputs(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      agent_path = temp_root / "no_inputs.agent.json"
      agent_data = load_json(AGENT_PATH)
      agent_data["agent_input_policy"] = []
      agent_path.write_text(json.dumps(agent_data, indent=2) + "\n", encoding="utf-8")
      output_path = temp_root / "agent_context_packet.json"

      packet = compile_agent_context_packet(
        agent_path=agent_path,
        output_path=output_path,
      )

      self.assertEqual(packet.input_coverage, [])
      self.assertIsNone(packet.resolved_inputs.static_context_packet)

  def test_agent_context_fails_for_unsupported_required_input(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      agent_path = temp_root / "unsupported_required.agent.json"
      agent_data = load_json(AGENT_PATH)
      agent_data["agent_input_policy"] = [
        {
          "input_id": "archive_context",
          "required": True,
          "schema_ref": "../archive/ArchiveContext.schema.json",
        }
      ]
      agent_path.write_text(json.dumps(agent_data, indent=2) + "\n", encoding="utf-8")
      output_path = temp_root / "agent_context_packet.json"

      with self.assertRaises(AgentContextCompilationError) as error:
        compile_agent_context_packet(
          agent_path=agent_path,
          output_path=output_path,
        )

      self.assertEqual(error.exception.input_coverage[0].status, "invalid")
      self.assertIn("not supported", error.exception.input_coverage[0].basis[0])

  def test_agent_routed_api_call_packet_preserves_layer_boundaries(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      static_packet = compile_static_packet(
        temp_root / "static_context_packet.json"
      )
      agent_packet = compile_agent_context_packet(
        agent_path=AGENT_PATH,
        static_context_packet=static_packet,
        output_path=temp_root / "agent_context_packet.json",
      )
      output_path = temp_root / "api_call_packet.json"

      packet = build_api_call_packet(
        task=task_from_cli("Assemble a provider-neutral packet."),
        call_mode="agent_routed",
        agent_context_packet=agent_packet,
        git_context=GitContext(
          available=True,
          branch="main",
          commit="abc123",
          is_dirty=False,
        ),
        output_path=output_path,
      )

      dumped = packet.model_dump(mode="json")
      self.assertTrue(output_path.is_file())
      self.assertIsInstance(packet, ApiCallPacket)
      self.assertEqual(packet.call_mode, "agent_routed")
      self.assertIn("task", dumped)
      self.assertIn("agent_context_packet", dumped)
      self.assertNotIn("task", dumped["agent_context_packet"])
      self.assertIn("agent_contract", dumped["agent_context_packet"])
      self.assertIn("resolved_inputs", dumped["agent_context_packet"])
      self.assertIn(
        "static_context_packet",
        dumped["agent_context_packet"]["resolved_inputs"],
      )
      self.assertIn("git_context", dumped)
      self.assertEqual(dumped["supplementary_context"], [])
      self.assertNotIn("model", dumped)
      self.assertNotIn("messages", dumped)
      self.assertNotIn("instructions", dumped)
      self.assertIn("runtime_budget", dumped)
      self.assertIsNone(dumped["runtime_budget"])

  def test_direct_api_call_packet_builds_without_agent_context(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      output_path = Path(temp_directory) / "api_call_packet.json"

      packet = build_api_call_packet(
        task=task_from_cli("Quick direct message."),
        call_mode="direct",
        git_context=GitContext(available=False, failure="test fixture"),
        output_path=output_path,
      )

      self.assertTrue(output_path.is_file())
      self.assertEqual(packet.call_mode, "direct")
      self.assertIsNone(packet.agent_context_packet)
      self.assertEqual(packet.supplementary_context, [])

  def test_api_call_packet_may_include_runtime_budget(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      output_path = Path(temp_directory) / "api_call_packet.json"
      runtime_budget = RuntimeBudgetPolicy.model_validate(
        load_json(RUNTIME_BUDGET_PATH)
      )

      packet = build_api_call_packet(
        task=task_from_cli("Direct call with runtime budget."),
        call_mode="direct",
        runtime_budget=runtime_budget,
        output_path=output_path,
      )

      self.assertIsNotNone(packet.runtime_budget)
      self.assertEqual(
        packet.runtime_budget.default.max_input_tokens,
        runtime_budget.default.max_input_tokens,
      )

  def test_direct_packet_rejects_non_null_agent_context(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      static_packet = compile_static_packet(
        temp_root / "static_context_packet.json"
      )
      agent_packet = compile_agent_context_packet(
        agent_path=AGENT_PATH,
        static_context_packet=static_packet,
        output_path=temp_root / "agent_context_packet.json",
      )

      with self.assertRaises(ValueError):
        build_api_call_packet(
          task=task_from_cli("Direct calls must not include agent context."),
          call_mode="direct",
          agent_context_packet=agent_packet,
          output_path=temp_root / "api_call_packet.json",
        )

  def test_agent_routed_packet_rejects_missing_agent_context(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      with self.assertRaises(ValueError):
        build_api_call_packet(
          task=task_from_cli("Agent-routed calls require agent context."),
          call_mode="agent_routed",
          output_path=Path(temp_directory) / "api_call_packet.json",
        )

  def test_direct_packet_may_attach_static_context_as_supplementary(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      static_packet = compile_static_packet(
        temp_root / "static_context_packet.json"
      )
      output_path = temp_root / "api_call_packet.json"

      packet = build_api_call_packet(
        task=task_from_cli("Direct call with explicit project context."),
        call_mode="direct",
        supplementary_context=[
          SupplementaryContextEntry(
            source_id="static_context_packet",
            source_type="static_context_packet",
            content=static_packet.model_dump(mode="json"),
            basis=["Explicitly attached compiled StaticContextPacket."],
          )
        ],
        output_path=output_path,
      )

      self.assertEqual(packet.call_mode, "direct")
      self.assertIsNone(packet.agent_context_packet)
      self.assertEqual(len(packet.supplementary_context), 1)
      self.assertEqual(
        packet.supplementary_context[0].source_type,
        "static_context_packet",
      )

  def test_static_context_is_not_duplicated_across_lanes(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      static_packet = compile_static_packet(
        temp_root / "static_context_packet.json"
      )
      agent_packet = compile_agent_context_packet(
        agent_path=AGENT_PATH,
        static_context_packet=static_packet,
        output_path=temp_root / "agent_context_packet.json",
      )

      packet = build_api_call_packet(
        task=task_from_cli("Avoid duplicate static context lanes."),
        call_mode="agent_routed",
        agent_context_packet=agent_packet,
        output_path=temp_root / "api_call_packet.json",
      )

      self.assertEqual(packet.supplementary_context, [])
      self.assertIsNotNone(packet.agent_context_packet)
      self.assertIsNotNone(
        packet.agent_context_packet.resolved_inputs.static_context_packet
      )

  def test_packet_rejects_duplicate_static_context_across_lanes(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      static_packet = compile_static_packet(
        temp_root / "static_context_packet.json"
      )
      agent_packet = compile_agent_context_packet(
        agent_path=AGENT_PATH,
        static_context_packet=static_packet,
        output_path=temp_root / "agent_context_packet.json",
      )

      with self.assertRaises(ValueError):
        build_api_call_packet(
          task=task_from_cli("Do not duplicate static context."),
          call_mode="agent_routed",
          agent_context_packet=agent_packet,
          supplementary_context=[
            SupplementaryContextEntry(
              source_id="static_context_packet",
              source_type="static_context_packet",
              content=static_packet.model_dump(mode="json"),
              basis=["Duplicate fixture."],
            )
          ],
          output_path=temp_root / "api_call_packet.json",
        )

  def test_git_context_is_non_fatal_outside_a_repository(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      context = collect_git_context(Path(temp_directory))

      self.assertFalse(context.available)
      self.assertIsNotNone(context.failure)

  def test_git_delta_context_collects_commits_files_stat_and_worktree(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      repo_root = Path(temp_directory)
      init_git_repo(repo_root)
      (repo_root / "README.md").write_text("base\n", encoding="utf-8")
      base_commit = git_commit_all(repo_root, "Base commit")
      (repo_root / "README.md").write_text("base\nnext\n", encoding="utf-8")
      git_commit_all(repo_root, "Add next line")
      (repo_root / "dirty.txt").write_text("dirty\n", encoding="utf-8")

      context = collect_git_delta_context(repo_root, base_commit)

      self.assertTrue(context.available, context.failure)
      self.assertEqual(context.base_commit, base_commit)
      self.assertEqual(context.head_commit, run_git_checked(repo_root, "rev-parse", "HEAD"))
      self.assertEqual(context.comparison_range, f"{base_commit}..{context.head_commit}")
      self.assertTrue(context.base_is_ancestor_of_head)
      self.assertEqual(len(context.commits_since_base), 1)
      self.assertEqual(context.commits_since_base[0].summary, "Add next line")
      self.assertEqual(
        context.changed_files_since_base,
        [GitChangedFile(status="M", path="README.md")],
      )
      self.assertIn("README.md", context.diff_stat_since_base or "")
      self.assertEqual(context.worktree_state, "dirty")
      self.assertIn("?? dirty.txt", context.uncommitted_status_summary)

  def test_git_delta_context_reports_invalid_base_as_unavailable(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      repo_root = Path(temp_directory)
      init_git_repo(repo_root)
      (repo_root / "README.md").write_text("base\n", encoding="utf-8")
      git_commit_all(repo_root, "Base commit")

      context = collect_git_delta_context(repo_root, "not-a-commit")

      self.assertFalse(context.available)
      self.assertIsNotNone(context.failure)

  def test_git_delta_context_marks_non_ancestor_as_lineage_limitation(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      repo_root = Path(temp_directory)
      init_git_repo(repo_root)
      (repo_root / "README.md").write_text("base\n", encoding="utf-8")
      git_commit_all(repo_root, "Base commit")
      run_git_checked(repo_root, "checkout", "-b", "side")
      (repo_root / "side.txt").write_text("side\n", encoding="utf-8")
      side_commit = git_commit_all(repo_root, "Side commit")
      run_git_checked(repo_root, "checkout", "main")
      (repo_root / "main.txt").write_text("main\n", encoding="utf-8")
      git_commit_all(repo_root, "Main commit")

      context = collect_git_delta_context(repo_root, side_commit)

      self.assertTrue(context.available, context.failure)
      self.assertFalse(context.base_is_ancestor_of_head)
      self.assertIsNotNone(context.lineage_limitation)
      self.assertIsNone(context.failure)

  def test_git_delta_context_preserves_rename_path_information(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      repo_root = Path(temp_directory)
      init_git_repo(repo_root)
      (repo_root / "old.txt").write_text("base\n", encoding="utf-8")
      base_commit = git_commit_all(repo_root, "Base commit")
      run_git_checked(repo_root, "mv", "old.txt", "new.txt")
      git_commit_all(repo_root, "Rename file")

      context = collect_git_delta_context(repo_root, base_commit)

      self.assertTrue(context.available, context.failure)
      self.assertEqual(len(context.changed_files_since_base), 1)
      self.assertTrue(context.changed_files_since_base[0].status.startswith("R"))
      self.assertIn("old.txt", context.changed_files_since_base[0].path)
      self.assertIn("new.txt", context.changed_files_since_base[0].path)

  def test_packet_schemas_validate_emitted_artifacts(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      static_packet = compile_static_packet(
        temp_root / "static_context_packet.json"
      )
      agent_path = temp_root / "agent_context_packet.json"
      agent_packet = compile_agent_context_packet(
        agent_path=AGENT_PATH,
        output_path=agent_path,
        manifest_path=MANIFEST_PATH,
        harness_root=REPO_ROOT,
        target_repo_root=REPO_ROOT,
        static_context_output_path=temp_root / "static_context_packet.compiled.json",
      )
      api_path = temp_root / "api_call_packet.json"
      build_api_call_packet(
        task=task_from_cli("Validate generated packet schemas."),
        call_mode="agent_routed",
        agent_context_packet=agent_packet,
        git_context=GitContext(available=False, failure="test fixture"),
        supplementary_context=None,
        output_path=api_path,
      )

      agent_schema = load_json(AGENT_CONTEXT_SCHEMA_PATH)
      api_schema = load_json(API_CALL_SCHEMA_PATH)
      Draft202012Validator.check_schema(agent_schema)
      Draft202012Validator.check_schema(api_schema)
      Draft202012Validator(agent_schema).validate(load_json(agent_path))
      Draft202012Validator(api_schema).validate(load_json(api_path))
      self.assertNotIn("static_context_packet", agent_schema["properties"])

  def test_pre_call_route_stops_before_provider_or_model_call(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      runs_root = Path(temp_directory) / "runs"
      openai_loaded_before = "openai" in sys.modules

      run_directory, artifact_paths = build_pre_call_artifacts(
        task_text="Build the bounded pre-call packet.",
        repo_root=REPO_ROOT,
        harness_root=REPO_ROOT,
        target_repo_root=REPO_ROOT,
        manifest_path=MANIFEST_PATH,
        runs_root=runs_root,
        agent_path=AGENT_PATH,
      )

      self.assertEqual(
        {path.name for path in artifact_paths},
        {
          "task.json",
          "static_context_packet.json",
          "repo_snapshot_packet.json",
          "agent_context_packet.json",
          "api_call_packet.json",
        },
      )
      self.assertEqual(
        {path.name for path in run_directory.iterdir()},
        {
          "task.json",
          "static_context_packet.json",
          "repo_snapshot_packet.json",
          "agent_context_packet.json",
          "api_call_packet.json",
        },
      )
      self.assertEqual("openai" in sys.modules, openai_loaded_before)
      self.assertNotIn("provider_payload.json", {path.name for path in artifact_paths})
      self.assertNotIn(
        "project_manager_report.json",
        {path.name for path in artifact_paths},
      )

  def test_agent_context_compiler_supports_direct_script_execution(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      output_path = temp_root / "agent_context_packet.json"
      static_path = temp_root / "static_context_packet.json"
      script_path = HARNESS_ROOT / "agents" / "agent_context_compiler.py"

      completed = subprocess.run(
        [
          sys.executable,
          str(script_path),
          "--agent",
          str(AGENT_PATH),
          "--manifest",
          str(MANIFEST_PATH),
          "--harness-root",
          str(REPO_ROOT),
          "--target-repo-root",
          str(REPO_ROOT),
          "--output",
          str(output_path),
        ],
        cwd=script_path.parent,
        capture_output=True,
        text=True,
        check=False,
      )

      self.assertEqual(completed.returncode, 0, completed.stderr)
      self.assertTrue(output_path.is_file())
      self.assertTrue(static_path.is_file())
      self.assertIn("PASS: Agent context packet written", completed.stdout)
      emitted = load_json(output_path)
      self.assertIsNotNone(
        emitted["resolved_inputs"]["static_context_packet"]
      )

  def test_agent_context_compiler_supports_direct_script_override(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      static_path = temp_root / "custom_static_context_packet.json"
      compile_static_packet(static_path)
      output_path = temp_root / "agent_context_packet.json"
      script_path = HARNESS_ROOT / "agents" / "agent_context_compiler.py"

      completed = subprocess.run(
        [
          sys.executable,
          str(script_path),
          "--agent",
          str(AGENT_PATH),
          "--static-context",
          str(static_path),
          "--output",
          str(output_path),
        ],
        cwd=script_path.parent,
        capture_output=True,
        text=True,
        check=False,
      )

      self.assertEqual(completed.returncode, 0, completed.stderr)
      self.assertTrue(output_path.is_file())

  def test_api_call_builder_supports_direct_script_execution_for_direct_mode(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      static_path = temp_root / "static_context_packet.json"
      compile_static_packet(static_path)
      output_path = temp_root / "api_call_packet.json"
      script_path = HARNESS_ROOT / "runtime" / "api_call_packet_builder.py"

      completed = subprocess.run(
        [
          sys.executable,
          str(script_path),
          "--task",
          "Review the current project trajectory.",
          "--direct",
          "--static-context",
          str(static_path),
          "--runtime-budget",
          str(RUNTIME_BUDGET_PATH),
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
      self.assertEqual(emitted["call_mode"], "direct")
      self.assertIn("runtime_budget", emitted)
      self.assertIsNone(emitted["agent_context_packet"])
      self.assertEqual(
        emitted["git_context"],
        collect_git_context(REPO_ROOT).model_dump(mode="json"),
      )
      self.assertEqual(
        emitted["supplementary_context"][0]["source_type"],
        "static_context_packet",
      )

  def test_api_call_builder_supports_no_git_context_flag(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      output_path = temp_root / "api_call_packet.json"
      script_path = HARNESS_ROOT / "runtime" / "api_call_packet_builder.py"

      completed = subprocess.run(
        [
          sys.executable,
          str(script_path),
          "--task",
          "Review the current project trajectory.",
          "--direct",
          "--no-git-context",
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
      self.assertIsNone(emitted["git_context"])
      self.assertIn("Git context: disabled.", completed.stdout)

  def test_api_call_builder_supports_direct_script_execution_for_agent_route(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      output_path = temp_root / "api_call_packet.json"
      script_path = HARNESS_ROOT / "runtime" / "api_call_packet_builder.py"
      agent_context_path = temp_root / "agent_context_packet.json"
      static_path = temp_root / "static_context_packet.json"

      completed = subprocess.run(
        [
          sys.executable,
          str(script_path),
          "--task",
          "Review this against project context.",
          "--agent",
          str(AGENT_PATH),
          "--manifest",
          str(MANIFEST_PATH),
          "--harness-root",
          str(REPO_ROOT),
          "--target-repo-root",
          str(REPO_ROOT),
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
      self.assertTrue(agent_context_path.is_file())
      self.assertTrue(static_path.is_file())
      self.assertEqual(emitted["call_mode"], "agent_routed")
      self.assertEqual(
        emitted["git_context"],
        collect_git_context(REPO_ROOT).model_dump(mode="json"),
      )
      self.assertIn("agent_context_packet", emitted)
      self.assertIn(
        "static_context_packet",
        emitted["agent_context_packet"]["resolved_inputs"],
      )

  def test_api_call_builder_rejects_static_context_flag_for_agent_route(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      static_path = temp_root / "static_context_packet.json"
      compile_static_packet(static_path)
      output_path = temp_root / "api_call_packet.json"
      script_path = HARNESS_ROOT / "runtime" / "api_call_packet_builder.py"

      completed = subprocess.run(
        [
          sys.executable,
          str(script_path),
          "--task",
          "Review this against project context.",
          "--agent",
          str(AGENT_PATH),
          "--static-context",
          str(static_path),
          "--output",
          str(output_path),
        ],
        cwd=script_path.parent,
        capture_output=True,
        text=True,
        check=False,
      )

      self.assertEqual(completed.returncode, 1)
      self.assertIn("only allowed for direct calls", completed.stderr)
      self.assertFalse(output_path.exists())

  def test_api_call_packet_may_include_git_delta_context(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      output_path = Path(temp_directory) / "api_call_packet.json"

      packet = build_api_call_packet(
        task=task_from_cli("Review the delta context."),
        call_mode="direct",
        git_delta_context=GitDeltaContext(
          available=True,
          base_commit="abc123",
          head_commit="def456",
          comparison_range="abc123..def456",
          base_is_ancestor_of_head=True,
          worktree_state="clean",
        ),
        output_path=output_path,
      )

      emitted = load_json(output_path)
      self.assertIsNone(packet.git_context)
      self.assertIsNotNone(packet.git_delta_context)
      self.assertIsNone(emitted["git_context"])
      self.assertEqual(emitted["git_delta_context"]["base_commit"], "abc123")

  def test_api_call_packet_rejects_both_git_context_lanes(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      with self.assertRaises(ValueError):
        build_api_call_packet(
          task=task_from_cli("Reject ambiguous git context."),
          call_mode="direct",
          git_context=GitContext(available=True, commit="abc123"),
          git_delta_context=GitDeltaContext(
            available=True,
            base_commit="abc123",
            head_commit="def456",
            comparison_range="abc123..def456",
          ),
          output_path=Path(temp_directory) / "api_call_packet.json",
        )

  def test_api_call_builder_refuses_to_write_packet_without_task(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      output_path = Path(temp_directory) / "api_call_packet.json"
      script_path = HARNESS_ROOT / "runtime" / "api_call_packet_builder.py"

      completed = subprocess.run(
        [
          sys.executable,
          str(script_path),
          "--direct",
          "--output",
          str(output_path),
        ],
        cwd=script_path.parent,
        capture_output=True,
        text=True,
        check=False,
      )

      self.assertEqual(completed.returncode, 1)
      self.assertIn("ApiCallPacket requires a task", completed.stderr)
      self.assertFalse(output_path.exists())


if __name__ == "__main__":
  unittest.main()
