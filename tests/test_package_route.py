from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from harness.providers.openai.openai_call_runner import OpenAICallRunnerError
from harness.providers.openai.openai_response_payload_compiler import (
  OpenAIResponsePayloadCompilationError,
)
from harness.contracts.project_manager_report_extractor import (
  ProjectManagerReportExtractorError,
)
from harness.contracts.project_manager_report_validation import (
  ProjectManagerReportValidationArtifact,
  default_validation_artifact_path,
)
from harness.runtime.artifact_facts import sha256_file
from harness.runtime.api_call_ledger import DEFAULT_RUNTIME_CALL_LEDGER_PATH
from harness.runtime import package_route
from tests.agent_test_support import create_test_project_manager_agent


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = REPO_ROOT / "harness"
LIVE_AGENT_PATH = HARNESS_ROOT / "agents" / "project_manager.agent.json"
AGENT_PATH = create_test_project_manager_agent(LIVE_AGENT_PATH)
RAW_RESPONSE_FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "raw_model_response.json"
RUNTIME_BUDGET_PATH = HARNESS_ROOT / "runtime" / "runtime_budget.policy.json"
LEDGER_PATH = DEFAULT_RUNTIME_CALL_LEDGER_PATH


def load_json(path: Path) -> dict:
  return json.loads(path.read_text(encoding="utf-8"))


def write_fake_openai_module(fake_root: Path, raw_response_artifact: dict) -> None:
  fake_root.mkdir(parents=True, exist_ok=True)
  module_text = textwrap.dedent(
    f"""
    import json

    RAW_RESPONSE_ARTIFACT = json.loads({json.dumps(json.dumps(raw_response_artifact))})

    class _Response:
      id = RAW_RESPONSE_ARTIFACT["response_id"]
      model = RAW_RESPONSE_ARTIFACT["model"]
      status = RAW_RESPONSE_ARTIFACT["status"]
      output_text = RAW_RESPONSE_ARTIFACT["output_text"]

      def model_dump(self, mode="json"):
        return dict(RAW_RESPONSE_ARTIFACT["raw_response"])

    class _Responses:
      def create(self, **kwargs):
        _ = kwargs
        return _Response()

    class OpenAI:
      def __init__(self):
        self.responses = _Responses()
    """
  )
  (fake_root / "openai.py").write_text(module_text, encoding="utf-8")


def only_run_directory(runs_root: Path) -> Path:
  run_directories = [path for path in runs_root.iterdir() if path.is_dir()]
  if len(run_directories) != 1:
    raise AssertionError(f"Expected one run directory, got {run_directories}.")
  return run_directories[0]


def display_path(path: Path) -> str:
  try:
    return path.relative_to(REPO_ROOT).as_posix()
  except ValueError:
    return path.as_posix()


def remove_ledger_artifact() -> None:
  if LEDGER_PATH.exists():
    LEDGER_PATH.unlink()

  ledger_parent = LEDGER_PATH.parent
  if ledger_parent.exists():
    try:
      ledger_parent.rmdir()
    except OSError:
      pass


def ensure_ledger_artifact() -> None:
  LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
  LEDGER_PATH.write_text("", encoding="utf-8")


class PackageRouteTests(unittest.TestCase):
  def _assert_successful_route(
    self,
    *,
    command: list[str],
    expected_banner: str,
    expected_route: str,
    expected_repo_snapshot_paths: list[str] | None = None,
    agent_path: Path = AGENT_PATH,
  ) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      runs_root = temp_root / "runs"
      fake_openai_root = temp_root / "fake_openai"
      write_fake_openai_module(fake_openai_root, load_json(RAW_RESPONSE_FIXTURE_PATH))
      source_paths = [
        REPO_ROOT / "README.md",
        HARNESS_ROOT / "__main__.py",
        agent_path,
      ]
      source_snapshots = {path: path.read_bytes() for path in source_paths}
      remove_ledger_artifact()
      ensure_ledger_artifact()
      try:
        completed = subprocess.run(
          [*command, "--runs-root", str(runs_root)],
          cwd=REPO_ROOT,
          capture_output=True,
          text=True,
          check=False,
          env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(fake_openai_root),
          },
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(expected_banner, completed.stdout)
        self.assertIn(f"Selected agent: {display_path(agent_path)}", completed.stdout)
        self.assertIn("Provider: openai", completed.stdout)
        self.assertIn(f"Model: {load_json(agent_path)['model']}", completed.stdout)
        self.assertIn("Run directory:", completed.stdout)
        self.assertIn("Report status: needs_clarification", completed.stdout)
        self.assertIn("Blocked: True", completed.stdout)

        run_directory = only_run_directory(runs_root)
        self.assertIn("agent-route", run_directory.name)
        self.assertTrue(runs_root.exists())
        self.assertEqual(len([path for path in runs_root.iterdir() if path.is_dir()]), 1)
        self.assertEqual(
          {path.name for path in run_directory.iterdir()},
          {
            "task.json",
            "static_context_packet.json",
            "repo_snapshot_packet.json",
            "agent_context_packet.json",
            "api_call_packet.json",
            "provider_payload.json",
            "raw_model_response.json",
            "project_manager_report.json",
            "project_manager_report.validation.json",
          },
        )
        self.assertFalse(
          any(
            forbidden in name
            for name in {path.name for path in run_directory.iterdir()}
            for forbidden in ("planner", "reviewer", "worker")
          )
        )

        api_call_packet = load_json(run_directory / "api_call_packet.json")
        provider_payload = load_json(run_directory / "provider_payload.json")
        raw_response = load_json(run_directory / "raw_model_response.json")
        report = load_json(run_directory / "project_manager_report.json")
        validation_path = default_validation_artifact_path(
          run_directory / "project_manager_report.json"
        )

        self.assertEqual(api_call_packet["runtime_budget"], load_json(RUNTIME_BUDGET_PATH))
        self.assertEqual(provider_payload["request"]["model"], load_json(AGENT_PATH)["model"])
        self.assertEqual(
          provider_payload["request"]["max_output_tokens"],
          load_json(RUNTIME_BUDGET_PATH)["default"]["reserved_output_tokens"],
        )
        self.assertEqual(
          provider_payload["request"]["text"]["format"]["name"],
          "project_manager_report",
        )
        self.assertEqual(raw_response["provider"], "openai")
        self.assertEqual(report["report_status"], "needs_clarification")
        self.assertTrue(report["proof_frontier"]["blocked"])
        self.assertTrue(report["report_source_coverage"]["repo_snapshot_packet"]["consumed"])
        self.assertTrue(report["report_source_coverage"]["git_context"]["consumed"])
        self.assertTrue(
          any(
            "harness/runs/20260612-214948-agent-route/project_manager_report.json"
            in basis
            for basis in report["report_source_coverage"]["repo_snapshot_packet"]["basis"]
          )
        )
        self.assertTrue(validation_path.exists())
        validation_artifact = load_json(validation_path)
        ProjectManagerReportValidationArtifact.model_validate(validation_artifact)
        self.assertEqual(
          validation_artifact["report_artifact_path"],
          (run_directory / "project_manager_report.json").resolve().as_posix(),
        )
        self.assertEqual(
          validation_artifact["report_artifact_sha256"],
          sha256_file(run_directory / "project_manager_report.json"),
        )
        self.assertEqual(
          validation_artifact["schema_name"],
          provider_payload["request"]["text"]["format"]["name"],
        )
        self.assertEqual(
          validation_artifact["schema_path"],
          (HARNESS_ROOT / "contracts" / "ProjectManagerReport.schema.json").as_posix(),
        )
        self.assertEqual(
          validation_artifact["schema_sha256"],
          sha256_file(HARNESS_ROOT / "contracts" / "ProjectManagerReport.schema.json"),
        )
        self.assertEqual(validation_artifact["report_status"], "needs_clarification")
        self.assertTrue(validation_artifact["proof_frontier_blocked"])

        self.assertTrue(LEDGER_PATH.exists())
        ledger_lines = LEDGER_PATH.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(ledger_lines), 1)
        ledger_record = json.loads(ledger_lines[0])
        self.assertEqual(ledger_record["route"], expected_route)
        self.assertEqual(ledger_record["agent"], display_path(agent_path))
        self.assertEqual(ledger_record["model"], load_json(agent_path)["model"])
        self.assertEqual(ledger_record["schema_name"], "project_manager_report")
        self.assertTrue(ledger_record["validation_passed"])
        self.assertEqual(ledger_record["contract_status"], "needs_clarification")
        self.assertEqual(
          ledger_record["output_artifact_path"],
          display_path(run_directory / "project_manager_report.json"),
        )
        self.assertEqual(
          ledger_record["report_artifact_path"],
          display_path(run_directory / "project_manager_report.json"),
        )
        self.assertEqual(
          ledger_record["report_artifact_sha256"],
          sha256_file(run_directory / "project_manager_report.json"),
        )
        self.assertEqual(
          ledger_record["validation_artifact_path"],
          display_path(validation_path),
        )
        self.assertEqual(
          ledger_record["validation_artifact_sha256"],
          sha256_file(validation_path),
        )
        self.assertNotIn("actual_input_tokens", ledger_record)
        self.assertNotIn("actual_output_tokens", ledger_record)
        self.assertNotIn("total_tokens", ledger_record)

        if expected_repo_snapshot_paths is not None:
          repo_snapshot_packet = load_json(run_directory / "repo_snapshot_packet.json")
          expected_repo_snapshot_paths = [
            path.replace("<run_directory>", run_directory.name)
            for path in expected_repo_snapshot_paths
          ]
          self.assertEqual(
            [file["path"] for file in repo_snapshot_packet["files"]],
            expected_repo_snapshot_paths,
          )
          self.assertEqual(
            repo_snapshot_packet["selection"]["requested_paths"],
            expected_repo_snapshot_paths,
          )
          self.assertTrue(
            repo_snapshot_packet["selection"][
              "explicit_path_overrides_default_exclusions"
            ]
          )
          self.assertTrue(
            all(file["explicit_requested_path"] for file in repo_snapshot_packet["files"])
          )
        else:
          repo_snapshot_packet = load_json(run_directory / "repo_snapshot_packet.json")
          resolution = load_json(agent_path)["agent_input_policy"][1]["resolution"]
          self.assertEqual(
            repo_snapshot_packet["selection"]["mode"],
            resolution["mode"],
          )
          self.assertEqual(
            repo_snapshot_packet["selection"]["include_harness"],
            resolution["include_harness"],
          )

        for path in source_paths:
          self.assertEqual(path.read_bytes(), source_snapshots[path])
      finally:
        remove_ledger_artifact()

  def test_package_cli_runs_plan_route(self) -> None:
    self._assert_successful_route(
      command=[
        sys.executable,
        "-m",
        "harness",
        "plan",
        "Review the current project trajectory.",
        "--agent",
        str(AGENT_PATH),
      ],
      expected_banner="PASS: Plan route completed.",
      expected_route="plan",
    )

  def test_package_cli_runs_generic_agent_route(self) -> None:
    self._assert_successful_route(
      command=[
        sys.executable,
        "-m",
        "harness",
        "--agent",
        str(AGENT_PATH),
        "Review the current project trajectory.",
      ],
      expected_banner="PASS: Agent route completed.",
      expected_route="agent",
    )

  def test_package_cli_runs_non_pm_agent_route(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      reviewer_agent_path = temp_root / "reviewer.agent.json"
      reviewer_agent_data = load_json(AGENT_PATH)
      reviewer_agent_data["metadata"]["id"] = "reviewer.agent.json"
      reviewer_agent_data["metadata"]["agent_name"] = "reviewer"
      reviewer_agent_path.write_text(
        json.dumps(reviewer_agent_data, indent=2) + "\n",
        encoding="utf-8",
      )

      self._assert_successful_route(
        command=[
          sys.executable,
          "-m",
          "harness",
          "--agent",
          str(reviewer_agent_path),
          "Review the current project trajectory.",
        ],
        expected_banner="PASS: Agent route completed.",
        expected_route="agent",
        agent_path=reviewer_agent_path,
      )

  def test_package_route_stops_before_provider_render_when_explicit_snapshot_path_is_missing(
    self,
  ) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      runs_root = temp_root / "runs"
      missing_agent_path = temp_root / "reviewer_missing_snapshot.agent.json"
      agent_data = load_json(AGENT_PATH)
      agent_data["metadata"]["id"] = "reviewer_missing_snapshot.agent.json"
      agent_data["metadata"]["agent_name"] = "reviewer"
      agent_data["agent_input_policy"][1]["resolution"] = {
        "mode": "paths",
        "paths": ["missing.txt"],
        "include_harness": False,
      }
      missing_agent_path.write_text(
        json.dumps(agent_data, indent=2) + "\n",
        encoding="utf-8",
      )
      stderr = io.StringIO()

      with patch(
        "harness.runtime.package_route.compile_openai_response_payload"
      ) as render_provider_payload:
        with redirect_stderr(stderr):
          code = package_route.main(
            [
              "Review the current project trajectory.",
              "--agent",
              str(missing_agent_path),
              "--runs-root",
              str(runs_root),
            ]
          )

      self.assertEqual(code, 1)
      self.assertIn("FAIL: compile_agent_context_packet:", stderr.getvalue())
      self.assertIn("repo_snapshot_packet (required, invalid)", stderr.getvalue())
      self.assertIn("Requested path missing.txt was not found.", stderr.getvalue())
      render_provider_payload.assert_not_called()

      run_directory = only_run_directory(runs_root)
      artifact_names = {path.name for path in run_directory.iterdir()}
      self.assertIn("task.json", artifact_names)
      self.assertIn("static_context_packet.json", artifact_names)
      self.assertNotIn("repo_snapshot_packet.json", artifact_names)
      self.assertNotIn("agent_context_packet.json", artifact_names)
      self.assertNotIn("project_manager_report.validation.json", artifact_names)
      self.assertNotIn("provider_payload.json", artifact_names)
      self.assertNotIn("raw_model_response.json", artifact_names)
      self.assertNotIn("project_manager_report.json", artifact_names)

  def test_package_route_requires_task_text(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      stderr = io.StringIO()
      with redirect_stderr(stderr):
        code = package_route.main(
          [
            "--agent",
            str(AGENT_PATH),
            "--runs-root",
            str(Path(temp_directory) / "runs"),
          ]
        )

      self.assertEqual(code, 1)
      self.assertIn("package CLI requires task text", stderr.getvalue())

  def test_package_route_requires_agent(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      stderr = io.StringIO()
      with redirect_stderr(stderr):
        code = package_route.main(
          [
            "Review the current project trajectory.",
            "--runs-root",
            str(Path(temp_directory) / "runs"),
          ]
        )

      self.assertEqual(code, 1)
      self.assertIn("package CLI requires --agent", stderr.getvalue())

  def test_package_route_rejects_non_openai_provider(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      runs_root = temp_root / "runs"
      anthropic_agent_path = temp_root / "anthropic.agent.json"
      agent_data = load_json(AGENT_PATH)
      agent_data["provider"] = "anthropic"
      anthropic_agent_path.write_text(
        json.dumps(agent_data, indent=2) + "\n",
        encoding="utf-8",
      )
      stderr = io.StringIO()
      ensure_ledger_artifact()
      self.addCleanup(remove_ledger_artifact)

      with redirect_stderr(stderr):
        code = package_route.main(
          [
            "Review the current project trajectory.",
            "--agent",
            str(anthropic_agent_path),
            "--runs-root",
            str(runs_root),
          ]
        )

      self.assertEqual(code, 1)
      self.assertIn(
        "provider 'anthropic' is declared by the selected agent, but no provider runner is implemented for it.",
        stderr.getvalue(),
      )

      run_directory = only_run_directory(runs_root)
      artifact_names = {path.name for path in run_directory.iterdir()}
      self.assertIn("task.json", artifact_names)
      self.assertIn("static_context_packet.json", artifact_names)
      self.assertIn("repo_snapshot_packet.json", artifact_names)
      self.assertIn("agent_context_packet.json", artifact_names)
      self.assertIn("api_call_packet.json", artifact_names)
      self.assertNotIn("project_manager_report.validation.json", artifact_names)
      self.assertNotIn("provider_payload.json", artifact_names)
      self.assertNotIn("raw_model_response.json", artifact_names)
      self.assertNotIn("project_manager_report.json", artifact_names)

  def test_package_route_stops_on_payload_render_failure(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      runs_root = temp_root / "runs"
      stderr = io.StringIO()
      ensure_ledger_artifact()
      self.addCleanup(remove_ledger_artifact)

      with patch(
        "harness.runtime.package_route.compile_openai_response_payload",
        side_effect=OpenAIResponsePayloadCompilationError("boom"),
      ):
        with redirect_stderr(stderr):
          code = package_route.main(
            [
              "Review the current project trajectory.",
              "--agent",
              str(AGENT_PATH),
              "--runs-root",
              str(runs_root),
            ]
          )

      self.assertEqual(code, 1)
      self.assertIn("FAIL: render_provider_payload: boom", stderr.getvalue())

      run_directory = only_run_directory(runs_root)
      artifact_names = {path.name for path in run_directory.iterdir()}
      self.assertIn("task.json", artifact_names)
      self.assertIn("static_context_packet.json", artifact_names)
      self.assertIn("repo_snapshot_packet.json", artifact_names)
      self.assertIn("agent_context_packet.json", artifact_names)
      self.assertIn("api_call_packet.json", artifact_names)
      self.assertNotIn("project_manager_report.validation.json", artifact_names)
      self.assertNotIn("provider_payload.json", artifact_names)
      self.assertNotIn("raw_model_response.json", artifact_names)
      self.assertNotIn("project_manager_report.json", artifact_names)

  def test_package_route_stops_on_runner_failure(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      runs_root = temp_root / "runs"
      stderr = io.StringIO()
      ensure_ledger_artifact()
      self.addCleanup(remove_ledger_artifact)

      with patch(
        "harness.runtime.package_route.run_openai_call",
        side_effect=OpenAICallRunnerError("boom"),
      ):
        with redirect_stderr(stderr):
          code = package_route.main(
            [
              "Review the current project trajectory.",
              "--agent",
              str(AGENT_PATH),
              "--runs-root",
              str(runs_root),
            ]
          )

      self.assertEqual(code, 1)
      self.assertIn("FAIL: run_provider: boom", stderr.getvalue())

      run_directory = only_run_directory(runs_root)
      artifact_names = {path.name for path in run_directory.iterdir()}
      self.assertIn("provider_payload.json", artifact_names)
      self.assertNotIn("raw_model_response.json", artifact_names)
      self.assertNotIn("project_manager_report.json", artifact_names)
      self.assertNotIn("project_manager_report.validation.json", artifact_names)

  def test_package_route_stops_on_extraction_failure(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      runs_root = temp_root / "runs"
      stderr = io.StringIO()
      ensure_ledger_artifact()
      self.addCleanup(remove_ledger_artifact)

      def fake_run_openai_call(
        *,
        provider_payload_path: Path,
        output_path: Path,
        payload: object | None = None,
      ):
        _ = provider_payload_path
        _ = payload
        output_path.write_text(
          json.dumps(load_json(RAW_RESPONSE_FIXTURE_PATH), indent=2) + "\n",
          encoding="utf-8",
        )
        return None

      with patch(
        "harness.runtime.package_route.run_openai_call",
        side_effect=fake_run_openai_call,
      ):
        with patch(
          "harness.runtime.package_route.extract_project_manager_report",
          side_effect=ProjectManagerReportExtractorError("boom"),
        ):
          with redirect_stderr(stderr):
            code = package_route.main(
              [
                "Review the current project trajectory.",
                "--agent",
                str(AGENT_PATH),
                "--runs-root",
                str(runs_root),
              ]
            )

      self.assertEqual(code, 1)
      self.assertIn("FAIL: extract_project_manager_report: boom", stderr.getvalue())

      run_directory = only_run_directory(runs_root)
      artifact_names = {path.name for path in run_directory.iterdir()}
      self.assertIn("raw_model_response.json", artifact_names)
      self.assertNotIn("project_manager_report.json", artifact_names)
      self.assertNotIn("project_manager_report.validation.json", artifact_names)


if __name__ == "__main__":
  unittest.main()
