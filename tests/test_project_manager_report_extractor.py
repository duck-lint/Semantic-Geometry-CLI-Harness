from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from harness.contracts.project_manager_report_extractor import (
  ProjectManagerReportExtractorError,
  extract_project_manager_report,
)
from harness.contracts.project_manager_report_validation import (
  ProjectManagerReportValidationArtifact,
  default_validation_artifact_path,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = REPO_ROOT / "harness"
FIXTURES_ROOT = REPO_ROOT / "tests" / "fixtures"
RAW_RESPONSE_FIXTURE_PATH = FIXTURES_ROOT / "raw_model_response.json"
REJECTED_UNBLOCKED_RAW_RESPONSE_FIXTURE_PATH = (
  FIXTURES_ROOT / "raw_model_response_rejected_unblocked.json"
)
PM_SCHEMA_PATH = HARNESS_ROOT / "contracts" / "ProjectManagerReport.schema.json"


def load_json(path: Path):
  return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


class ProjectManagerReportExtractorTests(unittest.TestCase):
  def test_extractor_reads_valid_raw_model_response_fixture(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      output_path = temp_root / "project_manager_report.json"
      validation_path = default_validation_artifact_path(output_path)

      report = extract_project_manager_report(
        raw_response_path=RAW_RESPONSE_FIXTURE_PATH,
        schema_path=PM_SCHEMA_PATH,
        output_path=output_path,
      )

      self.assertTrue(output_path.is_file())
      self.assertTrue(validation_path.is_file())
      ProjectManagerReportValidationArtifact.model_validate(load_json(validation_path))
      self.assertEqual(report.report_status, "needs_clarification")
      self.assertTrue(report.proof_frontier.blocked)
      self.assertIsNotNone(report.report_source_coverage.repo_snapshot_packet)
      self.assertTrue(report.report_source_coverage.repo_snapshot_packet.consumed)
      self.assertEqual(
        report.report_source_coverage.repo_snapshot_packet.disposition,
        "inspected_insufficient",
      )
      self.assertTrue(
        any(
          "harness/runs/20260612-214948-agent-route/project_manager_report.json"
          in basis
          for basis in report.report_source_coverage.repo_snapshot_packet.basis
        )
      )

  def test_extractor_accepts_rejected_report_with_unblocked_frontier(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      output_path = temp_root / "project_manager_report.json"

      report = extract_project_manager_report(
        raw_response_path=REJECTED_UNBLOCKED_RAW_RESPONSE_FIXTURE_PATH,
        schema_path=PM_SCHEMA_PATH,
        output_path=output_path,
      )

      self.assertTrue(output_path.is_file())
      self.assertEqual(report.report_status, "rejected")
      self.assertFalse(report.proof_frontier.blocked)
      self.assertIsNone(report.proof_frontier.blocking_reason)
      self.assertIsNotNone(report.report_source_coverage.repo_snapshot_packet)
      self.assertTrue(report.report_source_coverage.repo_snapshot_packet.consumed)
      self.assertEqual(
        report.report_source_coverage.repo_snapshot_packet.disposition,
        "inspected_contradictory",
      )
      self.assertTrue(
        any(
          "harness/runs/20260612-214948-agent-route/project_manager_report.json"
          in basis
          for basis in report.report_source_coverage.repo_snapshot_packet.basis
        )
      )
      self.assertEqual(
        report.proof_frontier.next_admissible_transformation,
        "Classify the ledger as evidence of a recorded API call, not as proof of runtime state; if runtime proof is needed, require the corresponding saved run artifacts and validation/probe outputs.",
      )

  def test_extractor_writes_only_after_schema_validation_succeeds(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      raw_response_path = temp_root / "raw_model_response.json"
      output_path = temp_root / "project_manager_report.json"
      validation_path = default_validation_artifact_path(output_path)
      write_json(raw_response_path, load_json(RAW_RESPONSE_FIXTURE_PATH))
      schema = load_json(PM_SCHEMA_PATH)
      Draft202012Validator.check_schema(schema)
      expected_output = json.loads(load_json(raw_response_path)["output_text"])

      report = extract_project_manager_report(
        raw_response_path=raw_response_path,
        schema_path=PM_SCHEMA_PATH,
        output_path=output_path,
      )

      self.assertTrue(output_path.is_file())
      self.assertTrue(validation_path.is_file())
      self.assertEqual(load_json(output_path), expected_output)
      ProjectManagerReportValidationArtifact.model_validate(load_json(validation_path))
      self.assertEqual(report.report_status, "needs_clarification")
      self.assertIsNotNone(report.report_source_coverage.repo_snapshot_packet)

  def test_extractor_rejects_missing_required_consumed_source_before_write(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      raw_response_path = temp_root / "raw_model_response.json"
      output_path = temp_root / "project_manager_report.json"
      raw_response = load_json(RAW_RESPONSE_FIXTURE_PATH)
      report = json.loads(raw_response["output_text"])
      report["report_source_coverage"]["repo_snapshot_packet"] = {
        "consumed": False,
        "disposition": "missing",
        "basis": ["The required route source was not supplied."],
      }
      raw_response["output_text"] = json.dumps(report)
      write_json(raw_response_path, raw_response)

      with self.assertRaises(ProjectManagerReportExtractorError) as error:
        extract_project_manager_report(
          raw_response_path=raw_response_path,
          schema_path=PM_SCHEMA_PATH,
          output_path=output_path,
          required_consumed_sources={"repo_snapshot_packet"},
        )

      self.assertIn("repo_snapshot_packet", str(error.exception))
      self.assertFalse(output_path.exists())
      self.assertFalse(default_validation_artifact_path(output_path).exists())

  def test_extractor_rejects_missing_disposition_before_write(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      raw_response_path = temp_root / "raw_model_response.json"
      output_path = temp_root / "project_manager_report.json"
      raw_response = load_json(RAW_RESPONSE_FIXTURE_PATH)
      report = json.loads(raw_response["output_text"])
      del report["report_source_coverage"]["repo_snapshot_packet"]["disposition"]
      raw_response["output_text"] = json.dumps(report)
      write_json(raw_response_path, raw_response)

      with self.assertRaises(ProjectManagerReportExtractorError) as error:
        extract_project_manager_report(
          raw_response_path=raw_response_path,
          schema_path=PM_SCHEMA_PATH,
          output_path=output_path,
        )

      self.assertIn("$.report_source_coverage.repo_snapshot_packet", str(error.exception))
      self.assertIn("'disposition' is a required property", str(error.exception))
      self.assertFalse(output_path.exists())
      self.assertFalse(default_validation_artifact_path(output_path).exists())

  def test_extractor_fails_if_raw_response_status_is_not_completed(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      raw_response_path = temp_root / "raw_model_response.json"
      output_path = temp_root / "project_manager_report.json"
      raw_response = load_json(RAW_RESPONSE_FIXTURE_PATH)
      raw_response["status"] = "failed"
      write_json(raw_response_path, raw_response)

      with self.assertRaises(ProjectManagerReportExtractorError) as error:
        extract_project_manager_report(
          raw_response_path=raw_response_path,
          schema_path=PM_SCHEMA_PATH,
          output_path=output_path,
        )

      self.assertIn("status must be 'completed'", str(error.exception))
      self.assertFalse(output_path.exists())
      self.assertFalse(default_validation_artifact_path(output_path).exists())

  def test_extractor_fails_if_output_text_is_null(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      raw_response_path = temp_root / "raw_model_response.json"
      output_path = temp_root / "project_manager_report.json"
      raw_response = load_json(RAW_RESPONSE_FIXTURE_PATH)
      raw_response["output_text"] = None
      write_json(raw_response_path, raw_response)

      with self.assertRaises(ProjectManagerReportExtractorError) as error:
        extract_project_manager_report(
          raw_response_path=raw_response_path,
          schema_path=PM_SCHEMA_PATH,
          output_path=output_path,
        )

      self.assertIn("output_text must be present and non-empty", str(error.exception))
      self.assertFalse(output_path.exists())

  def test_extractor_fails_if_output_text_is_empty(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      raw_response_path = temp_root / "raw_model_response.json"
      output_path = temp_root / "project_manager_report.json"
      raw_response = load_json(RAW_RESPONSE_FIXTURE_PATH)
      raw_response["output_text"] = ""
      write_json(raw_response_path, raw_response)

      with self.assertRaises(ProjectManagerReportExtractorError) as error:
        extract_project_manager_report(
          raw_response_path=raw_response_path,
          schema_path=PM_SCHEMA_PATH,
          output_path=output_path,
        )

      self.assertIn("output_text must be present and non-empty", str(error.exception))
      self.assertFalse(output_path.exists())

  def test_extractor_fails_if_output_text_is_not_valid_json(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      raw_response_path = temp_root / "raw_model_response.json"
      output_path = temp_root / "project_manager_report.json"
      raw_response = load_json(RAW_RESPONSE_FIXTURE_PATH)
      raw_response["output_text"] = "{not json}"
      write_json(raw_response_path, raw_response)

      with self.assertRaises(ProjectManagerReportExtractorError) as error:
        extract_project_manager_report(
          raw_response_path=raw_response_path,
          schema_path=PM_SCHEMA_PATH,
          output_path=output_path,
        )

      self.assertEqual(
        str(error.exception),
        "raw_model_response.output_text is not valid JSON.",
      )
      self.assertFalse(output_path.exists())
      self.assertFalse(default_validation_artifact_path(output_path).exists())

  def test_extractor_fails_if_output_json_does_not_match_schema(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      raw_response_path = temp_root / "raw_model_response.json"
      output_path = temp_root / "project_manager_report.json"
      raw_response = load_json(RAW_RESPONSE_FIXTURE_PATH)
      parsed_output = json.loads(raw_response["output_text"])
      parsed_output["proof_frontier"]["blocked"] = "yes"
      raw_response["output_text"] = json.dumps(parsed_output)
      write_json(raw_response_path, raw_response)

      with self.assertRaises(ProjectManagerReportExtractorError) as error:
        extract_project_manager_report(
          raw_response_path=raw_response_path,
          schema_path=PM_SCHEMA_PATH,
          output_path=output_path,
        )

      self.assertIn("ProjectManagerReport validation failed at", str(error.exception))
      self.assertIn("$.proof_frontier.blocked", str(error.exception))
      self.assertFalse(output_path.exists())

  def test_extractor_does_not_write_output_on_parse_failure(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      raw_response_path = temp_root / "raw_model_response.json"
      output_path = temp_root / "project_manager_report.json"
      raw_response = load_json(RAW_RESPONSE_FIXTURE_PATH)
      raw_response["output_text"] = "{not json}"
      write_json(raw_response_path, raw_response)

      with self.assertRaises(ProjectManagerReportExtractorError):
        extract_project_manager_report(
          raw_response_path=raw_response_path,
          schema_path=PM_SCHEMA_PATH,
          output_path=output_path,
        )

      self.assertFalse(output_path.exists())

  def test_extractor_does_not_write_output_on_schema_validation_failure(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      raw_response_path = temp_root / "raw_model_response.json"
      schema_path = temp_root / "ProjectManagerReport.schema.json"
      output_path = temp_root / "project_manager_report.json"
      raw_response = load_json(RAW_RESPONSE_FIXTURE_PATH)
      parsed_output = json.loads(raw_response["output_text"])
      raw_response["output_text"] = json.dumps(parsed_output)
      write_json(raw_response_path, raw_response)
      schema = load_json(PM_SCHEMA_PATH)
      schema["properties"]["report_summary"]["minLength"] = 10_000
      write_json(schema_path, schema)

      with self.assertRaises(ProjectManagerReportExtractorError):
        extract_project_manager_report(
          raw_response_path=raw_response_path,
          schema_path=schema_path,
          output_path=output_path,
        )

      self.assertFalse(output_path.exists())
      self.assertFalse(default_validation_artifact_path(output_path).exists())

  def test_extractor_preserves_valid_parsed_json_without_wrapper_metadata(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      raw_response_path = temp_root / "raw_model_response.json"
      output_path = temp_root / "project_manager_report.json"
      raw_response = load_json(RAW_RESPONSE_FIXTURE_PATH)
      write_json(raw_response_path, raw_response)

      extract_project_manager_report(
        raw_response_path=raw_response_path,
        schema_path=PM_SCHEMA_PATH,
        output_path=output_path,
      )

      written = load_json(output_path)
      self.assertNotIn("raw_response", written)
      self.assertNotIn("provider", written)
      self.assertNotIn("endpoint", written)
      self.assertNotIn("source_artifacts", written)
      self.assertNotIn("basis", written)
      self.assertEqual(written, json.loads(raw_response["output_text"]))

  def test_extractor_direct_script_supports_cli_arguments(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      raw_response_path = temp_root / "raw_model_response.json"
      output_path = temp_root / "project_manager_report.json"
      write_json(raw_response_path, load_json(RAW_RESPONSE_FIXTURE_PATH))

      script_path = HARNESS_ROOT / "contracts" / "project_manager_report_extractor.py"
      completed = subprocess.run(
        [
          sys.executable,
          str(script_path),
          "--raw-response",
          str(raw_response_path),
          "--schema",
          str(PM_SCHEMA_PATH),
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
      self.assertTrue(default_validation_artifact_path(output_path).is_file())
      self.assertIn("PASS: Project Manager report written to", completed.stdout)
      self.assertIn("Status: needs_clarification", completed.stdout)
      self.assertIn("Blocked: True", completed.stdout)

  def test_extractor_does_not_call_openai(self) -> None:
    module_source = inspect.getsource(
      sys.modules["harness.contracts.project_manager_report_extractor"]
    )

    self.assertNotIn("from openai import", module_source)
    self.assertNotIn("OpenAI(", module_source)

  def test_extractor_does_not_create_or_modify_provider_payload_artifacts(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      raw_response_path = temp_root / "raw_model_response.json"
      provider_payload_path = temp_root / "provider_payload.json"
      output_path = temp_root / "project_manager_report.json"
      raw_response = load_json(RAW_RESPONSE_FIXTURE_PATH)
      write_json(raw_response_path, raw_response)
      provider_payload_path.write_text("sentinel payload\n", encoding="utf-8")

      extract_project_manager_report(
        raw_response_path=raw_response_path,
        schema_path=PM_SCHEMA_PATH,
        output_path=output_path,
      )

      self.assertEqual(provider_payload_path.read_text(encoding="utf-8"), "sentinel payload\n")
      self.assertTrue(output_path.is_file())

if __name__ == "__main__":
  unittest.main()
