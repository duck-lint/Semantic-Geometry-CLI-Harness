from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from jsonschema import Draft202012Validator

from harness.contracts.archive_manager_report import ArchiveManagerReport
from harness.contracts.archive_manager_report_extractor import (
  ArchiveManagerReportExtractorError,
  extract_archive_manager_report,
)
from harness.contracts.archive_manager_report_validation import (
  ARCHIVE_MANAGER_REPORT_VALIDATOR,
  ArchiveManagerReportValidationArtifact,
  default_validation_artifact_path,
)
from harness.runtime.artifact_facts import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = REPO_ROOT / "harness"
AM_EXAMPLE_PATH = HARNESS_ROOT / "contracts" / "archive_manager_report.example.json"
AM_SCHEMA_PATH = HARNESS_ROOT / "contracts" / "ArchiveManagerReport.schema.json"


def load_json(path: Path):
  return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def raw_response_for(report: dict) -> dict:
  return {
    "metadata": {
      "document_id": "raw_model_response.json",
      "title": "OpenAI Raw Model Response",
      "purpose": "Raw OpenAI Responses API result captured before harness output validation.",
      "source_format": "json",
      "document_authority": "raw_provider_artifact",
    },
    "provider": "openai",
    "endpoint": "responses.create",
    "response_id": "resp_archive_manager_fixture",
    "model": "gpt-5.4-2026-03-05",
    "status": "completed",
    "output_text": json.dumps(report),
    "raw_response": {
      "id": "resp_archive_manager_fixture",
      "status": "completed",
      "model": "gpt-5.4-2026-03-05",
    },
    "source_artifacts": [
      "provider_payload.json",
    ],
    "basis": [
      "Mocked OpenAI raw response artifact for Archive Manager tests.",
    ],
  }


class ArchiveManagerReportExtractorTests(unittest.TestCase):
  def test_example_validates_against_schema_and_model(self) -> None:
    schema = load_json(AM_SCHEMA_PATH)
    example = load_json(AM_EXAMPLE_PATH)

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(example)
    ArchiveManagerReport.model_validate(example)

  def test_validation_artifact_model_defaults_and_path_helper(self) -> None:
    artifact = ArchiveManagerReportValidationArtifact(
      report_artifact_path="/tmp/archive_manager_report.json",
      report_artifact_sha256="abc123",
      schema_name="archive_manager_report",
      schema_path="/tmp/ArchiveManagerReport.schema.json",
      schema_sha256="def456",
      validation_passed=True,
      archive_record_id="implementation_01_pm_route_closeout",
      archive_record_type="implementation_closeout",
    )

    self.assertEqual(artifact.validator, ARCHIVE_MANAGER_REPORT_VALIDATOR)
    self.assertTrue(artifact.validation_passed)
    self.assertTrue(artifact.validation_timestamp_utc.endswith("Z"))
    datetime.fromisoformat(artifact.validation_timestamp_utc.replace("Z", "+00:00"))
    self.assertEqual(
      default_validation_artifact_path(Path("/tmp/archive_manager_report.json")),
      Path("/tmp/archive_manager_report.validation.json"),
    )

  def test_extractor_reads_valid_raw_response_and_writes_validation_artifact(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      raw_response_path = temp_root / "raw_model_response.json"
      output_path = temp_root / "archive_manager_report.json"
      validation_path = default_validation_artifact_path(output_path)
      expected_output = load_json(AM_EXAMPLE_PATH)
      write_json(raw_response_path, raw_response_for(expected_output))

      report = extract_archive_manager_report(
        raw_response_path=raw_response_path,
        schema_path=AM_SCHEMA_PATH,
        output_path=output_path,
      )

      self.assertTrue(output_path.is_file())
      self.assertTrue(validation_path.is_file())
      self.assertEqual(load_json(output_path), expected_output)
      self.assertEqual(report.archive_record.record_id, "implementation_01_pm_route_closeout")

      artifact = ArchiveManagerReportValidationArtifact.model_validate(
        load_json(validation_path)
      )
      self.assertEqual(artifact.report_artifact_path, output_path.as_posix())
      self.assertEqual(artifact.report_artifact_sha256, sha256_file(output_path))
      self.assertEqual(artifact.schema_name, "archive_manager_report")
      self.assertEqual(artifact.schema_path, AM_SCHEMA_PATH.as_posix())
      self.assertEqual(artifact.schema_sha256, sha256_file(AM_SCHEMA_PATH))
      self.assertTrue(artifact.validation_passed)
      self.assertEqual(artifact.archive_record_id, report.archive_record.record_id)
      self.assertEqual(artifact.archive_record_type, report.archive_record.record_type)

  def test_extractor_rejects_missing_required_consumed_source_before_write(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      raw_response_path = temp_root / "raw_model_response.json"
      output_path = temp_root / "archive_manager_report.json"
      report = load_json(AM_EXAMPLE_PATH)
      report["report_source_coverage"]["repo_snapshot_packet"] = {
        "consumed": False,
        "disposition": "missing",
        "basis": ["The required archive evidence packet was not supplied."],
      }
      write_json(raw_response_path, raw_response_for(report))

      with self.assertRaises(ArchiveManagerReportExtractorError) as error:
        extract_archive_manager_report(
          raw_response_path=raw_response_path,
          schema_path=AM_SCHEMA_PATH,
          output_path=output_path,
          required_consumed_sources={"repo_snapshot_packet"},
        )

      self.assertIn("repo_snapshot_packet", str(error.exception))
      self.assertFalse(output_path.exists())
      self.assertFalse(default_validation_artifact_path(output_path).exists())

  def test_extractor_rejects_schema_mismatch_before_write(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      raw_response_path = temp_root / "raw_model_response.json"
      output_path = temp_root / "archive_manager_report.json"
      report = load_json(AM_EXAMPLE_PATH)
      del report["archive_record"]["claim"]
      write_json(raw_response_path, raw_response_for(report))

      with self.assertRaises(ArchiveManagerReportExtractorError) as error:
        extract_archive_manager_report(
          raw_response_path=raw_response_path,
          schema_path=AM_SCHEMA_PATH,
          output_path=output_path,
        )

      self.assertIn("ArchiveManagerReport validation failed at", str(error.exception))
      self.assertIn("$.archive_record", str(error.exception))
      self.assertIn("'claim' is a required property", str(error.exception))
      self.assertFalse(output_path.exists())
      self.assertFalse(default_validation_artifact_path(output_path).exists())

  def test_extractor_fails_if_raw_response_status_is_not_completed(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      raw_response_path = temp_root / "raw_model_response.json"
      output_path = temp_root / "archive_manager_report.json"
      raw_response = raw_response_for(load_json(AM_EXAMPLE_PATH))
      raw_response["status"] = "failed"
      write_json(raw_response_path, raw_response)

      with self.assertRaises(ArchiveManagerReportExtractorError) as error:
        extract_archive_manager_report(
          raw_response_path=raw_response_path,
          schema_path=AM_SCHEMA_PATH,
          output_path=output_path,
        )

      self.assertIn("status must be 'completed'", str(error.exception))
      self.assertFalse(output_path.exists())
      self.assertFalse(default_validation_artifact_path(output_path).exists())

  def test_extractor_fails_if_output_text_is_not_valid_json(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      raw_response_path = temp_root / "raw_model_response.json"
      output_path = temp_root / "archive_manager_report.json"
      raw_response = raw_response_for(load_json(AM_EXAMPLE_PATH))
      raw_response["output_text"] = "{not json}"
      write_json(raw_response_path, raw_response)

      with self.assertRaises(ArchiveManagerReportExtractorError) as error:
        extract_archive_manager_report(
          raw_response_path=raw_response_path,
          schema_path=AM_SCHEMA_PATH,
          output_path=output_path,
        )

      self.assertEqual(
        str(error.exception),
        "raw_model_response.output_text is not valid JSON.",
      )
      self.assertFalse(output_path.exists())
      self.assertFalse(default_validation_artifact_path(output_path).exists())

  def test_extractor_preserves_valid_parsed_json_without_wrapper_metadata(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      raw_response_path = temp_root / "raw_model_response.json"
      output_path = temp_root / "archive_manager_report.json"
      raw_response = raw_response_for(load_json(AM_EXAMPLE_PATH))
      write_json(raw_response_path, raw_response)

      extract_archive_manager_report(
        raw_response_path=raw_response_path,
        schema_path=AM_SCHEMA_PATH,
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
      output_path = temp_root / "archive_manager_report.json"
      write_json(raw_response_path, raw_response_for(load_json(AM_EXAMPLE_PATH)))

      script_path = HARNESS_ROOT / "contracts" / "archive_manager_report_extractor.py"
      completed = subprocess.run(
        [
          sys.executable,
          str(script_path),
          "--raw-response",
          str(raw_response_path),
          "--schema",
          str(AM_SCHEMA_PATH),
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
      self.assertIn("PASS: Archive Manager report written to", completed.stdout)
      self.assertIn("Record id: implementation_01_pm_route_closeout", completed.stdout)
      self.assertIn("Record type: implementation_closeout", completed.stdout)

  def test_extractor_does_not_call_openai(self) -> None:
    module_source = inspect.getsource(
      sys.modules["harness.contracts.archive_manager_report_extractor"]
    )

    self.assertNotIn("from openai import", module_source)
    self.assertNotIn("OpenAI(", module_source)


if __name__ == "__main__":
  unittest.main()
