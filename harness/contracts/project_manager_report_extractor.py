from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError as JsonSchemaValidationError
from pydantic import ValidationError

# Support direct execution from harness/contracts while preserving package imports.
if __package__ in {None, ""}:
  sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness.contracts.project_manager_report import ProjectManagerReport
from harness.contracts.project_manager_report_validation import (
  PROJECT_MANAGER_REPORT_VALIDATOR,
  ProjectManagerReportValidationArtifact,
  default_validation_artifact_path,
)
from harness.providers.openai.openai_raw_response import OpenAIRawResponse
from harness.runtime.artifact_facts import sha256_file


class ProjectManagerReportExtractorError(RuntimeError):
  pass


def _load_json_object(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as file:
    data = json.load(file)

  if not isinstance(data, dict):
    raise ProjectManagerReportExtractorError(f"Expected a JSON object in {path}.")

  return data


def _format_jsonschema_error(error: JsonSchemaValidationError) -> str:
  json_path = getattr(error, "json_path", None)
  if not json_path:
    json_path = "$"

  return f"{json_path}: {error.message}"


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(
    mode="w",
    encoding="utf-8",
    dir=path.parent,
    prefix=f"{path.name}.",
    suffix=".tmp",
    delete=False,
  ) as file:
    temp_path = Path(file.name)
    json.dump(data, file, indent=2)
    file.write("\n")

  try:
    os.replace(temp_path, path)
  finally:
    if temp_path.exists():
      temp_path.unlink()


def extract_project_manager_report(
  *,
  raw_response_path: Path,
  schema_path: Path,
  output_path: Path,
  required_consumed_sources: set[str] | None = None,
) -> ProjectManagerReport:
  raw_response_data = _load_json_object(raw_response_path)
  raw_response = OpenAIRawResponse.model_validate(raw_response_data)

  if raw_response.status != "completed":
    raise ProjectManagerReportExtractorError(
      "raw_model_response.status must be 'completed'."
    )

  output_text = raw_response.output_text
  if not isinstance(output_text, str) or not output_text.strip():
    raise ProjectManagerReportExtractorError(
      "raw_model_response.output_text must be present and non-empty."
    )

  try:
    parsed_output = json.loads(output_text)
  except json.JSONDecodeError as error:
    raise ProjectManagerReportExtractorError(
      "raw_model_response.output_text is not valid JSON."
    ) from error

  schema = _load_json_object(schema_path)
  Draft202012Validator.check_schema(schema)
  validator = Draft202012Validator(schema)

  try:
    validator.validate(parsed_output)
  except JsonSchemaValidationError as error:
    raise ProjectManagerReportExtractorError(
      f"ProjectManagerReport validation failed at {_format_jsonschema_error(error)}"
    ) from error

  report = ProjectManagerReport.model_validate(parsed_output)
  if required_consumed_sources:
    coverage = report.report_source_coverage
    missing_or_unconsumed: list[str] = []
    for source_id in sorted(required_consumed_sources):
      entry = getattr(coverage, source_id, None)
      if entry is None or not entry.consumed:
        missing_or_unconsumed.append(source_id)

    if missing_or_unconsumed:
      raise ProjectManagerReportExtractorError(
        "ProjectManagerReport did not consume required resolved input sources: "
        + ", ".join(missing_or_unconsumed)
        + "."
      )

  _write_json_atomic(output_path, parsed_output)

  validation_artifact_path = default_validation_artifact_path(output_path)
  validation_artifact = ProjectManagerReportValidationArtifact(
    report_artifact_path=output_path.as_posix(),
    report_artifact_sha256=sha256_file(output_path),
    schema_name=report.metadata.document_id.removesuffix(".json"),
    schema_path=schema_path.as_posix(),
    schema_sha256=sha256_file(schema_path),
    validator=PROJECT_MANAGER_REPORT_VALIDATOR,
    validation_passed=True,
    report_status=report.report_status,
    proof_frontier_blocked=report.proof_frontier.blocked,
  )
  _write_json_atomic(
    validation_artifact_path,
    validation_artifact.model_dump(mode="json", exclude_none=True),
  )
  return report


def build_argument_parser() -> argparse.ArgumentParser:
  script_path = Path(__file__).resolve()
  harness_root = script_path.parents[1]

  parser = argparse.ArgumentParser(
    description=(
      "Extract output_text from a raw OpenAI response, validate it as a "
      "ProjectManagerReport, and write the report plus validation evidence "
      "only if validation succeeds."
    ),
  )
  parser.add_argument(
    "--raw-response",
    type=Path,
    default=harness_root / "runs" / "raw_model_response.json",
    help="Path to the raw OpenAI response artifact.",
  )
  parser.add_argument(
    "--schema",
    type=Path,
    default=script_path.with_name("ProjectManagerReport.schema.json"),
    help="Path to the ProjectManagerReport JSON schema.",
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=harness_root / "runs" / "project_manager_report.json",
    help="Destination for the validated ProjectManager report.",
  )
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_argument_parser().parse_args(argv)

  try:
    report = extract_project_manager_report(
      raw_response_path=args.raw_response.resolve(),
      schema_path=args.schema.resolve(),
      output_path=args.output.resolve(),
    )
  except (
    ProjectManagerReportExtractorError,
    OSError,
    TypeError,
    ValidationError,
    SchemaError,
    JsonSchemaValidationError,
    json.JSONDecodeError,
    ValueError,
  ) as error:
    print(f"FAIL: {error}", file=sys.stderr)
    return 1

  print(f"PASS: Project Manager report written to {args.output.resolve()}")
  print(f"Status: {report.report_status}")
  print(f"Blocked: {report.proof_frontier.blocked}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
