from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from harness.runtime.artifact_facts import utc_now_isoformat


ARCHIVE_MANAGER_REPORT_VALIDATOR = (
  "harness.contracts.archive_manager_report_extractor.extract_archive_manager_report"
)


def default_validation_artifact_path(report_artifact_path: Path) -> Path:
  return report_artifact_path.with_name(f"{report_artifact_path.stem}.validation.json")


class ArchiveManagerReportValidationArtifact(BaseModel):
  model_config = ConfigDict(extra="forbid")

  report_artifact_path: str
  report_artifact_sha256: str
  schema_name: str
  schema_path: str
  schema_sha256: str
  validator: str = ARCHIVE_MANAGER_REPORT_VALIDATOR
  validation_timestamp_utc: str = Field(default_factory=utc_now_isoformat)
  validation_passed: bool
  archive_record_id: str
  archive_record_type: str
