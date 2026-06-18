from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from harness.agents.agent_contract import AgentContract
from harness.agents.archive_manager import ArchiveManager
from harness.agents.project_manager_agent import ProjectManagerAgent
from harness.contracts.archive_manager_report import ArchiveManagerReport
from harness.contracts.archive_manager_report_extractor import (
  ArchiveManagerReportExtractorError,
  extract_archive_manager_report,
)
from harness.contracts.archive_manager_report_validation import (
  ArchiveManagerReportValidationArtifact,
  default_validation_artifact_path as default_archive_validation_artifact_path,
)
from harness.contracts.project_manager_report import ProjectManagerReport
from harness.contracts.project_manager_report_extractor import (
  ProjectManagerReportExtractorError,
  extract_project_manager_report,
)
from harness.contracts.project_manager_report_validation import (
  ProjectManagerReportValidationArtifact,
  default_validation_artifact_path as default_project_manager_validation_artifact_path,
)


@dataclass(frozen=True, slots=True)
class AgentRouteSpec:
  output_id: str
  agent_model: type[Any]
  output_filename: str
  extractor: Callable[..., Any]
  extractor_error: type[Exception]
  validation_artifact_model: type[Any]
  default_validation_artifact_path: Callable[[Path], Path]
  extract_step: str
  validation_step: str
  contract_status: Callable[[Any], str | None]
  display_lines: Callable[[Any], list[str]]
  validate_specific_artifact: Callable[[Any, Any], None]


def _validate_project_manager_artifact(
  report: ProjectManagerReport,
  validation_artifact: ProjectManagerReportValidationArtifact,
) -> None:
  if validation_artifact.report_status != report.report_status:
    raise ValueError(
      "Validation artifact report_status does not match the validated report."
    )

  if validation_artifact.proof_frontier_blocked != report.proof_frontier.blocked:
    raise ValueError(
      "Validation artifact proof_frontier_blocked does not match the validated report."
    )


def _validate_archive_manager_artifact(
  report: ArchiveManagerReport,
  validation_artifact: ArchiveManagerReportValidationArtifact,
) -> None:
  if validation_artifact.archive_record_id != report.archive_record.record_id:
    raise ValueError(
      "Validation artifact archive_record_id does not match the validated report."
    )

  if validation_artifact.archive_record_type != report.archive_record.record_type:
    raise ValueError(
      "Validation artifact archive_record_type does not match the validated report."
    )


def _project_manager_display_lines(report: ProjectManagerReport) -> list[str]:
  return [
    f"Report status: {report.report_status}",
    f"Blocked: {report.proof_frontier.blocked}",
  ]


def _archive_manager_display_lines(report: ArchiveManagerReport) -> list[str]:
  return [
    f"Archive record id: {report.archive_record.record_id}",
    f"Archive record type: {report.archive_record.record_type}",
  ]


AGENT_ROUTE_SPECS: dict[str, AgentRouteSpec] = {
  "project_manager_report": AgentRouteSpec(
    output_id="project_manager_report",
    agent_model=ProjectManagerAgent,
    output_filename="project_manager_report.json",
    extractor=lambda **kwargs: extract_project_manager_report(**kwargs),
    extractor_error=ProjectManagerReportExtractorError,
    validation_artifact_model=ProjectManagerReportValidationArtifact,
    default_validation_artifact_path=default_project_manager_validation_artifact_path,
    extract_step="extract_project_manager_report",
    validation_step="validate_project_manager_report_validation_artifact",
    contract_status=lambda report: report.report_status,
    display_lines=_project_manager_display_lines,
    validate_specific_artifact=_validate_project_manager_artifact,
  ),
  "archive_manager_report": AgentRouteSpec(
    output_id="archive_manager_report",
    agent_model=ArchiveManager,
    output_filename="archive_manager_report.json",
    extractor=lambda **kwargs: extract_archive_manager_report(**kwargs),
    extractor_error=ArchiveManagerReportExtractorError,
    validation_artifact_model=ArchiveManagerReportValidationArtifact,
    default_validation_artifact_path=default_archive_validation_artifact_path,
    extract_step="extract_archive_manager_report",
    validation_step="validate_archive_manager_report_validation_artifact",
    contract_status=lambda report: report.archive_record.record_type,
    display_lines=_archive_manager_display_lines,
    validate_specific_artifact=_validate_archive_manager_artifact,
  ),
}


def validate_agent_contract(agent_data: dict[str, Any]) -> AgentContract:
  agent = AgentContract.model_validate(agent_data)
  if len(agent.agent_output_policy) != 1:
    return agent

  route_spec = AGENT_ROUTE_SPECS.get(agent.agent_output_policy[0].output_id)
  if route_spec is not None:
    route_spec.agent_model.model_validate(agent_data)

  return agent
