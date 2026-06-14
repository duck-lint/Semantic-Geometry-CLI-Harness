from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

from harness.agents.agent_contract import (
  AgentInputPolicyEntry,
  AgentOutputPolicyEntry,
  ProviderId,
)


class Metadata(BaseModel):
  model_config = ConfigDict(extra="forbid")

  id: Literal["project_manager.agent.json"]
  agent_name: Literal["Project Manager"]
  document_authority: Literal["global_harness"]


class SourceCoverage(BaseModel):
  model_config = ConfigDict(extra="forbid")

  basis: str
  truthfulness_check: str


class TrajectoryReview(BaseModel):
  model_config = ConfigDict(extra="forbid")

  current_posture: str
  thesis_attractor: str
  drift_detection: str
  structural_tension: str


class ProofFrontier(BaseModel):
  model_config = ConfigDict(extra="forbid")

  constraint_conflicts: str
  dominant_tension_justification: str
  missing_basis: str
  next_admissible_transformation: str
  affected_surfaces: str
  non_affected_surfaces: str
  stop_conditions: str
  authority_constraints: str


class ArtifactTruthfulness(BaseModel):
  model_config = ConfigDict(extra="forbid")

  compiled_runtime_artifacts: str
  output_policy_artifacts: str
  raw_provider_artifacts: str
  non_runtime_templates: str


class StatusSemantics(BaseModel):
  model_config = ConfigDict(extra="forbid")

  admissible: str
  admissibility_blocked: str
  rejected: str
  needs_clarification: str


class ReportDerivationLenses(BaseModel):
  model_config = ConfigDict(extra="forbid")

  source_coverage: SourceCoverage
  trajectory_review: TrajectoryReview
  proof_frontier: ProofFrontier
  artifact_truthfulness: ArtifactTruthfulness
  status_semantics: StatusSemantics


class InstructionContract(BaseModel):
  model_config = ConfigDict(extra="forbid")

  role: str
  posture: str
  objective: str
  operating_context: list[str]
  project_manager_conduct: list[str]
  report_derivation_lenses: ReportDerivationLenses


class ProjectManagerOutputPolicyEntry(AgentOutputPolicyEntry):
  output_id: Literal["project_manager_report"]
  required: bool
  schema_ref: Literal["../contracts/ProjectManagerReport.schema.json"]


class ProjectManagerAgent(BaseModel):
  model_config = ConfigDict(extra="forbid")

  schema_ref: str = Field(..., alias='$schema')
  metadata: Metadata
  provider: ProviderId
  model: str = Field(min_length=1)
  instruction_contract: InstructionContract
  agent_input_policy: list[AgentInputPolicyEntry]
  agent_output_policy: list[ProjectManagerOutputPolicyEntry]
