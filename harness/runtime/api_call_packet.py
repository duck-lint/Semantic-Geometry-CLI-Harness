from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.agents.agent_context_packet import AgentContextPacket
from harness.runtime.git_context import GitContext, GitDeltaContext
from harness.runtime.runtime_budget_policy import RuntimeBudgetPolicy
from harness.runtime.supplementary_context import SupplementaryContextEntry
from harness.runtime.task import Task


CallMode = Literal["direct", "agent_routed"]


class ApiCallPacketMetadata(BaseModel):
  model_config = ConfigDict(extra="forbid")

  document_id: Literal["api_call_packet.json"]
  title: Literal["API Call Packet"]
  purpose: Literal[
    "Provider-neutral packet for one model call before provider-specific rendering."
  ]
  source_format: Literal["json"]
  document_authority: Literal["compiled_runtime_artifact"]


class ApiCallPacket(BaseModel):
  model_config = ConfigDict(extra="forbid")

  metadata: ApiCallPacketMetadata
  call_mode: CallMode
  task: Task
  runtime_budget: RuntimeBudgetPolicy | None = None
  agent_context_packet: AgentContextPacket | None = None
  git_context: GitContext | None = None
  git_delta_context: GitDeltaContext | None = None
  supplementary_context: list[SupplementaryContextEntry] = Field(
    default_factory=list
  )

  @model_validator(mode="after")
  def enforce_call_mode_shape(self):
    if self.call_mode == "agent_routed" and self.agent_context_packet is None:
      raise ValueError(
        "agent_routed ApiCallPacket requires agent_context_packet."
      )

    if self.call_mode == "direct" and self.agent_context_packet is not None:
      raise ValueError(
        "direct ApiCallPacket must not include agent_context_packet."
      )

    if self.git_context is not None and self.git_delta_context is not None:
      raise ValueError(
        "ApiCallPacket must not include both git_context and git_delta_context."
      )

    if (
      self.agent_context_packet is not None
      and self.agent_context_packet.resolved_inputs.static_context_packet
      is not None
      and any(
        entry.source_type == "static_context_packet"
        for entry in self.supplementary_context
      )
    ):
      raise ValueError(
        "StaticContextPacket must not be duplicated across agent and supplementary lanes."
      )

    if (
      self.agent_context_packet is not None
      and self.agent_context_packet.resolved_inputs.repo_snapshot_packet
      is not None
      and any(
        entry.source_type == "repo_snapshot_packet"
        for entry in self.supplementary_context
      )
    ):
      raise ValueError(
        "RepoSnapshotPacket must not be duplicated across agent and supplementary lanes."
      )

    return self
