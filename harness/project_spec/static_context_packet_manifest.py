from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Metadata(BaseModel):
  model_config = ConfigDict(extra="forbid")

  id: Literal["static_context_packet.manifest.json"]
  name: Literal["Static Context Packet Manifest"]


class Source(BaseModel):
  model_config = ConfigDict(extra="forbid")

  source_id: Literal[
    "governance_primitives",
    "project_spec",
    "known_failures",
    "open_decisions",
    "active_implementation_plan",
    "active_implementation_tracker",
  ]

  scope: Literal["harness_global", "target_repo"]

  required: bool
  required_when: str | None = None

  document_authority: Literal[
    "harness_target",
    "operational_state",
    "global_harness",
    "compiled_runtime_artifact",
    "output_policy_artifact",
    "raw_provider_artifact"
  ]

  document: str | None = None
  document_glob: str | None = None

  schema_id: Literal[
    "governance_primitives",
    "project_spec",
    "known_failures",
    "open_decisions",
    "implementation_plan",
    "implementation_tracker",
  ]

  cardinality: Literal[
    "exactly_one",
    "zero_or_one",
    "zero_or_more",
    "one_or_more",
]

  @model_validator(mode="after")
  def enforce_document_reference_shape(self):
    has_document = self.document is not None
    has_glob = self.document_glob is not None

    if has_document == has_glob:
      raise ValueError(
        "Each manifest source must define exactly one of document or document_glob."
        )

    if self.document is not None and "*" in self.document:
      raise ValueError(
        "Use document_glob for glob patterns; document must be a single path."
      )

    return self

  @model_validator(mode="after")
  def enforce_source_specific_contract(self):
    expected = SOURCE_CONTRACTS[self.source_id]

    for field_name, expected_value in expected.items():
      actual_value = getattr(self, field_name)

      if actual_value != expected_value:
        raise ValueError(
          f"{self.source_id}.{field_name} must be {expected_value!r}, "
          f"got {actual_value!r}."
        )

    return self

class StaticContextPacketManifest(BaseModel):
  model_config = ConfigDict(extra="forbid")

  schema_ref: Literal["./StaticContextPacketManifest.schema.json"] = Field(
    ...,
    alias="$schema",
  )
  metadata: Metadata
  sources: list[Source]

  @model_validator(mode="after")
  def enforce_complete_source_set(self):
    seen = [source.source_id for source in self.sources]
    expected = set(SOURCE_CONTRACTS)

    duplicates = sorted(
      source_id for source_id in set(seen) if seen.count(source_id) > 1
    )

    missing = sorted(expected - set(seen))
    unexpected = sorted(set(seen) - expected)

    if duplicates:
      raise ValueError(f"Duplicate manifest sources are not allowed: {duplicates}")

    if missing:
      raise ValueError(f"Manifest is missing required source entries: {missing}")

    if unexpected:
      raise ValueError(f"Manifest contains unexpected source entries: {unexpected}")

    return self


SOURCE_CONTRACTS = {
  "governance_primitives": {
    "scope": "harness_global",
    "required": True,
    "required_when": None,
    "document_authority": "global_harness",
    "schema_id": "governance_primitives",
    "cardinality": "exactly_one",
  },
  "project_spec": {
    "scope": "target_repo",
    "required": True,
    "required_when": None,
    "document_authority": "harness_target",
    "schema_id": "project_spec",
    "cardinality": "exactly_one",
  },
  "known_failures": {
    "scope": "target_repo",
    "required": True,
    "required_when": None,
    "document_authority": "harness_target",
    "schema_id": "known_failures",
    "cardinality": "exactly_one",
  },
  "open_decisions": {
    "scope": "target_repo",
    "required": True,
    "required_when": None,
    "document_authority": "harness_target",
    "schema_id": "open_decisions",
    "cardinality": "exactly_one",
  },
  "active_implementation_plan": {
    "scope": "target_repo",
    "required": False,
    "required_when": "active_implementation_exists",
    "document_authority": "operational_state",
    "document_glob": "harness/implementations/active/implementation_plan_*.json",
    "schema_id": "implementation_plan",
    "cardinality": "zero_or_one",
  },
  "active_implementation_tracker": {
    "scope": "target_repo",
    "required": False,
    "required_when": "active_implementation_exists",
    "document_authority": "operational_state",
    "document_glob": "harness/implementations/active/implementation_tracker_*.json",
    "schema_id": "implementation_tracker",
    "cardinality": "zero_or_one",
  },
}
