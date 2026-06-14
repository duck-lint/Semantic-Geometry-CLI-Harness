from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Metadata(BaseModel):
  model_config = ConfigDict(extra="forbid")

  id: Literal["static_context_packet.manifest.json"]
  name: Literal["Static Context Packet Manifest"]


class Source(BaseModel):
  model_config = ConfigDict(extra="forbid")

  source_id: str = Field(min_length=1)

  scope: Literal["harness_global", "target_repo"]

  required: bool

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

  schema_id: str = Field(min_length=1)

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

class StaticContextPacketManifest(BaseModel):
  model_config = ConfigDict(extra="forbid")

  schema_ref: Literal["./StaticContextPacketManifest.schema.json"] = Field(
    ...,
    alias="$schema",
  )
  metadata: Metadata
  sources: list[Source] = Field(min_length=1)

  @model_validator(mode="after")
  def enforce_unique_source_ids(self):
    seen = [source.source_id for source in self.sources]
    duplicates = sorted(
      source_id for source_id in set(seen) if seen.count(source_id) > 1
    )

    if duplicates:
      raise ValueError(f"Duplicate manifest sources are not allowed: {duplicates}")

    return self
