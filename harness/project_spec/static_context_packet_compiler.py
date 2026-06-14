from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Support direct execution from harness/project_spec while preserving package imports.
if __package__ in {None, ""}:
  sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic import ValidationError

from harness.project_spec.static_context_packet import (
  CoverageStatus,
  MissingSourceEntry,
  SourceCoverageEntry,
  SourceValidation,
  StaticContextPacket,
  StaticContextPacketMetadata,
  ValidationStatus,
)
from harness.project_spec.static_context_packet_manifest import (
  Source,
  StaticContextPacketManifest,
)
class SourceCardinalityError(ValueError):
  pass


class StaticContextCompilationError(RuntimeError):
  def __init__(
    self,
    message: str,
    *,
    source_coverage: list[SourceCoverageEntry],
    missing_sources: list[MissingSourceEntry],
  ) -> None:
    super().__init__(message)
    self.source_coverage = source_coverage
    self.missing_sources = missing_sources


def load_json(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as file:
    data = json.load(file)

  if not isinstance(data, dict):
    raise TypeError(f"Expected a JSON object in {path}.")

  return data


def load_source_json(path: Path) -> Any:
  with path.open("r", encoding="utf-8") as file:
    return json.load(file)


def write_json(path: Path, data: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)

  with path.open("w", encoding="utf-8") as file:
    json.dump(data, file, indent=2)
    file.write("\n")


def load_and_validate_manifest(path: Path) -> StaticContextPacketManifest:
  return StaticContextPacketManifest.model_validate(load_json(path))


def resolve_source_paths(
  source: Source,
  harness_root: Path,
  target_repo_root: Path,
) -> list[Path]:
  root = harness_root if source.scope == "harness_global" else target_repo_root

  if source.document is not None:
    path = root / source.document
    return [path] if path.is_file() else []

  if source.document_glob is None:
    raise ValueError(f"{source.source_id} has no document reference.")

  return sorted(path for path in root.glob(source.document_glob) if path.is_file())


def enforce_cardinality(source: Source, paths: list[Path]) -> None:
  count = len(paths)
  valid = {
    "exactly_one": count == 1,
    "zero_or_one": count <= 1,
    "zero_or_more": True,
    "one_or_more": count >= 1,
  }[source.cardinality]

  if not valid:
    raise SourceCardinalityError(
      f"{source.source_id} requires cardinality {source.cardinality}, "
      f"but resolved {count} paths."
    )


def _coverage_entry(
  source: Source,
  *,
  status: CoverageStatus,
  resolved_path: Path | None,
  validation_status: ValidationStatus,
  parsed_content_available: bool,
  basis: list[str],
) -> SourceCoverageEntry:
  return SourceCoverageEntry(
    source_id=source.source_id,
    layer="static_context",
    required=source.required,
    document_authority=source.document_authority,
    schema_id=source.schema_id,
    status=status,
    resolved_path=str(resolved_path.resolve()) if resolved_path is not None else None,
    validation=SourceValidation(
      status=validation_status,
      validator="json_parse",
      parsed_content_available=parsed_content_available,
    ),
    basis=basis,
  )


def _reference_description(source: Source) -> str:
  reference = source.document or source.document_glob
  return f"{source.scope}:{reference}"


def compile_static_context_packet(
  manifest_path: Path,
  harness_root: Path,
  target_repo_root: Path,
  output_path: Path,
) -> StaticContextPacket:
  manifest = load_and_validate_manifest(manifest_path)

  included: dict[str, Any] = {}
  source_coverage: list[SourceCoverageEntry] = []
  missing_sources: list[MissingSourceEntry] = []

  for source in manifest.sources:
    paths = resolve_source_paths(source, harness_root, target_repo_root)

    if not paths:
      missing = MissingSourceEntry(
        source_id=source.source_id,
        schema_id=source.schema_id,
        document_authority=source.document_authority,
        effect="blocks_compilation" if source.required else "recorded_absent",
      )
      missing_sources.append(missing)
      included[source.source_id] = None
      source_coverage.append(
        _coverage_entry(
          source,
          status="missing",
          resolved_path=None,
          validation_status="not_run",
          parsed_content_available=False,
          basis=[f"No source matched {_reference_description(source)}."],
        )
      )
      continue

    try:
      enforce_cardinality(source, paths)
    except SourceCardinalityError as error:
      included[source.source_id] = None
      source_coverage.append(
        _coverage_entry(
          source,
          status="invalid",
          resolved_path=None,
          validation_status="not_run",
          parsed_content_available=False,
          basis=[str(error), *(str(path.resolve()) for path in paths)],
        )
      )
      continue

    parsed_documents: list[Any] = []

    try:
      for path in paths:
        parsed_documents.append(load_source_json(path))
    except (OSError, json.JSONDecodeError) as error:
      included[source.source_id] = None
      coverage = _coverage_entry(
        source,
        status="invalid",
        resolved_path=paths[0] if len(paths) == 1 else None,
        validation_status="failed",
        parsed_content_available=False,
        basis=[f"JSON parsing failed: {error}"],
      )
      coverage.validation.failure = str(error)
      source_coverage.append(coverage)
      continue

    if source.cardinality in {"zero_or_more", "one_or_more"}:
      included[source.source_id] = parsed_documents
    else:
      included[source.source_id] = parsed_documents[0]
    source_coverage.append(
      _coverage_entry(
        source,
        status="included",
        resolved_path=paths[0] if len(paths) == 1 else None,
        validation_status="passed",
        parsed_content_available=True,
        basis=[
          f"Resolved from {_reference_description(source)}.",
          "Parsed declared source content as JSON without normalization.",
          *(
            [f"Included {len(paths)} documents in resolved path order."]
            if len(paths) > 1
            else []
          ),
        ],
      )
    )

  blocking_missing = [
    entry for entry in missing_sources if entry.effect == "blocks_compilation"
  ]
  blocking_source_failures = [
    entry
    for entry in source_coverage
    if entry.status == "invalid"
  ]

  if blocking_missing or blocking_source_failures:
    raise StaticContextCompilationError(
      "Static context compilation blocked by source diagnostics.",
      source_coverage=source_coverage,
      missing_sources=missing_sources,
    )

  packet = StaticContextPacket(
    metadata=StaticContextPacketMetadata(
      document_id="static_context_packet.json",
      title="Static Context Packet",
      purpose="Compiled static authority and operational context from StaticContextPacketManifest.",
      source_format="json",
      document_authority="compiled_runtime_artifact",
    ),
    sources=included,
    source_coverage=source_coverage,
    missing_sources=missing_sources,
  )

  emitted = packet.model_dump(mode="json")
  write_json(output_path, emitted)
  return StaticContextPacket.model_validate(load_json(output_path))


def build_argument_parser() -> argparse.ArgumentParser:
  script_path = Path(__file__).resolve()
  repo_root = script_path.parents[2]

  parser = argparse.ArgumentParser(
    description="Compile manifest-declared JSON into a StaticContextPacket artifact.",
  )
  parser.add_argument(
    "--manifest",
    type=Path,
    default=script_path.with_name("static_context_packet.manifest.json"),
    help="Path to StaticContextPacketManifest JSON.",
  )
  parser.add_argument(
    "--harness-root",
    type=Path,
    default=repo_root,
    help="Repository root used to resolve harness_global manifest sources.",
  )
  parser.add_argument(
    "--target-repo-root",
    type=Path,
    default=repo_root,
    help="Repository root used to resolve target_repo manifest sources.",
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=repo_root / "harness" / "runs" / "static_context_packet.json",
    help="Destination for the emitted StaticContextPacket JSON.",
  )
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_argument_parser().parse_args(argv)

  try:
    packet = compile_static_context_packet(
      manifest_path=args.manifest.resolve(),
      harness_root=args.harness_root.resolve(),
      target_repo_root=args.target_repo_root.resolve(),
      output_path=args.output.resolve(),
    )
  except (
    OSError,
    TypeError,
    ValueError,
    ValidationError,
    StaticContextCompilationError,
  ) as error:
    print(f"FAIL: {error}", file=sys.stderr)
    return 1

  print(f"PASS: Static context packet written to {args.output.resolve()}")
  print(
    "Sources: "
    f"{sum(entry.status == 'included' for entry in packet.source_coverage)} included, "
    f"{len(packet.missing_sources)} missing, "
    f"{sum(entry.status == 'invalid' for entry in packet.source_coverage)} invalid."
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
