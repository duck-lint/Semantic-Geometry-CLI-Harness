from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Support direct execution from harness/agents while preserving package imports.
if __package__ in {None, ""}:
  sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic import ValidationError

from harness.agents.agent_contract import AgentContract
from harness.agents.agent_context_packet import (
  AgentContextInputCoverageEntry,
  AgentContextPacket,
  AgentContextPacketMetadata,
  AgentResolvedInputs,
)
from harness.project_spec.static_context_packet import StaticContextPacket
from harness.project_spec.static_context_packet_compiler import (
  StaticContextCompilationError,
  compile_static_context_packet,
)
from harness.repo_snapshot.repo_snapshot_compiler import (
  RepoSnapshotCompilationError,
  compile_repo_snapshot_packet,
)
from harness.repo_snapshot.repo_snapshot_packet import RepoSnapshotPacket


class AgentContextCompilationError(RuntimeError):
  def __init__(
    self,
    message: str,
    *,
    input_coverage: list[AgentContextInputCoverageEntry],
  ) -> None:
    super().__init__(message)
    self.input_coverage = input_coverage


@dataclass(slots=True)
class AgentContextCompileOptions:
  manifest_path: Path | None
  repo_root: Path | None
  harness_root: Path | None
  target_repo_root: Path | None
  static_context_output_path: Path | None
  repo_snapshot_output_path: Path | None
  static_context_packet: StaticContextPacket | dict[str, Any] | None = None
  static_context_override_path: Path | None = None


@dataclass(slots=True)
class ResolvedInputResult:
  resolved_value: Any
  coverage_entry: AgentContextInputCoverageEntry


def _load_json(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as file:
    data = json.load(file)

  if not isinstance(data, dict):
    raise TypeError(f"Expected a JSON object in {path}.")

  return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as file:
    json.dump(data, file, indent=2)
    file.write("\n")


def _coverage_entry(
  *,
  input_id: str,
  required: bool,
  status: str,
  schema_ref: str | None,
  basis: list[str],
) -> AgentContextInputCoverageEntry:
  return AgentContextInputCoverageEntry(
    input_id=input_id,
    required=required,
    status=status,
    schema_ref=schema_ref,
    basis=basis,
  )


def _format_blocking_input_coverage(
  input_coverage: list[AgentContextInputCoverageEntry],
) -> str:
  blocking_entries = [
    entry
    for entry in input_coverage
    if entry.status == "invalid"
    or (entry.required and entry.status == "missing")
  ]

  lines = [
    "Agent context compilation blocked by agent input policy.",
    f"Blocking inputs: {len(blocking_entries)}.",
  ]

  for entry in blocking_entries:
    requirement = "required" if entry.required else "optional"
    lines.append(f"- {entry.input_id} ({requirement}, {entry.status})")
    for basis_line in entry.basis:
      lines.append(f"  - {basis_line}")

  return "\n".join(lines)


def resolve_static_context_packet(
  *,
  policy: Any,
  options: AgentContextCompileOptions,
) -> ResolvedInputResult:
  basis: list[str] = []

  try:
    if options.static_context_packet is not None:
      packet = StaticContextPacket.model_validate(options.static_context_packet)
      if options.static_context_override_path is not None:
        basis.append(
          "Loaded StaticContextPacket from explicit override "
          f"{options.static_context_override_path.resolve()}."
        )
      else:
        basis.append("Loaded explicit StaticContextPacket override.")
      return ResolvedInputResult(
        resolved_value=packet,
        coverage_entry=_coverage_entry(
          input_id=policy.input_id,
          required=policy.required,
          status="included",
          schema_ref=policy.schema_ref,
          basis=basis,
        ),
      )

    missing_resolution_inputs = [
      name
      for name, value in (
        ("manifest_path", options.manifest_path),
        ("harness_root", options.harness_root),
        ("target_repo_root", options.target_repo_root),
        ("static_context_output_path", options.static_context_output_path),
      )
      if value is None
    ]

    if missing_resolution_inputs:
      return ResolvedInputResult(
        resolved_value=None,
        coverage_entry=_coverage_entry(
          input_id=policy.input_id,
          required=policy.required,
          status="missing",
          schema_ref=policy.schema_ref,
          basis=[
            "No StaticContextPacket override was provided and automatic "
            "resolution was not fully configured.",
            f"Missing compiler inputs: {', '.join(missing_resolution_inputs)}.",
          ],
        ),
      )

    packet = compile_static_context_packet(
      manifest_path=options.manifest_path.resolve(),
      harness_root=options.harness_root.resolve(),
      target_repo_root=options.target_repo_root.resolve(),
      output_path=options.static_context_output_path.resolve(),
    )
    basis.extend(
      [
        "StaticContextPacket was resolved automatically from the selected "
        "agent input policy.",
        f"Supporting artifact written to {options.static_context_output_path.resolve()}.",
      ]
    )
    return ResolvedInputResult(
      resolved_value=packet,
      coverage_entry=_coverage_entry(
        input_id=policy.input_id,
        required=policy.required,
        status="included",
        schema_ref=policy.schema_ref,
        basis=basis,
      ),
    )
  except (
    OSError,
    TypeError,
    ValueError,
    ValidationError,
    StaticContextCompilationError,
  ) as error:
    return ResolvedInputResult(
      resolved_value=None,
      coverage_entry=_coverage_entry(
        input_id=policy.input_id,
        required=policy.required,
        status="invalid",
        schema_ref=policy.schema_ref,
        basis=[f"StaticContextPacket resolution failed: {error}"],
      ),
    )


def resolve_repo_snapshot_packet(
  *,
  policy: Any,
  options: AgentContextCompileOptions,
) -> ResolvedInputResult:
  resolution = getattr(policy, "resolution", None)

  if resolution is None:
    return ResolvedInputResult(
      resolved_value=None,
      coverage_entry=_coverage_entry(
        input_id=policy.input_id,
        required=policy.required,
        status="missing",
        schema_ref=policy.schema_ref,
        basis=[
          "repo_snapshot_packet requires an explicit resolution object in agent_input_policy."
        ],
      ),
    )

  missing_resolution_inputs = [
    name
    for name, value in (
      ("repo_root", options.repo_root),
      ("repo_snapshot_output_path", options.repo_snapshot_output_path),
    )
    if value is None
  ]
  if missing_resolution_inputs:
    return ResolvedInputResult(
      resolved_value=None,
      coverage_entry=_coverage_entry(
        input_id=policy.input_id,
        required=policy.required,
        status="missing",
        schema_ref=policy.schema_ref,
        basis=[
          "RepoSnapshotPacket automatic resolution was not fully configured.",
          f"Missing compiler inputs: {', '.join(missing_resolution_inputs)}.",
        ],
      ),
    )

  try:
    packet = compile_repo_snapshot_packet(
      repo_root=options.repo_root.resolve(),
      output_path=options.repo_snapshot_output_path.resolve(),
      mode=resolution.mode,
      include_harness=resolution.include_harness,
      requested_paths=resolution.paths,
      requested_globs=resolution.globs,
      max_file_bytes=resolution.max_file_bytes or 100_000,
      max_total_bytes=resolution.max_total_bytes or 1_000_000,
    )
    return ResolvedInputResult(
      resolved_value=packet,
      coverage_entry=_coverage_entry(
        input_id=policy.input_id,
        required=policy.required,
        status="included",
        schema_ref=policy.schema_ref,
        basis=[
          "RepoSnapshotPacket was resolved automatically from the selected agent input policy.",
          f"Supporting artifact written to {options.repo_snapshot_output_path.resolve()}.",
        ],
      ),
    )
  except (
    OSError,
    RepoSnapshotCompilationError,
    TypeError,
    ValidationError,
    ValueError,
  ) as error:
    return ResolvedInputResult(
      resolved_value=None,
      coverage_entry=_coverage_entry(
        input_id=policy.input_id,
        required=policy.required,
        status="invalid",
        schema_ref=policy.schema_ref,
        basis=[f"RepoSnapshotPacket resolution failed: {error}"],
      ),
    )


InputResolver = Callable[..., ResolvedInputResult]

INPUT_RESOLVERS: dict[str, InputResolver] = {
  "static_context_packet": resolve_static_context_packet,
  "repo_snapshot_packet": resolve_repo_snapshot_packet,
}


def compile_agent_context_packet(
  *,
  agent_path: Path,
  output_path: Path,
  static_context_packet: StaticContextPacket | dict[str, Any] | None = None,
  manifest_path: Path | None = None,
  repo_root: Path | None = None,
  harness_root: Path | None = None,
  target_repo_root: Path | None = None,
  static_context_output_path: Path | None = None,
  repo_snapshot_output_path: Path | None = None,
  static_context_override_path: Path | None = None,
) -> AgentContextPacket:
  script_path = Path(__file__).resolve()
  agent = AgentContract.model_validate(_load_json(agent_path))
  if repo_root is None and any(
    policy.input_id == "repo_snapshot_packet"
    for policy in agent.agent_input_policy
  ):
    repo_root = script_path.parents[2]
  if repo_snapshot_output_path is None and any(
    policy.input_id == "repo_snapshot_packet"
    for policy in agent.agent_input_policy
  ):
    repo_snapshot_output_path = output_path.with_name("repo_snapshot_packet.json")

  input_coverage: list[AgentContextInputCoverageEntry] = []
  resolved_inputs = AgentResolvedInputs()
  compile_options = AgentContextCompileOptions(
    manifest_path=manifest_path,
    repo_root=repo_root,
    harness_root=harness_root,
    target_repo_root=target_repo_root,
    static_context_output_path=static_context_output_path,
    repo_snapshot_output_path=repo_snapshot_output_path,
    static_context_packet=static_context_packet,
    static_context_override_path=static_context_override_path,
  )

  for policy in agent.agent_input_policy:
    resolver = INPUT_RESOLVERS.get(policy.input_id)

    if resolver is None:
      input_coverage.append(
        _coverage_entry(
          input_id=policy.input_id,
          required=policy.required,
          status="invalid" if policy.required else "missing",
          schema_ref=policy.schema_ref,
          basis=[
            "Input is not supported by this agent context compiler."
          ],
        )
      )
      continue

    result = resolver(policy=policy, options=compile_options)
    input_coverage.append(result.coverage_entry)

    if policy.input_id == "static_context_packet":
      resolved_inputs.static_context_packet = result.resolved_value
    elif policy.input_id == "repo_snapshot_packet":
      resolved_inputs.repo_snapshot_packet = result.resolved_value

  blocking_entries = [
    entry
    for entry in input_coverage
    if entry.status == "invalid"
    or (entry.required and entry.status == "missing")
  ]

  if blocking_entries:
    raise AgentContextCompilationError(
      _format_blocking_input_coverage(input_coverage),
      input_coverage=input_coverage,
    )

  packet = AgentContextPacket(
    metadata=AgentContextPacketMetadata(
      document_id="agent_context_packet.json",
      title="Agent Context Packet",
      purpose="Agent-specific context compiled from the selected agent input policy.",
      source_format="json",
      document_authority="compiled_runtime_artifact",
    ),
    agent_contract=agent,
    resolved_inputs=resolved_inputs,
    input_coverage=input_coverage,
  )

  _write_json(output_path, packet.model_dump(mode="json", by_alias=True))
  return AgentContextPacket.model_validate(_load_json(output_path))


def build_argument_parser() -> argparse.ArgumentParser:
  script_path = Path(__file__).resolve()
  repo_root = script_path.parents[2]
  harness_root = repo_root / "harness"

  parser = argparse.ArgumentParser(
    description="Compile and validate an AgentContextPacket artifact.",
  )
  parser.add_argument(
    "--agent",
    type=Path,
    required=True,
    help="Path to the selected .agent.json contract.",
  )
  parser.add_argument(
    "--static-context",
    type=Path,
    default=None,
    help="Optional explicit StaticContextPacket override for fixture/debug use.",
  )
  parser.add_argument(
    "--manifest",
    type=Path,
    default=harness_root / "project_spec" / "static_context_packet.manifest.json",
    help="Path to StaticContextPacket manifest for automatic policy-driven resolution.",
  )
  parser.add_argument(
    "--repo-root",
    type=Path,
    default=repo_root,
    help="Root of the repo that repo_snapshot_packet resolution should inspect.",
  )
  parser.add_argument(
    "--harness-root",
    type=Path,
    default=repo_root,
    help="Repository root used to resolve harness_global static context sources.",
  )
  parser.add_argument(
    "--target-repo-root",
    type=Path,
    default=repo_root,
    help="Repository root used to resolve target_repo static context sources.",
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=harness_root / "runs" / "agent_context_packet.json",
    help="Destination for the emitted AgentContextPacket JSON.",
  )
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_argument_parser().parse_args(argv)

  static_context_packet = None

  try:
    if args.static_context is not None:
      static_context_packet = _load_json(args.static_context.resolve())

    packet = compile_agent_context_packet(
      agent_path=args.agent.resolve(),
      output_path=args.output.resolve(),
      static_context_packet=static_context_packet,
      manifest_path=args.manifest.resolve(),
      repo_root=args.repo_root.resolve(),
      harness_root=args.harness_root.resolve(),
      target_repo_root=args.target_repo_root.resolve(),
      static_context_output_path=(
        args.output.resolve().with_name("static_context_packet.json")
      ),
      repo_snapshot_output_path=(
        args.output.resolve().with_name("repo_snapshot_packet.json")
      ),
      static_context_override_path=(
        args.static_context.resolve()
        if args.static_context is not None
        else None
      ),
    )
  except (
    AgentContextCompilationError,
    OSError,
    TypeError,
    ValueError,
    ValidationError,
  ) as error:
    print(f"FAIL: {error}", file=sys.stderr)
    return 1

  print(f"PASS: Agent context packet written to {args.output.resolve()}")
  print(
    "Inputs: "
    f"{sum(entry.status == 'included' for entry in packet.input_coverage)} included, "
    f"{sum(entry.status == 'missing' for entry in packet.input_coverage)} missing, "
    f"{sum(entry.status == 'invalid' for entry in packet.input_coverage)} invalid."
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
