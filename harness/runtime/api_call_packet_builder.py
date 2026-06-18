from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Support direct execution from harness/runtime while preserving package imports.
if __package__ in {None, ""}:
  sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic import ValidationError

from harness.agents.agent_context_compiler import (
  AgentContextCompilationError,
  compile_agent_context_packet,
)
from harness.agents.agent_context_packet import AgentContextPacket
from harness.runtime.api_call_packet import (
  ApiCallPacket,
  ApiCallPacketMetadata,
  CallMode,
)
from harness.runtime.git_context import (
  GitContext,
  GitDeltaContext,
  collect_git_context,
)
from harness.runtime.runtime_budget_policy import RuntimeBudgetPolicy
from harness.runtime.supplementary_context import SupplementaryContextEntry
from harness.runtime.task import Task, task_from_cli


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


def build_api_call_packet(
  *,
  task: Task,
  call_mode: CallMode,
  output_path: Path,
  runtime_budget: RuntimeBudgetPolicy | None = None,
  git_context: GitContext | None = None,
  git_delta_context: GitDeltaContext | None = None,
  agent_context_packet: AgentContextPacket | None = None,
  supplementary_context: list[SupplementaryContextEntry] | None = None,
) -> ApiCallPacket:
  packet = ApiCallPacket(
    metadata=ApiCallPacketMetadata(
      document_id="api_call_packet.json",
      title="API Call Packet",
      purpose=(
        "Provider-neutral packet for one model call before provider-specific "
        "rendering."
      ),
      source_format="json",
      document_authority="compiled_runtime_artifact",
    ),
    call_mode=call_mode,
    task=task,
    runtime_budget=runtime_budget,
    agent_context_packet=agent_context_packet,
    git_context=git_context,
    git_delta_context=git_delta_context,
    supplementary_context=supplementary_context or [],
  )

  _write_json(output_path, packet.model_dump(mode="json", by_alias=True))
  return ApiCallPacket.model_validate(_load_json(output_path))


def _load_task_from_cli_text(task_text: str | None) -> Task:
  if task_text is None or not task_text.strip():
    raise ValueError("ApiCallPacket requires a task.")

  return task_from_cli(task_text.strip())


def _build_static_context_supplementary_entry(
  static_context_path: Path,
) -> SupplementaryContextEntry:
  return SupplementaryContextEntry(
    source_id="static_context_packet",
    source_type="static_context_packet",
    content=_load_json(static_context_path),
    basis=["Explicitly attached compiled StaticContextPacket."],
  )


def _build_repo_snapshot_supplementary_entry(
  repo_snapshot_path: Path,
) -> SupplementaryContextEntry:
  return SupplementaryContextEntry(
    source_id="repo_snapshot_packet",
    source_type="repo_snapshot_packet",
    content=_load_json(repo_snapshot_path),
    basis=["Explicitly attached compiled RepoSnapshotPacket."],
  )


def build_argument_parser() -> argparse.ArgumentParser:
  script_path = Path(__file__).resolve()
  harness_root = script_path.parents[1]
  repo_root = script_path.parents[2]

  parser = argparse.ArgumentParser(
    description="Build and validate an ApiCallPacket artifact.",
  )
  parser.add_argument(
    "--task",
    default=None,
    help="Task text to place into task.json-equivalent packet content.",
  )
  parser.add_argument(
    "--direct",
    action="store_true",
    help="Build a direct ApiCallPacket with no agent context.",
  )
  parser.add_argument(
    "--agent",
    type=Path,
    default=None,
    help="Path to the selected .agent.json contract for agent-routed calls.",
  )
  parser.add_argument(
    "--static-context",
    type=Path,
    default=None,
    help="Optional path to a StaticContextPacket JSON to attach as supplementary context in direct mode.",
  )
  parser.add_argument(
    "--repo-snapshot",
    type=Path,
    default=None,
    help="Optional path to a RepoSnapshotPacket JSON to attach as supplementary context in direct mode.",
  )
  parser.add_argument(
    "--runtime-budget",
    type=Path,
    default=None,
    help="Optional path to a RuntimeBudgetPolicy JSON.",
  )
  parser.add_argument(
    "--manifest",
    type=Path,
    default=harness_root / "project_spec" / "static_context_packet.manifest.json",
    help="Path to StaticContextPacket manifest for policy-driven agent input resolution.",
  )
  parser.add_argument(
    "--repo-root",
    type=Path,
    default=repo_root,
    help="Root of the repo used for repo_snapshot_packet resolution and git context collection.",
  )
  parser.add_argument(
    "--no-git-context",
    action="store_true",
    help="Do not collect git context into the emitted ApiCallPacket.",
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
    default=harness_root / "runs" / "api_call_packet.json",
    help="Destination for the emitted ApiCallPacket JSON.",
  )
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_argument_parser().parse_args(argv)

  if args.task is None or not args.task.strip():
    print("FAIL: ApiCallPacket requires a task.", file=sys.stderr)
    return 1

  if args.direct and args.agent is not None:
    print(
      "FAIL: direct ApiCallPacket must not include agent_context_packet.",
      file=sys.stderr,
    )
    return 1

  if args.agent is not None and args.static_context is not None:
    print(
      "FAIL: --static-context is only allowed for direct calls. "
      "Agent-routed static context must be resolved through the selected agent contract.",
      file=sys.stderr,
    )
    return 1

  if args.agent is not None and args.repo_snapshot is not None:
    print(
      "FAIL: --repo-snapshot is only allowed for direct calls. "
      "Agent-routed repo snapshots must be resolved through the selected agent contract.",
      file=sys.stderr,
    )
    return 1

  try:
    task = _load_task_from_cli_text(args.task)
    call_mode: CallMode = "agent_routed" if args.agent is not None else "direct"
    runtime_budget = (
      RuntimeBudgetPolicy.model_validate(_load_json(args.runtime_budget.resolve()))
      if args.runtime_budget is not None
      else None
    )
    output_path = args.output.resolve()
    agent_context_packet = None
    if args.agent is not None:
      agent_context_packet = compile_agent_context_packet(
        agent_path=args.agent.resolve(),
        output_path=output_path.with_name("agent_context_packet.json"),
        manifest_path=args.manifest.resolve(),
        repo_root=args.repo_root.resolve(),
        harness_root=args.harness_root.resolve(),
        target_repo_root=args.target_repo_root.resolve(),
        static_context_output_path=output_path.with_name("static_context_packet.json"),
        repo_snapshot_output_path=output_path.with_name("repo_snapshot_packet.json"),
      )

    supplementary_context_entries: list[SupplementaryContextEntry] = []
    if args.static_context is not None and args.agent is None:
      supplementary_context_entries.append(
        _build_static_context_supplementary_entry(args.static_context.resolve())
      )
    if args.repo_snapshot is not None and args.agent is None:
      supplementary_context_entries.append(
        _build_repo_snapshot_supplementary_entry(args.repo_snapshot.resolve())
      )
    supplementary_context = supplementary_context_entries or None
    git_context = (
      None
      if args.no_git_context
      else collect_git_context(args.repo_root.resolve())
    )

    packet = build_api_call_packet(
      task=task,
      call_mode=call_mode,
      runtime_budget=runtime_budget,
      agent_context_packet=agent_context_packet,
      git_context=git_context,
      supplementary_context=supplementary_context,
      output_path=output_path,
    )
  except (
    AgentContextCompilationError,
    OSError,
    TypeError,
    ValidationError,
    ValueError,
  ) as error:
    print(f"FAIL: {error}", file=sys.stderr)
    return 1

  print(f"PASS: API call packet written to {args.output.resolve()}")
  print(f"Mode: {packet.call_mode}")
  print(
    "Git context: "
    f"{'collected' if packet.git_context is not None else 'disabled'}."
  )
  print(
    "Supplementary context: "
    f"{len(packet.supplementary_context)} attached."
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
