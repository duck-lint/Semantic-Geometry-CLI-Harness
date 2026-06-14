from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from harness.agents.agent_context_compiler import (
  AgentContextCompilationError,
  compile_agent_context_packet,
)
from harness.agents.agent_context_packet import AgentContextPacket
from harness.project_spec.static_context_packet_compiler import (
  StaticContextCompilationError,
  compile_static_context_packet,
)
from harness.runtime.api_call_packet_builder import build_api_call_packet
from harness.runtime.git_context import collect_git_context
from harness.runtime.supplementary_context import SupplementaryContextEntry
from harness.runtime.task import Task, task_from_cli


def _write_json(path: Path, data: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as file:
    json.dump(data, file, indent=2)
    file.write("\n")


def _load_task(path: Path) -> Task:
  with path.open("r", encoding="utf-8") as file:
    return Task.model_validate(json.load(file))


def _new_run_id() -> str:
  return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _load_json(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as file:
    data = json.load(file)

  if not isinstance(data, dict):
    raise TypeError(f"Expected a JSON object in {path}.")

  return data


def _build_static_context_supplementary_entry(
  static_context_path: Path,
) -> SupplementaryContextEntry:
  return SupplementaryContextEntry(
    source_id="static_context_packet",
    source_type="static_context_packet",
    content=_load_json(static_context_path),
    basis=["Explicitly attached compiled StaticContextPacket."],
  )


def build_pre_call_artifacts(
  *,
  task_text: str,
  repo_root: Path,
  harness_root: Path,
  target_repo_root: Path,
  manifest_path: Path,
  runs_root: Path,
  agent_path: Path | None = None,
  attach_static_context: bool = False,
) -> tuple[Path, list[Path]]:
  run_directory = runs_root / _new_run_id()
  run_directory.mkdir(parents=True, exist_ok=False)

  task_path = run_directory / "task.json"
  api_call_path = run_directory / "api_call_packet.json"
  static_context_path = run_directory / "static_context_packet.json"
  agent_context_path = run_directory / "agent_context_packet.json"
  artifact_paths = [task_path]

  task = task_from_cli(task_text)
  _write_json(task_path, task.model_dump(mode="json"))
  task = _load_task(task_path)

  call_mode = "direct"
  supplementary_context: list[SupplementaryContextEntry] | None = None
  agent_context_packet: AgentContextPacket | None = None

  if agent_path is not None:
    call_mode = "agent_routed"
    agent_context_packet = compile_agent_context_packet(
      agent_path=agent_path,
      output_path=agent_context_path,
      manifest_path=manifest_path,
      repo_root=repo_root,
      harness_root=harness_root,
      target_repo_root=target_repo_root,
      static_context_output_path=static_context_path,
      repo_snapshot_output_path=run_directory / "repo_snapshot_packet.json",
    )
    if static_context_path.is_file():
      artifact_paths.append(static_context_path)
    repo_snapshot_path = run_directory / "repo_snapshot_packet.json"
    if repo_snapshot_path.is_file():
      artifact_paths.append(repo_snapshot_path)
    artifact_paths.append(agent_context_path)
  elif attach_static_context:
    compile_static_context_packet(
      manifest_path=manifest_path,
      harness_root=harness_root,
      target_repo_root=target_repo_root,
      output_path=static_context_path,
    )
    supplementary_context = [
      _build_static_context_supplementary_entry(static_context_path)
    ]
    artifact_paths.append(static_context_path)

  git_context = collect_git_context(repo_root)
  build_api_call_packet(
    task=task,
    call_mode=call_mode,
    agent_context_packet=agent_context_packet,
    git_context=git_context,
    supplementary_context=supplementary_context,
    output_path=api_call_path,
  )
  artifact_paths.append(api_call_path)

  return run_directory, artifact_paths


def build_argument_parser() -> argparse.ArgumentParser:
  repo_root = Path(__file__).resolve().parents[2]
  harness_root = repo_root / "harness"

  parser = argparse.ArgumentParser(
    description="Build provider-neutral harness call artifacts.",
  )
  subparsers = parser.add_subparsers(dest="command", required=True)
  plan = subparsers.add_parser(
    "plan",
    help="Internal pre-call route. Builds artifacts and stops before provider rendering.",
  )
  plan.add_argument("task_text")
  plan.add_argument("--repo-root", type=Path, default=repo_root)
  plan.add_argument(
    "--harness-root",
    type=Path,
    default=repo_root,
    help="Repository root used to resolve harness_global static context sources.",
  )
  plan.add_argument(
    "--target-repo-root",
    type=Path,
    default=repo_root,
    help="Repository root used to resolve target_repo static context sources.",
  )
  plan.add_argument(
    "--agent",
    nargs="?",
    const=str(harness_root / "agents" / "project_manager.agent.json"),
    default=None,
  )
  plan.add_argument(
    "--attach-static-context",
    action="store_true",
    help="Attach a compiled StaticContextPacket as supplementary context.",
  )
  plan.add_argument(
    "--manifest",
    type=Path,
    default=(
      harness_root
      / "project_spec"
      / "static_context_packet.manifest.json"
    ),
  )
  plan.add_argument(
    "--runs-root",
    type=Path,
    default=harness_root / "runs",
  )
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_argument_parser().parse_args(argv)

  try:
    run_directory, artifact_paths = build_pre_call_artifacts(
        task_text=args.task_text,
        repo_root=args.repo_root.resolve(),
        harness_root=args.harness_root.resolve(),
        target_repo_root=args.target_repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        runs_root=args.runs_root.resolve(),
        agent_path=Path(args.agent).resolve() if args.agent is not None else None,
        attach_static_context=args.attach_static_context,
      )
  except (
    AgentContextCompilationError,
    OSError,
    StaticContextCompilationError,
    TypeError,
    ValueError,
    ValidationError,
  ) as error:
    print(f"FAIL: {error}", file=sys.stderr)
    return 1

  print(f"PASS: Provider-neutral pre-call artifacts written to {run_directory}")
  for artifact_path in artifact_paths:
    print(f"- {artifact_path.name}")
  print("STOP: No provider payload was rendered and no model call was made.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
