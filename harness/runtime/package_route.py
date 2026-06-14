from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
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
from harness.contracts.project_manager_report import ProjectManagerReport
from harness.contracts.project_manager_report_extractor import (
  ProjectManagerReportExtractorError,
  extract_project_manager_report,
)
from harness.contracts.project_manager_report_validation import (
  ProjectManagerReportValidationArtifact,
  default_validation_artifact_path,
)
from harness.runtime.artifact_facts import sha256_file
from harness.runtime.api_call_ledger import (
  DEFAULT_RUNTIME_CALL_LEDGER_PATH,
  finalize_runtime_call_ledger,
)
from harness.providers.openai.openai_call_runner import (
  OpenAICallRunnerError,
  run_openai_call,
)
from harness.providers.openai.openai_raw_response import OpenAIRawResponse
from harness.providers.openai.openai_response_payload import OpenAIResponsePayload
from harness.providers.openai.openai_response_payload_compiler import (
  OpenAIResponsePayloadCompilationError,
  compile_openai_response_payload,
)
from harness.runtime.api_call_packet import ApiCallPacket
from harness.runtime.api_call_packet_builder import build_api_call_packet
from harness.runtime.git_context import collect_git_context
from harness.runtime.runtime_budget_policy import RuntimeBudgetPolicy
from harness.runtime.task import Task, task_from_cli

DEFAULT_PM_AGENT_PATH = Path(__file__).resolve().parents[1] / "agents" / "project_manager.agent.json"


@dataclass(slots=True)
class PackageRouteResult:
  selected_agent: Path
  run_directory: Path
  artifact_paths: list[Path]
  task: Task
  agent_context_packet: AgentContextPacket
  api_call_packet: ApiCallPacket
  provider_payload: OpenAIResponsePayload
  raw_model_response: OpenAIRawResponse
  report: ProjectManagerReport


class PackageRouteStepError(RuntimeError):
  def __init__(self, step: str, message: str) -> None:
    super().__init__(message)
    self.step = step


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


def _new_run_directory(runs_root: Path) -> Path:
  runs_root.mkdir(parents=True, exist_ok=True)
  timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
  base_name = f"{timestamp}-agent-route"
  candidate = runs_root / base_name
  if not candidate.exists():
    candidate.mkdir()
    return candidate

  while True:
    candidate = runs_root / f"{base_name}-{uuid.uuid4().hex[:8]}"
    if not candidate.exists():
      candidate.mkdir()
      return candidate


def _display_path(path: Path, repo_root: Path) -> str:
  try:
    return path.relative_to(repo_root).as_posix()
  except ValueError:
    return path.as_posix()


def _load_runtime_budget_policy(harness_root: Path) -> RuntimeBudgetPolicy:
  policy_path = harness_root / "runtime" / "runtime_budget.policy.json"
  return RuntimeBudgetPolicy.model_validate(_load_json(policy_path))


def _fail(step: str, error: BaseException) -> int:
  print(f"FAIL: {step}: {error}", file=sys.stderr)
  return 1


def run_package_route(
  *,
  route: str,
  task_text: str,
  agent_path: Path,
  runs_root: Path,
  repo_root: Path,
) -> PackageRouteResult:
  harness_root = Path(__file__).resolve().parents[1]
  task = task_from_cli(task_text.strip())
  run_directory = _new_run_directory(runs_root)
  task_path = run_directory / "task.json"
  static_context_path = run_directory / "static_context_packet.json"
  repo_snapshot_path = run_directory / "repo_snapshot_packet.json"
  agent_context_path = run_directory / "agent_context_packet.json"
  api_call_path = run_directory / "api_call_packet.json"
  provider_payload_path = run_directory / "provider_payload.json"
  raw_model_response_path = run_directory / "raw_model_response.json"
  report_path = run_directory / "project_manager_report.json"
  validation_path = default_validation_artifact_path(report_path)
  schema_path = harness_root / "contracts" / "ProjectManagerReport.schema.json"

  _write_json(task_path, task.model_dump(mode="json", by_alias=True))

  runtime_budget = _load_runtime_budget_policy(harness_root)
  git_context = collect_git_context(repo_root.resolve())

  try:
    agent_context_packet = compile_agent_context_packet(
      agent_path=agent_path.resolve(),
      output_path=agent_context_path,
      manifest_path=harness_root / "project_spec" / "static_context_packet.manifest.json",
      repo_root=repo_root.resolve(),
      harness_root=harness_root.parent.resolve(),
      target_repo_root=repo_root.resolve(),
      static_context_output_path=static_context_path,
      repo_snapshot_output_path=repo_snapshot_path,
    )
  except (
    AgentContextCompilationError,
    OSError,
    TypeError,
    ValidationError,
    ValueError,
  ) as error:
    raise PackageRouteStepError("compile_agent_context_packet", str(error)) from error

  artifact_paths: list[Path] = [task_path, static_context_path]
  if repo_snapshot_path.exists():
    artifact_paths.append(repo_snapshot_path)
  artifact_paths.append(agent_context_path)

  try:
    api_call_packet = build_api_call_packet(
      task=task,
      call_mode="agent_routed",
      agent_context_packet=agent_context_packet,
      runtime_budget=runtime_budget,
      git_context=git_context,
      output_path=api_call_path,
    )
  except (OSError, TypeError, ValidationError, ValueError) as error:
    raise PackageRouteStepError("build_api_call_packet", str(error)) from error

  artifact_paths.append(api_call_path)

  provider = agent_context_packet.agent_contract.provider
  if provider != "openai":
    raise PackageRouteStepError(
      "render_provider_payload",
      (
        f"provider '{provider}' is declared by the selected agent, but no "
        f"provider runner is implemented for it."
      ),
    )

  try:
    provider_payload = compile_openai_response_payload(
      api_call_packet_path=api_call_path,
      output_path=provider_payload_path,
    )
  except (
    OpenAIResponsePayloadCompilationError,
    OSError,
    TypeError,
    ValidationError,
    ValueError,
  ) as error:
    raise PackageRouteStepError("render_provider_payload", str(error)) from error

  artifact_paths.append(provider_payload_path)

  try:
    raw_model_response = run_openai_call(
      provider_payload_path=provider_payload_path,
      output_path=raw_model_response_path,
      payload=provider_payload,
    )
  except (OpenAICallRunnerError, OSError, TypeError, ValidationError, ValueError) as error:
    raise PackageRouteStepError("run_provider", str(error)) from error

  artifact_paths.append(raw_model_response_path)

  output_policy = agent_context_packet.agent_contract.agent_output_policy
  if len(output_policy) != 1 or output_policy[0].output_id != "project_manager_report":
    raise PackageRouteStepError(
      "extract_project_manager_report",
      "Selected agent output policy is not supported by the current package route.",
    )

  try:
    report = extract_project_manager_report(
      raw_response_path=raw_model_response_path,
      schema_path=schema_path,
      output_path=report_path,
    )
  except (
    ProjectManagerReportExtractorError,
    OSError,
    TypeError,
    ValidationError,
    ValueError,
  ) as error:
    raise PackageRouteStepError("extract_project_manager_report", str(error)) from error

  artifact_paths.append(report_path)

  try:
    validation_artifact = ProjectManagerReportValidationArtifact.model_validate(
      _load_json(validation_path)
    )
  except (OSError, TypeError, ValidationError, ValueError) as error:
    raise PackageRouteStepError(
      "validate_project_manager_report_validation_artifact",
      str(error),
    ) from error

  if not validation_artifact.validation_passed:
    raise PackageRouteStepError(
      "validate_project_manager_report_validation_artifact",
      "Validation artifact reports failure.",
    )

  if validation_artifact.report_artifact_path != report_path.as_posix():
    raise PackageRouteStepError(
      "validate_project_manager_report_validation_artifact",
      "Validation artifact report_artifact_path does not match the report path.",
    )

  if validation_artifact.schema_path != schema_path.as_posix():
    raise PackageRouteStepError(
      "validate_project_manager_report_validation_artifact",
      "Validation artifact schema_path does not match the PM schema path.",
    )

  if validation_artifact.schema_name != provider_payload.request.text.format.name:
    raise PackageRouteStepError(
      "validate_project_manager_report_validation_artifact",
      "Validation artifact schema_name does not match the rendered output schema name.",
    )

  if validation_artifact.report_status != report.report_status:
    raise PackageRouteStepError(
      "validate_project_manager_report_validation_artifact",
      "Validation artifact report_status does not match the validated report.",
    )

  if validation_artifact.proof_frontier_blocked != report.proof_frontier.blocked:
    raise PackageRouteStepError(
      "validate_project_manager_report_validation_artifact",
      "Validation artifact proof_frontier_blocked does not match the validated report.",
    )

  report_artifact_sha256 = sha256_file(report_path)
  validation_artifact_sha256 = sha256_file(validation_path)
  schema_sha256 = sha256_file(schema_path)

  if validation_artifact.report_artifact_sha256 != report_artifact_sha256:
    raise PackageRouteStepError(
      "validate_project_manager_report_validation_artifact",
      "Validation artifact report hash does not match the report file.",
    )

  if validation_artifact.schema_sha256 != schema_sha256:
    raise PackageRouteStepError(
      "validate_project_manager_report_validation_artifact",
      "Validation artifact schema hash does not match the PM schema file.",
    )

  artifact_paths.append(validation_path)

  try:
    finalize_runtime_call_ledger(
      route=route,
      agent=_display_path(agent_path.resolve(), repo_root.resolve()),
      model=provider_payload.request.model,
      schema_name=provider_payload.request.text.format.name,
      context_packet_sha256=sha256_file(agent_context_path),
      validation_passed=validation_artifact.validation_passed,
      provider_response=raw_model_response,
      contract_status=report.report_status,
      output_artifact_path=_display_path(report_path, repo_root.resolve()),
      report_artifact_path=_display_path(report_path, repo_root.resolve()),
      report_artifact_sha256=report_artifact_sha256,
      validation_artifact_path=_display_path(validation_path, repo_root.resolve()),
      validation_artifact_sha256=validation_artifact_sha256,
      git_commit=git_context.commit,
      worktree_dirty=git_context.is_dirty,
      path=DEFAULT_RUNTIME_CALL_LEDGER_PATH,
    )
  except RuntimeError as error:
    raise PackageRouteStepError("finalize_runtime_call_ledger", str(error)) from error

  return PackageRouteResult(
    selected_agent=agent_path.resolve(),
    run_directory=run_directory,
    artifact_paths=artifact_paths,
    task=task,
    agent_context_packet=agent_context_packet,
    api_call_packet=api_call_packet,
    provider_payload=provider_payload,
    raw_model_response=raw_model_response,
    report=report,
  )


def build_argument_parser() -> argparse.ArgumentParser:
  parser = _build_base_argument_parser(
    "Run the package-level selected-agent orchestration route.",
  )
  parser.add_argument(
    "--agent",
    type=Path,
    default=None,
    help="Path to the selected .agent.json contract for the generic lower-level agent route.",
  )
  return parser


def build_plan_argument_parser() -> argparse.ArgumentParser:
  return _build_base_argument_parser(
    "Run the Project Manager plan alias over the selected PM agent route.",
  )


def _build_base_argument_parser(description: str) -> argparse.ArgumentParser:
  script_path = Path(__file__).resolve()
  harness_root = script_path.parents[1]
  repo_root = script_path.parents[2]

  parser = argparse.ArgumentParser(description=description)
  parser.add_argument("task_text", nargs="?", default=None)
  parser.add_argument(
    "--runs-root",
    type=Path,
    default=harness_root / "runs",
    help="Directory that will receive per-run artifacts.",
  )
  parser.add_argument(
    "--repo-root",
    type=Path,
    default=repo_root,
    help="Root of the repo used for git context and repo snapshot resolution.",
  )
  return parser


def _run_cli_route(
  *,
  route_name: str,
  route: str,
  task_text: str,
  agent_path: Path,
  runs_root: Path,
  repo_root: Path,
) -> int:
  try:
    result = run_package_route(
      route=route,
      task_text=task_text,
      agent_path=agent_path.resolve(),
      runs_root=runs_root.resolve(),
      repo_root=repo_root.resolve(),
    )
  except PackageRouteStepError as error:
    return _fail(error.step, error)
  except (
    OSError,
    TypeError,
    ValidationError,
    ValueError,
  ) as error:
    return _fail("package_cli", error)

  _print_route_result(route_name=route_name, result=result, repo_root=repo_root)
  return 0


def _print_route_result(
  *,
  route_name: str,
  result: PackageRouteResult,
  repo_root: Path,
) -> None:
  print(f"PASS: {route_name} route completed.")
  print(f"Selected agent: {_display_path(result.selected_agent, repo_root.resolve())}")
  print(f"Provider: {result.agent_context_packet.agent_contract.provider}")
  print(f"Model: {result.agent_context_packet.agent_contract.model}")
  print(f"Run directory: {_display_path(result.run_directory, repo_root.resolve())}")
  print("Artifacts:")
  for artifact_path in result.artifact_paths:
    print(f"  {artifact_path.name}")
  print(f"Report status: {result.report.report_status}")
  print(f"Blocked: {result.report.proof_frontier.blocked}")


def _run_plan_route(argv: list[str]) -> int:
  args = build_plan_argument_parser().parse_args(argv)

  if args.task_text is None or not args.task_text.strip():
    print("FAIL: package_cli: package CLI requires task text.", file=sys.stderr)
    return 1

  return _run_cli_route(
    route_name="Plan",
    route="plan",
    task_text=args.task_text,
    agent_path=DEFAULT_PM_AGENT_PATH,
    runs_root=args.runs_root,
    repo_root=args.repo_root,
  )


def _run_generic_agent_route(argv: list[str]) -> int:
  args = build_argument_parser().parse_args(argv)

  if args.task_text is None or not args.task_text.strip():
    print("FAIL: package_cli: package CLI requires task text.", file=sys.stderr)
    return 1

  if args.agent is None:
    print("FAIL: package_cli: package CLI requires --agent.", file=sys.stderr)
    return 1

  return _run_cli_route(
    route_name="Agent",
    route="agent",
    task_text=args.task_text,
    agent_path=args.agent,
    runs_root=args.runs_root,
    repo_root=args.repo_root,
  )


def main(argv: list[str] | None = None) -> int:
  argv_list = list(sys.argv[1:] if argv is None else argv)
  if argv_list and argv_list[0] == "plan":
    return _run_plan_route(argv_list[1:])
  return _run_generic_agent_route(argv_list)


if __name__ == "__main__":
  raise SystemExit(main())
