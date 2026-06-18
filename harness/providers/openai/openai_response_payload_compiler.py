from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Support direct execution from harness/providers/openai while preserving package imports.
if __package__ in {None, ""}:
  sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pydantic import ValidationError

from harness.runtime.api_call_packet import ApiCallPacket
from harness.runtime.provider_runtime_policy import ProviderRuntimePolicy
from harness.providers.openai.openai_response_payload import (
  OpenAIResponsePayload,
  OpenAIResponsePayloadMetadata,
  OpenAIResponseRequest,
  PayloadInputText,
  PayloadMessage,
  PayloadTextFormat,
  PayloadTextFormatSchema,
)


class OpenAIResponsePayloadCompilationError(RuntimeError):
  pass


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


def _json_block(value: Any) -> str:
  return json.dumps(value, indent=2, ensure_ascii=False)


SCHEMA_COMPATIBILITY_STRIP_KEYS = {
  "$schema",
  "$id",
}


UNSUPPORTED_STRUCTURED_OUTPUT_KEYWORDS = {
  "allOf",
  "not",
  "dependentRequired",
  "dependentSchemas",
  "if",
  "then",
  "else",
}


def _find_unsupported_schema_keywords(
  value: Any,
  *,
  path: str = "$",
) -> list[str]:
  findings: list[str] = []

  if isinstance(value, dict):
    for key, nested_value in value.items():
      child_path = f"{path}.{key}"
      if key in UNSUPPORTED_STRUCTURED_OUTPUT_KEYWORDS:
        findings.append(child_path)
      findings.extend(
        _find_unsupported_schema_keywords(nested_value, path=child_path)
      )
  elif isinstance(value, list):
    for index, nested_value in enumerate(value):
      findings.extend(
        _find_unsupported_schema_keywords(nested_value, path=f"{path}[{index}]")
      )

  return findings


def _strip_schema_compatibility_keys(value: Any) -> Any:
  if isinstance(value, dict):
    return {
      key: _strip_schema_compatibility_keys(nested_value)
      for key, nested_value in value.items()
      if key not in SCHEMA_COMPATIBILITY_STRIP_KEYS
    }
  if isinstance(value, list):
    return [_strip_schema_compatibility_keys(nested_value) for nested_value in value]
  return value


def _sanitize_output_schema(schema: dict[str, Any]) -> dict[str, Any]:
  cleaned = _strip_schema_compatibility_keys(schema)
  unsupported_paths = _find_unsupported_schema_keywords(cleaned)
  if unsupported_paths:
    raise OpenAIResponsePayloadCompilationError(
      "Output schema contains unsupported Structured Outputs keywords at: "
      + ", ".join(unsupported_paths)
    )
  return cleaned


def _camel_to_snake(name: str) -> str:
  snake = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
  return snake.replace(".schema", "").replace(".json", "")


def _derive_output_schema_path(api_call_packet: ApiCallPacket, harness_root: Path) -> Path:
  if api_call_packet.agent_context_packet is None:
    raise OpenAIResponsePayloadCompilationError(
      "Direct payload compilation requires --output-schema."
    )

  output_policies = api_call_packet.agent_context_packet.agent_contract.agent_output_policy
  if len(output_policies) != 1:
    raise OpenAIResponsePayloadCompilationError(
      "Could not derive output schema from agent output policy."
    )

  schema_ref = output_policies[0].schema_ref
  return (harness_root / "agents" / schema_ref).resolve()


def _derive_output_schema_name(
  api_call_packet: ApiCallPacket,
  output_schema_path: Path,
) -> str:
  if api_call_packet.agent_context_packet is not None:
    output_policies = api_call_packet.agent_context_packet.agent_contract.agent_output_policy
    if len(output_policies) == 1:
      return output_policies[0].output_id

  schema_name = output_schema_path.name
  if schema_name.endswith(".schema.json"):
    schema_name = schema_name[:-12]
  elif schema_name.endswith(".json"):
    schema_name = schema_name[:-5]
  return _camel_to_snake(schema_name)


def _resolve_request_model(
  *,
  api_call_packet: ApiCallPacket,
  api_call_packet_path: Path,
  direct_model: str | None,
  provider_runtime_policy_path: Path | None,
) -> tuple[str, list[str], list[str]]:
  if api_call_packet.call_mode == "agent_routed":
    if api_call_packet.agent_context_packet is None:
      raise OpenAIResponsePayloadCompilationError(
        "agent_routed ApiCallPacket requires agent_context_packet."
      )

    provider = api_call_packet.agent_context_packet.agent_contract.provider
    if provider != "openai":
      raise OpenAIResponsePayloadCompilationError(
        "OpenAI payload compiler requires api_call_packet.agent_context_packet.agent_contract.provider == 'openai'."
      )

    model = api_call_packet.agent_context_packet.agent_contract.model
    if not model:
      raise OpenAIResponsePayloadCompilationError(
        "Agent contract model is missing for agent-routed call."
      )

    return (
      model,
      [api_call_packet_path.name],
      [
        "Provider sourced directly from api_call_packet.agent_context_packet.agent_contract.provider.",
        "Model sourced directly from api_call_packet.agent_context_packet.agent_contract.model.",
      ],
    )

  if direct_model:
    return (
      direct_model,
      [api_call_packet_path.name],
      [
        "Direct-call model sourced from explicit --model.",
      ],
    )

  if provider_runtime_policy_path is not None:
    provider_runtime_policy = ProviderRuntimePolicy.model_validate(
      _load_json(provider_runtime_policy_path)
    )
    default_direct_model = provider_runtime_policy.default_direct_model
    if default_direct_model.provider != "openai":
      raise OpenAIResponsePayloadCompilationError(
        "OpenAI payload compiler requires provider_runtime_policy.default_direct_model.provider == 'openai'."
      )

    return (
      default_direct_model.model,
      [api_call_packet_path.name, provider_runtime_policy_path.name],
      [
        "Direct-call model sourced from provider_runtime_policy.default_direct_model.",
      ],
    )

  raise OpenAIResponsePayloadCompilationError(
    "Direct payload compilation requires --model or --provider-runtime-policy."
  )


def _render_developer_message(api_call_packet: ApiCallPacket) -> str:
  sections: list[str] = []

  if api_call_packet.agent_context_packet is not None:
    agent_context_packet = api_call_packet.agent_context_packet
    sections.extend(
      [
        "AGENT CONTRACT",
        _json_block(agent_context_packet.agent_contract.model_dump(mode="json", by_alias=True)),
        "",
        "RESOLVED INPUT COVERAGE",
        _json_block([entry.model_dump(mode="json") for entry in agent_context_packet.input_coverage]),
      ]
    )

    static_context_packet = agent_context_packet.resolved_inputs.static_context_packet
    if static_context_packet is not None:
      sections.extend(
        [
          "",
          "STATIC CONTEXT PACKET",
          _json_block(static_context_packet.model_dump(mode="json")),
        ]
      )

    repo_snapshot_packet = agent_context_packet.resolved_inputs.repo_snapshot_packet
    if repo_snapshot_packet is not None:
      sections.extend(
        [
          "",
          "REPO SNAPSHOT PACKET",
          _json_block(repo_snapshot_packet.model_dump(mode="json")),
        ]
      )
  else:
    sections.extend(
      [
        "DIRECT CALL FRAMING",
        "This payload was rendered from a direct ApiCallPacket with no agent contract.",
      ]
    )

  if api_call_packet.supplementary_context:
    sections.extend(
      [
        "",
        "SUPPLEMENTARY CONTEXT",
        _json_block(
          [entry.model_dump(mode="json") for entry in api_call_packet.supplementary_context]
        ),
      ]
    )

  if api_call_packet.git_context is not None:
    sections.extend(
      [
        "",
        "GIT CONTEXT",
        _json_block(api_call_packet.git_context.model_dump(mode="json")),
      ]
    )

  if api_call_packet.git_delta_context is not None:
    sections.extend(
      [
        "",
        "GIT DELTA CONTEXT",
        (
          "Treat this supplied git_delta_context as the run's git context "
          "source with delta semantics. Until report schemas are migrated, "
          "source_coverage.git_context covers either git_context or "
          "git_delta_context, depending on which packet lane is present."
        ),
        _json_block(api_call_packet.git_delta_context.model_dump(mode="json")),
      ]
    )

  return "\n".join(sections)


def _render_user_message(api_call_packet: ApiCallPacket) -> str:
  return "\n".join(
    [
      "TASK",
      _json_block(api_call_packet.task.model_dump(mode="json")),
    ]
  )


def compile_openai_response_payload(
  *,
  api_call_packet_path: Path,
  output_path: Path,
  output_schema_path: Path | None = None,
  direct_model: str | None = None,
  provider_runtime_policy_path: Path | None = None,
) -> OpenAIResponsePayload:
  script_path = Path(__file__).resolve()
  harness_root = script_path.parents[2]

  api_call_packet = ApiCallPacket.model_validate(_load_json(api_call_packet_path))
  request_model, model_source_artifacts, model_basis = _resolve_request_model(
    api_call_packet=api_call_packet,
    api_call_packet_path=api_call_packet_path,
    direct_model=direct_model,
    provider_runtime_policy_path=provider_runtime_policy_path,
  )

  resolved_output_schema_path = (
    output_schema_path.resolve()
    if output_schema_path is not None
    else _derive_output_schema_path(api_call_packet, harness_root)
  )
  output_schema = _load_json(resolved_output_schema_path)
  output_schema_name = _derive_output_schema_name(
    api_call_packet,
    resolved_output_schema_path,
  )
  embedded_schema = _sanitize_output_schema(output_schema)

  max_output_tokens = None
  if api_call_packet.runtime_budget is not None:
    max_output_tokens = api_call_packet.runtime_budget.default.reserved_output_tokens

  payload = OpenAIResponsePayload(
    metadata=OpenAIResponsePayloadMetadata(
      document_id="provider_payload.json",
      title="OpenAI Response Payload",
      purpose=(
        "Provider-specific OpenAI Responses API payload rendered from a "
        "provider-neutral ApiCallPacket."
      ),
      source_format="json",
      document_authority="compiled_runtime_artifact",
    ),
    provider="openai",
    endpoint="responses.create",
    request=OpenAIResponseRequest(
      model=request_model,
      input=[
        PayloadMessage(
          role="developer",
          content=[
            PayloadInputText(
              type="input_text",
              text=_render_developer_message(api_call_packet),
            )
          ],
        ),
        PayloadMessage(
          role="user",
          content=[
            PayloadInputText(
              type="input_text",
              text=_render_user_message(api_call_packet),
            )
          ],
        ),
      ],
      text=PayloadTextFormat(
        format=PayloadTextFormatSchema(
          type="json_schema",
          name=output_schema_name,
          strict=True,
          schema_=embedded_schema,
        )
      ),
      max_output_tokens=max_output_tokens,
      tools=[],
    ),
    source_artifacts=[
      *model_source_artifacts,
      resolved_output_schema_path.name,
    ],
    basis=[
      "Rendered OpenAI Responses API payload from provider-neutral ApiCallPacket.",
      *model_basis,
      f"Structured output schema loaded from {resolved_output_schema_path.name}.",
      "No model call was performed.",
    ],
  )

  _write_json(output_path, payload.model_dump(mode="json", by_alias=True))
  return OpenAIResponsePayload.model_validate(_load_json(output_path))


def build_argument_parser() -> argparse.ArgumentParser:
  script_path = Path(__file__).resolve()
  harness_root = script_path.parents[2]

  parser = argparse.ArgumentParser(
    description="Render an OpenAI Responses API payload artifact from existing harness artifacts.",
  )
  parser.add_argument(
    "--api-call-packet",
    type=Path,
    required=True,
    help="Path to ApiCallPacket JSON artifact.",
  )
  parser.add_argument(
    "--model",
    type=str,
    default=None,
    help="Explicit direct-call model. Ignored for agent-routed packets.",
  )
  parser.add_argument(
    "--provider-runtime-policy",
    type=Path,
    default=None,
    help="Optional ProviderRuntimePolicy JSON used only to supply default_direct_model for direct calls.",
  )
  parser.add_argument(
    "--output-schema",
    type=Path,
    default=None,
    help="Optional explicit output schema path. Required for direct calls.",
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=harness_root / "runs" / "provider_payload.json",
    help="Destination for the emitted OpenAI provider payload JSON.",
  )
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_argument_parser().parse_args(argv)

  try:
    payload = compile_openai_response_payload(
      api_call_packet_path=args.api_call_packet.resolve(),
      output_schema_path=(
        args.output_schema.resolve()
        if args.output_schema is not None
        else None
      ),
      direct_model=args.model,
      provider_runtime_policy_path=(
        args.provider_runtime_policy.resolve()
        if args.provider_runtime_policy is not None
        else None
      ),
      output_path=args.output.resolve(),
    )
  except (
    OpenAIResponsePayloadCompilationError,
    OSError,
    TypeError,
    ValidationError,
    ValueError,
  ) as error:
    print(f"FAIL: {error}", file=sys.stderr)
    return 1

  print(f"PASS: OpenAI provider payload written to {args.output.resolve()}")
  print(f"Model: {payload.request.model}")
  print(f"Endpoint: {payload.endpoint}")
  print(f"Input messages: {len(payload.request.input)}")
  print(f"Tools: {len(payload.request.tools)}")
  print(f"Structured output: {payload.request.text.format.name}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
