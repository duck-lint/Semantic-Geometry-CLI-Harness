from __future__ import annotations

import importlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from harness.agents.agent_context_compiler import compile_agent_context_packet
from harness.providers.openai.openai_response_payload import OpenAIResponsePayload
from harness.providers.openai.openai_response_payload_compiler import (
  OpenAIResponsePayloadCompilationError,
  compile_openai_response_payload,
)
from harness.runtime.api_call_packet_builder import build_api_call_packet
from harness.runtime.api_call_ledger import DEFAULT_RUNTIME_CALL_LEDGER_PATH
from harness.runtime.runtime_budget_policy import RuntimeBudgetPolicy
from harness.runtime.task import task_from_cli


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = REPO_ROOT / "harness"
MANIFEST_PATH = HARNESS_ROOT / "project_spec" / "static_context_packet.manifest.json"
AGENT_PATH = HARNESS_ROOT / "agents" / "project_manager.agent.json"
PM_SCHEMA_PATH = HARNESS_ROOT / "contracts" / "ProjectManagerReport.schema.json"
PAYLOAD_SCHEMA_PATH = (
  HARNESS_ROOT / "providers" / "openai" / "OpenAIResponsePayload.schema.json"
)
RUNTIME_BUDGET_PATH = HARNESS_ROOT / "runtime" / "runtime_budget.policy.json"
PROVIDER_RUNTIME_POLICY_PATH = HARNESS_ROOT / "runtime" / "provider_runtime.policy.json"


def load_json(path: Path) -> dict:
  return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def ensure_ledger_artifact() -> None:
  ledger_path = DEFAULT_RUNTIME_CALL_LEDGER_PATH
  ledger_path.parent.mkdir(parents=True, exist_ok=True)
  ledger_path.write_text("", encoding="utf-8")


def build_agent_routed_api_call_packet(
  temp_root: Path,
  *,
  runtime_budget: RuntimeBudgetPolicy | None = None,
):
  ensure_ledger_artifact()
  try:
    agent_context_packet = compile_agent_context_packet(
      agent_path=AGENT_PATH,
      output_path=temp_root / "agent_context_packet.json",
      manifest_path=MANIFEST_PATH,
      harness_root=REPO_ROOT,
      target_repo_root=REPO_ROOT,
      static_context_output_path=temp_root / "static_context_packet.json",
    )
    packet = build_api_call_packet(
      task=task_from_cli("Review the current project trajectory."),
      call_mode="agent_routed",
      agent_context_packet=agent_context_packet,
      runtime_budget=runtime_budget,
      output_path=temp_root / "api_call_packet.json",
    )
    return packet
  finally:
    ledger_path = DEFAULT_RUNTIME_CALL_LEDGER_PATH
    if ledger_path.exists():
      ledger_path.unlink()
      try:
        ledger_path.parent.rmdir()
      except OSError:
        pass


def build_direct_api_call_packet(temp_root: Path):
  return build_api_call_packet(
    task=task_from_cli("Direct task payload."),
    call_mode="direct",
    output_path=temp_root / "api_call_packet.json",
  )


class OpenAIResponsePayloadCompilerTests(unittest.TestCase):
  def test_payload_compiler_emits_provider_payload_from_agent_routed_packet(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      build_agent_routed_api_call_packet(temp_root)
      output_path = temp_root / "provider_payload.json"

      payload = compile_openai_response_payload(
        api_call_packet_path=temp_root / "api_call_packet.json",
        output_path=output_path,
      )

      self.assertTrue(output_path.is_file())
      self.assertIsInstance(payload, OpenAIResponsePayload)
      self.assertEqual(payload.provider, "openai")
      self.assertEqual(payload.endpoint, "responses.create")

  def test_payload_compiler_uses_agent_contract_model_for_agent_routed_call(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      packet = build_agent_routed_api_call_packet(temp_root)
      expected_model = packet.agent_context_packet.agent_contract.model

      payload = compile_openai_response_payload(
        api_call_packet_path=temp_root / "api_call_packet.json",
        output_path=temp_root / "provider_payload.json",
      )

      self.assertEqual(payload.request.model, expected_model)
      self.assertIn(
        "Provider sourced directly from api_call_packet.agent_context_packet.agent_contract.provider.",
        payload.basis,
      )

  def test_openai_payload_compiler_rejects_non_openai_agent_provider(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      agent_path = temp_root / "anthropic_pm.agent.json"
      agent_data = load_json(AGENT_PATH)
      agent_data["provider"] = "anthropic"
      agent_path.write_text(json.dumps(agent_data, indent=2) + "\n", encoding="utf-8")

      ensure_ledger_artifact()
      ledger_path = DEFAULT_RUNTIME_CALL_LEDGER_PATH
      try:
        agent_context_packet = compile_agent_context_packet(
          agent_path=agent_path,
          output_path=temp_root / "agent_context_packet.json",
          manifest_path=MANIFEST_PATH,
          harness_root=REPO_ROOT,
          target_repo_root=REPO_ROOT,
          static_context_output_path=temp_root / "static_context_packet.json",
        )
        build_api_call_packet(
          task=task_from_cli("Review the current project trajectory."),
          call_mode="agent_routed",
          agent_context_packet=agent_context_packet,
          output_path=temp_root / "api_call_packet.json",
        )
      finally:
        if ledger_path.exists():
          ledger_path.unlink()
          try:
            ledger_path.parent.rmdir()
          except OSError:
            pass

      with self.assertRaises(OpenAIResponsePayloadCompilationError) as error:
        compile_openai_response_payload(
          api_call_packet_path=temp_root / "api_call_packet.json",
          output_path=temp_root / "provider_payload.json",
        )

      self.assertIn("agent_contract.provider == 'openai'", str(error.exception))

  def test_payload_compiler_embeds_strict_json_schema_output_format(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      build_agent_routed_api_call_packet(temp_root)

      payload = compile_openai_response_payload(
        api_call_packet_path=temp_root / "api_call_packet.json",
        output_path=temp_root / "provider_payload.json",
      )

      self.assertEqual(payload.request.text.format.type, "json_schema")
      self.assertTrue(payload.request.text.format.strict)
      self.assertNotIn("$schema", payload.request.text.format.schema_)
      self.assertNotIn("$id", payload.request.text.format.schema_)

  def test_payload_compiler_uses_project_manager_report_schema_for_pm(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      build_agent_routed_api_call_packet(temp_root)
      expected_schema = load_json(PM_SCHEMA_PATH)
      required = expected_schema["$defs"]["ReportSourceCoverage"]["required"]

      payload = compile_openai_response_payload(
        api_call_packet_path=temp_root / "api_call_packet.json",
        output_path=temp_root / "provider_payload.json",
      )

      self.assertEqual(payload.request.text.format.name, "project_manager_report")
      self.assertEqual(
        payload.request.text.format.schema_,
        {
          k: v
          for k, v in expected_schema.items()
          if k not in {"$schema", "$id"}
        },
      )
      self.assertIn("repo_snapshot_packet", required)

  def test_payload_compiler_emits_responses_api_message_items(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      build_agent_routed_api_call_packet(temp_root)

      payload = compile_openai_response_payload(
        api_call_packet_path=temp_root / "api_call_packet.json",
        output_path=temp_root / "provider_payload.json",
      )

      developer_message = payload.request.input[0]
      user_message = payload.request.input[1]
      self.assertEqual(developer_message.type, "message")
      self.assertEqual(user_message.type, "message")
      self.assertEqual(developer_message.content[0].type, "input_text")
      self.assertEqual(user_message.content[0].type, "input_text")

  def test_payload_compiler_puts_agent_and_context_framing_in_developer_message(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      build_agent_routed_api_call_packet(temp_root)

      payload = compile_openai_response_payload(
        api_call_packet_path=temp_root / "api_call_packet.json",
        output_path=temp_root / "provider_payload.json",
      )

      developer_message = payload.request.input[0]
      developer_text = developer_message.content[0].text
      self.assertEqual(developer_message.role, "developer")
      self.assertIn("AGENT CONTRACT", developer_text)
      self.assertIn("RESOLVED INPUT COVERAGE", developer_text)
      self.assertIn("STATIC CONTEXT PACKET", developer_text)

  def test_payload_compiler_puts_task_authority_in_user_message(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      build_agent_routed_api_call_packet(temp_root)

      payload = compile_openai_response_payload(
        api_call_packet_path=temp_root / "api_call_packet.json",
        output_path=temp_root / "provider_payload.json",
      )

      user_message = payload.request.input[1]
      developer_text = payload.request.input[0].content[0].text
      user_text = user_message.content[0].text
      self.assertEqual(user_message.role, "user")
      self.assertIn("TASK", user_text)
      self.assertIn("Review the current project trajectory.", user_text)
      self.assertNotIn("\nTASK\n", developer_text)

  def test_payload_compiler_does_not_include_tools(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      build_agent_routed_api_call_packet(temp_root)

      payload = compile_openai_response_payload(
        api_call_packet_path=temp_root / "api_call_packet.json",
        output_path=temp_root / "provider_payload.json",
      )

      self.assertEqual(payload.request.tools, [])

  def test_payload_compiler_maps_runtime_budget_reserved_output_tokens(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      runtime_budget = RuntimeBudgetPolicy.model_validate(load_json(RUNTIME_BUDGET_PATH))
      build_agent_routed_api_call_packet(temp_root, runtime_budget=runtime_budget)

      payload = compile_openai_response_payload(
        api_call_packet_path=temp_root / "api_call_packet.json",
        output_path=temp_root / "provider_payload.json",
      )

      self.assertEqual(
        payload.request.max_output_tokens,
        runtime_budget.default.reserved_output_tokens,
      )

  def test_payload_compiler_does_not_select_model_from_runtime_budget(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      runtime_budget = RuntimeBudgetPolicy.model_validate(load_json(RUNTIME_BUDGET_PATH))
      packet = build_agent_routed_api_call_packet(temp_root, runtime_budget=runtime_budget)

      payload = compile_openai_response_payload(
        api_call_packet_path=temp_root / "api_call_packet.json",
        output_path=temp_root / "provider_payload.json",
      )

      self.assertEqual(
        payload.request.model,
        packet.agent_context_packet.agent_contract.model,
      )

  def test_agent_routed_payload_does_not_require_provider_runtime_policy(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      packet = build_agent_routed_api_call_packet(temp_root)

      payload = compile_openai_response_payload(
        api_call_packet_path=temp_root / "api_call_packet.json",
        output_path=temp_root / "provider_payload.json",
      )

      self.assertEqual(payload.request.model, packet.agent_context_packet.agent_contract.model)

  def test_agent_routed_payload_does_not_require_effective_model_selection_artifact(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      build_agent_routed_api_call_packet(temp_root)

      payload = compile_openai_response_payload(
        api_call_packet_path=temp_root / "api_call_packet.json",
        output_path=temp_root / "provider_payload.json",
      )

      self.assertNotIn("effective_model_selection.json", payload.source_artifacts)

  def test_direct_payload_requires_model_source(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      build_direct_api_call_packet(temp_root)

      with self.assertRaises(OpenAIResponsePayloadCompilationError) as error:
        compile_openai_response_payload(
          api_call_packet_path=temp_root / "api_call_packet.json",
          output_schema_path=PM_SCHEMA_PATH,
          output_path=temp_root / "provider_payload.json",
        )

      self.assertIn("--model or --provider-runtime-policy", str(error.exception))

  def test_payload_compiler_does_not_import_openai_client(self) -> None:
    if "openai" in sys.modules:
      del sys.modules["openai"]

    module = importlib.import_module(
      "harness.providers.openai.openai_response_payload_compiler"
    )

    self.assertNotIn("openai", sys.modules)
    self.assertNotIn("import openai", inspect.getsource(module))

  def test_payload_compiler_focused_no_client_and_no_runner_artifacts(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      build_agent_routed_api_call_packet(temp_root)
      output_path = temp_root / "provider_payload.json"
      script_path = (
        HARNESS_ROOT
        / "providers"
        / "openai"
        / "openai_response_payload_compiler.py"
      )

      if "openai" in sys.modules:
        del sys.modules["openai"]

      completed = subprocess.run(
        [
          sys.executable,
          str(script_path),
          "--api-call-packet",
          str(temp_root / "api_call_packet.json"),
          "--output",
          str(output_path),
        ],
        cwd=script_path.parent,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
      )

      source = script_path.read_text(encoding="utf-8")
      self.assertEqual(completed.returncode, 0, completed.stderr)
      self.assertTrue(output_path.is_file())
      self.assertFalse((temp_root / "raw_model_response.json").exists())
      self.assertFalse((temp_root / "project_manager_report.json").exists())
      self.assertNotIn("openai", sys.modules)
      self.assertNotIn("import openai", source)
      self.assertNotIn("OpenAI(", source)
      self.assertNotIn("client.responses.create", source)

  def test_payload_compiler_script_does_not_create_raw_response_or_report(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      build_agent_routed_api_call_packet(temp_root)
      output_path = temp_root / "provider_payload.json"
      script_path = (
        HARNESS_ROOT
        / "providers"
        / "openai"
        / "openai_response_payload_compiler.py"
      )
      openai_loaded_before = "openai" in sys.modules

      completed = subprocess.run(
        [
          sys.executable,
          str(script_path),
          "--api-call-packet",
          str(temp_root / "api_call_packet.json"),
          "--output",
          str(output_path),
        ],
        cwd=script_path.parent,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
      )

      self.assertEqual(completed.returncode, 0, completed.stderr)
      self.assertTrue(output_path.is_file())
      self.assertFalse((temp_root / "raw_model_response.json").exists())
      self.assertFalse((temp_root / "project_manager_report.json").exists())
      self.assertEqual("openai" in sys.modules, openai_loaded_before)

  def test_payload_compiler_rejects_unsupported_structured_output_keywords(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      build_agent_routed_api_call_packet(temp_root)
      unsupported_schema_path = temp_root / "unsupported.schema.json"
      write_json(
        unsupported_schema_path,
        {
          "$schema": "https://json-schema.org/draft/2020-12/schema",
          "type": "object",
          "properties": {
            "value": {
              "allOf": [
                {"type": "string"},
                {"minLength": 1},
              ]
            }
          },
          "required": ["value"],
          "additionalProperties": False,
        },
      )

      with self.assertRaises(OpenAIResponsePayloadCompilationError) as error:
        compile_openai_response_payload(
          api_call_packet_path=temp_root / "api_call_packet.json",
          output_schema_path=unsupported_schema_path,
          output_path=temp_root / "provider_payload.json",
        )

      self.assertIn("$.properties.value.allOf", str(error.exception))

  def test_payload_compiler_strips_schema_compatibility_keys_before_embedding(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      build_agent_routed_api_call_packet(temp_root)
      schema_path = temp_root / "compat.schema.json"
      write_json(
        schema_path,
        {
          "$schema": "https://json-schema.org/draft/2020-12/schema",
          "$id": "https://example.test/project-manager-report.schema.json",
          "type": "object",
          "properties": {
            "report": {
              "$id": "#report",
              "type": "string",
            }
          },
          "required": ["report"],
          "additionalProperties": False,
        },
      )

      payload = compile_openai_response_payload(
        api_call_packet_path=temp_root / "api_call_packet.json",
        output_schema_path=schema_path,
        output_path=temp_root / "provider_payload.json",
      )

      embedded_schema = payload.request.text.format.schema_
      self.assertNotIn("$schema", embedded_schema)
      self.assertNotIn("$id", embedded_schema)
      self.assertNotIn("$id", embedded_schema["properties"]["report"])

  def test_payload_compiler_accepts_explicit_model_for_direct_call(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      build_direct_api_call_packet(temp_root)

      payload = compile_openai_response_payload(
        api_call_packet_path=temp_root / "api_call_packet.json",
        output_schema_path=PM_SCHEMA_PATH,
        direct_model="gpt-5.4-mini",
        output_path=temp_root / "provider_payload.json",
      )

      self.assertEqual(payload.request.text.format.name, "project_manager_report")
      self.assertEqual(payload.request.model, "gpt-5.4-mini")

  def test_payload_compiler_accepts_direct_default_model_from_policy(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      build_direct_api_call_packet(temp_root)

      payload = compile_openai_response_payload(
        api_call_packet_path=temp_root / "api_call_packet.json",
        output_schema_path=PM_SCHEMA_PATH,
        provider_runtime_policy_path=PROVIDER_RUNTIME_POLICY_PATH,
        output_path=temp_root / "provider_payload.json",
      )

      self.assertEqual(payload.request.model, "gpt-5.4-mini")

  def test_emitted_payload_validates_against_generated_schema(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      build_agent_routed_api_call_packet(temp_root)
      output_path = temp_root / "provider_payload.json"

      payload = compile_openai_response_payload(
        api_call_packet_path=temp_root / "api_call_packet.json",
        output_path=output_path,
      )

      schema = load_json(PAYLOAD_SCHEMA_PATH)
      Draft202012Validator.check_schema(schema)
      Draft202012Validator(schema).validate(load_json(output_path))
      self.assertIsInstance(payload, OpenAIResponsePayload)
