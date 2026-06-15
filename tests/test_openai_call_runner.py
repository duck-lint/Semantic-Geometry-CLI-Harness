from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from jsonschema import Draft202012Validator

from harness.agents.agent_context_compiler import compile_agent_context_packet
import harness.providers.openai.openai_call_runner as openai_call_runner
from harness.providers.openai.openai_call_runner import (
  OpenAICallRunnerError,
  run_openai_call,
)
from harness.providers.openai.openai_raw_response import (
  OpenAIRawResponse,
  OpenAIRawResponseMetadata,
)
from harness.providers.openai.openai_response_payload_compiler import (
  compile_openai_response_payload,
)
from harness.runtime.api_call_packet_builder import build_api_call_packet
from harness.runtime.api_call_ledger import DEFAULT_RUNTIME_CALL_LEDGER_PATH
from harness.runtime.task import task_from_cli
from tests.agent_test_support import create_test_project_manager_agent


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = REPO_ROOT / "harness"
MANIFEST_PATH = HARNESS_ROOT / "project_spec" / "static_context_packet.manifest.json"
LIVE_AGENT_PATH = HARNESS_ROOT / "agents" / "project_manager.agent.json"
AGENT_PATH = create_test_project_manager_agent(LIVE_AGENT_PATH)
PM_SCHEMA_PATH = HARNESS_ROOT / "contracts" / "ProjectManagerReport.schema.json"
RAW_RESPONSE_SCHEMA_PATH = (
  HARNESS_ROOT / "providers" / "openai" / "OpenAIRawResponse.schema.json"
)
LEDGER_PATH = DEFAULT_RUNTIME_CALL_LEDGER_PATH


def load_json(path: Path) -> dict:
  return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def remove_ledger_artifact() -> None:
  if LEDGER_PATH.exists():
    LEDGER_PATH.unlink()

  ledger_parent = LEDGER_PATH.parent
  if ledger_parent.exists():
    try:
      ledger_parent.rmdir()
    except OSError:
      pass


def ensure_ledger_artifact() -> None:
  LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
  LEDGER_PATH.write_text("", encoding="utf-8")


def build_agent_routed_provider_payload(temp_root: Path) -> Path:
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
    build_api_call_packet(
      task=task_from_cli("Review the current project trajectory."),
      call_mode="agent_routed",
      agent_context_packet=agent_context_packet,
      output_path=temp_root / "api_call_packet.json",
    )
    provider_payload_path = temp_root / "provider_payload.json"
    compile_openai_response_payload(
      api_call_packet_path=temp_root / "api_call_packet.json",
      output_path=provider_payload_path,
    )
    return provider_payload_path
  finally:
    remove_ledger_artifact()


def build_raw_response_artifact(*, raw_response: dict) -> OpenAIRawResponse:
  return OpenAIRawResponse(
    metadata=OpenAIRawResponseMetadata(
      document_id="raw_model_response.json",
      title="OpenAI Raw Model Response",
      purpose=(
        "Raw OpenAI Responses API result captured before harness output validation."
      ),
      source_format="json",
      document_authority="raw_provider_artifact",
    ),
    provider="openai",
    endpoint="responses.create",
    response_id=raw_response.get("id"),
    model=raw_response.get("model"),
    status=raw_response.get("status"),
    output_text=raw_response.get("output_text"),
    raw_response=raw_response,
    source_artifacts=["provider_payload.json"],
    basis=[
      "Mocked OpenAI raw response artifact for ledger tests.",
    ],
  )


class FakeResponse:
  def __init__(
    self,
    *,
    raw_response: dict,
    response_id: str | None = "resp_123",
    model: str | None = "gpt-5.4",
    status: str | None = "completed",
    output_text: str | None = "{\"status\":\"admissible\"}",
  ) -> None:
    self.id = response_id
    self.model = model
    self.status = status
    self.output_text = output_text
    self._raw_response = raw_response

  def model_dump(self, mode: str = "json") -> dict:
    return dict(self._raw_response)


class FakeOpenAIClient:
  last_request_kwargs: dict | None = None
  next_response: FakeResponse | None = None
  error_to_raise: Exception | None = None

  class _Responses:
    def create(self, **kwargs):
      FakeOpenAIClient.last_request_kwargs = kwargs
      if FakeOpenAIClient.error_to_raise is not None:
        raise FakeOpenAIClient.error_to_raise
      if FakeOpenAIClient.next_response is None:
        raise AssertionError("No fake response configured.")
      return FakeOpenAIClient.next_response

  def __init__(self) -> None:
    self.responses = self._Responses()


class OpenAICallRunnerTests(unittest.TestCase):
  def setUp(self) -> None:
    FakeOpenAIClient.last_request_kwargs = None
    FakeOpenAIClient.error_to_raise = None
    FakeOpenAIClient.next_response = None
    remove_ledger_artifact()

  def tearDown(self) -> None:
    remove_ledger_artifact()

  def test_runner_does_not_write_ledger_on_its_own(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      provider_payload_path = build_agent_routed_provider_payload(temp_root)
      FakeOpenAIClient.next_response = FakeResponse(
        raw_response={"id": "resp_123", "model": "gpt-5.4", "status": "completed"}
      )

      remove_ledger_artifact()
      try:
        with patch(
          "harness.providers.openai.openai_call_runner._load_openai_client_class",
          return_value=FakeOpenAIClient,
        ):
          run_openai_call(
            provider_payload_path=provider_payload_path,
            output_path=temp_root / "raw_model_response.json",
          )

        self.assertFalse(LEDGER_PATH.exists())
      finally:
        remove_ledger_artifact()

  def test_runner_main_appends_single_ledger_row_with_usage(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      provider_payload_path = build_agent_routed_provider_payload(temp_root)
      output_path = temp_root / "raw_model_response.json"
      ledger_path = temp_root / "api_call_ledger.jsonl"
      artifact = build_raw_response_artifact(
        raw_response={
          "id": "resp_usage",
          "model": "gpt-5.4",
          "status": "completed",
          "usage": {
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
          },
          "output_text": "{\"status\":\"admissible\"}",
        }
      )

      def fake_run_openai_call(
        *,
        provider_payload_path: Path,
        output_path: Path,
        payload: object | None = None,
      ) -> OpenAIRawResponse:
        _ = provider_payload_path, payload
        output_path.write_text(
          json.dumps(artifact.model_dump(mode="json", by_alias=True), indent=2)
          + "\n",
          encoding="utf-8",
        )
        return artifact

      with patch(
        "harness.providers.openai.openai_call_runner.run_openai_call",
        side_effect=fake_run_openai_call,
      ):
        with patch(
          "harness.providers.openai.openai_call_runner.DEFAULT_RUNTIME_CALL_LEDGER_PATH",
          ledger_path,
        ):
          code = openai_call_runner.main(
            [
              "--provider-payload",
              str(provider_payload_path),
              "--output",
              str(output_path),
            ]
          )

      self.assertEqual(code, 0)
      self.assertTrue(output_path.exists())
      self.assertTrue(ledger_path.exists())
      lines = ledger_path.read_text(encoding="utf-8").splitlines()
      self.assertEqual(len(lines), 1)
      payload = json.loads(lines[0])
      self.assertEqual(payload["route"], "openai_call_runner")
      self.assertEqual(payload["agent"], "unknown")
      self.assertEqual(payload["schema_name"], "project_manager_report")
      self.assertEqual(
        payload["model"],
        load_json(provider_payload_path)["request"]["model"],
      )
      self.assertTrue(payload["validation_passed"])
      self.assertEqual(payload["openai_response_id"], "resp_usage")
      self.assertEqual(payload["actual_input_tokens"], 11)
      self.assertEqual(payload["actual_output_tokens"], 7)
      self.assertEqual(payload["total_tokens"], 18)
      self.assertEqual(payload["output_artifact_path"], output_path.resolve().as_posix())
      self.assertNotIn("report_artifact_path", payload)
      self.assertNotIn("report_artifact_sha256", payload)
      self.assertNotIn("validation_artifact_path", payload)
      self.assertNotIn("validation_artifact_sha256", payload)

  def test_runner_main_appends_single_ledger_row_without_usage(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      provider_payload_path = build_agent_routed_provider_payload(temp_root)
      output_path = temp_root / "raw_model_response.json"
      ledger_path = temp_root / "api_call_ledger.jsonl"
      artifact = build_raw_response_artifact(
        raw_response={
          "id": "resp_no_usage",
          "model": "gpt-5.4",
          "status": "completed",
          "output_text": "{\"status\":\"admissible\"}",
        }
      )

      def fake_run_openai_call(
        *,
        provider_payload_path: Path,
        output_path: Path,
        payload: object | None = None,
      ) -> OpenAIRawResponse:
        _ = provider_payload_path, payload
        output_path.write_text(
          json.dumps(artifact.model_dump(mode="json", by_alias=True), indent=2)
          + "\n",
          encoding="utf-8",
        )
        return artifact

      with patch(
        "harness.providers.openai.openai_call_runner.run_openai_call",
        side_effect=fake_run_openai_call,
      ):
        with patch(
          "harness.providers.openai.openai_call_runner.DEFAULT_RUNTIME_CALL_LEDGER_PATH",
          ledger_path,
        ):
          code = openai_call_runner.main(
            [
              "--provider-payload",
              str(provider_payload_path),
              "--output",
              str(output_path),
            ]
          )

      self.assertEqual(code, 0)
      self.assertTrue(ledger_path.exists())
      lines = ledger_path.read_text(encoding="utf-8").splitlines()
      self.assertEqual(len(lines), 1)
      payload = json.loads(lines[0])
      self.assertEqual(payload["route"], "openai_call_runner")
      self.assertEqual(payload["openai_response_id"], "resp_no_usage")
      self.assertNotIn("actual_input_tokens", payload)
      self.assertNotIn("actual_output_tokens", payload)
      self.assertNotIn("total_tokens", payload)
      self.assertNotIn("report_artifact_path", payload)
      self.assertNotIn("report_artifact_sha256", payload)
      self.assertNotIn("validation_artifact_path", payload)
      self.assertNotIn("validation_artifact_sha256", payload)

  def test_runner_loads_and_validates_openai_response_payload(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      provider_payload_path = build_agent_routed_provider_payload(temp_root)
      FakeOpenAIClient.next_response = FakeResponse(
        raw_response={"id": "resp_123", "model": "gpt-5.4", "status": "completed"}
      )

      with patch(
        "harness.providers.openai.openai_call_runner._load_openai_client_class",
        return_value=FakeOpenAIClient,
      ):
        artifact = run_openai_call(
          provider_payload_path=provider_payload_path,
          output_path=temp_root / "raw_model_response.json",
        )

      self.assertIsInstance(artifact, OpenAIRawResponse)

  def test_runner_sends_only_payload_request_not_harness_metadata(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      provider_payload_path = build_agent_routed_provider_payload(temp_root)
      payload = load_json(provider_payload_path)
      expected_request = {
        key: value
        for key, value in payload["request"].items()
        if value is not None
      }
      FakeOpenAIClient.next_response = FakeResponse(
        raw_response={"id": "resp_123", "model": "gpt-5.4", "status": "completed"}
      )

      with patch(
        "harness.providers.openai.openai_call_runner._load_openai_client_class",
        return_value=FakeOpenAIClient,
      ):
        run_openai_call(
          provider_payload_path=provider_payload_path,
          output_path=temp_root / "raw_model_response.json",
        )

      self.assertEqual(FakeOpenAIClient.last_request_kwargs, expected_request)
      self.assertNotIn("metadata", FakeOpenAIClient.last_request_kwargs)
      self.assertNotIn("provider", FakeOpenAIClient.last_request_kwargs)
      self.assertNotIn("endpoint", FakeOpenAIClient.last_request_kwargs)
      self.assertNotIn("source_artifacts", FakeOpenAIClient.last_request_kwargs)

  def test_runner_fails_if_payload_provider_is_not_openai(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      provider_payload_path = build_agent_routed_provider_payload(temp_root)
      payload = load_json(provider_payload_path)
      payload["provider"] = "anthropic"
      write_json(provider_payload_path, payload)

      with self.assertRaises(OpenAICallRunnerError) as error:
        run_openai_call(
          provider_payload_path=provider_payload_path,
          output_path=temp_root / "raw_model_response.json",
        )

      self.assertIn("provider == 'openai'", str(error.exception))
      self.assertFalse((temp_root / "raw_model_response.json").exists())

  def test_runner_fails_if_payload_endpoint_is_not_responses_create(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      provider_payload_path = build_agent_routed_provider_payload(temp_root)
      payload = load_json(provider_payload_path)
      payload["endpoint"] = "chat.completions.create"
      write_json(provider_payload_path, payload)

      with self.assertRaises(OpenAICallRunnerError) as error:
        run_openai_call(
          provider_payload_path=provider_payload_path,
          output_path=temp_root / "raw_model_response.json",
        )

      self.assertIn("endpoint == 'responses.create'", str(error.exception))
      self.assertFalse((temp_root / "raw_model_response.json").exists())

  def test_runner_writes_raw_model_response_on_mocked_success(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      provider_payload_path = build_agent_routed_provider_payload(temp_root)
      output_path = temp_root / "raw_model_response.json"
      FakeOpenAIClient.next_response = FakeResponse(
        raw_response={"id": "resp_123", "model": "gpt-5.4", "status": "completed"}
      )

      with patch(
        "harness.providers.openai.openai_call_runner._load_openai_client_class",
        return_value=FakeOpenAIClient,
      ):
        artifact = run_openai_call(
          provider_payload_path=provider_payload_path,
          output_path=output_path,
        )

      self.assertTrue(output_path.is_file())
      self.assertEqual(artifact.source_artifacts, ["provider_payload.json"])

  def test_runner_records_raw_response_body(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      provider_payload_path = build_agent_routed_provider_payload(temp_root)
      raw_response = {
        "id": "resp_123",
        "model": "gpt-5.4",
        "status": "completed",
        "output": [{"type": "message"}],
      }
      FakeOpenAIClient.next_response = FakeResponse(raw_response=raw_response)

      with patch(
        "harness.providers.openai.openai_call_runner._load_openai_client_class",
        return_value=FakeOpenAIClient,
      ):
        artifact = run_openai_call(
          provider_payload_path=provider_payload_path,
          output_path=temp_root / "raw_model_response.json",
        )

      self.assertEqual(artifact.raw_response, raw_response)

  def test_runner_copies_response_fields_when_available(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      provider_payload_path = build_agent_routed_provider_payload(temp_root)
      FakeOpenAIClient.next_response = FakeResponse(
        raw_response={
          "id": "resp_123",
          "model": "gpt-5.4",
          "status": "completed",
          "output_text": "{\"status\":\"admissible\"}",
        }
      )

      with patch(
        "harness.providers.openai.openai_call_runner._load_openai_client_class",
        return_value=FakeOpenAIClient,
      ):
        artifact = run_openai_call(
          provider_payload_path=provider_payload_path,
          output_path=temp_root / "raw_model_response.json",
        )

      self.assertEqual(artifact.response_id, "resp_123")
      self.assertEqual(artifact.model, "gpt-5.4")
      self.assertEqual(artifact.status, "completed")
      self.assertEqual(artifact.output_text, "{\"status\":\"admissible\"}")

  def test_runner_does_not_create_project_manager_report_json(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      provider_payload_path = build_agent_routed_provider_payload(temp_root)
      FakeOpenAIClient.next_response = FakeResponse(
        raw_response={"id": "resp_123", "model": "gpt-5.4", "status": "completed"}
      )

      with patch(
        "harness.providers.openai.openai_call_runner._load_openai_client_class",
        return_value=FakeOpenAIClient,
      ):
        run_openai_call(
          provider_payload_path=provider_payload_path,
          output_path=temp_root / "raw_model_response.json",
        )

      self.assertFalse((temp_root / "project_manager_report.json").exists())
      self.assertFalse(
        (temp_root / "project_manager_report.validation.json").exists()
      )

  def test_runner_does_not_perform_pm_report_schema_validation(self) -> None:
    module_source = inspect.getsource(
      sys.modules["harness.providers.openai.openai_call_runner"]
    )

    self.assertNotIn("ProjectManagerReport.schema.json", module_source)
    self.assertNotIn("project_manager_report.py", module_source)
    self.assertNotIn("jsonschema", module_source)

  def test_runner_does_not_make_real_network_calls_in_tests(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      provider_payload_path = build_agent_routed_provider_payload(temp_root)
      FakeOpenAIClient.error_to_raise = AssertionError("mocked client path only")

      with patch(
        "harness.providers.openai.openai_call_runner._load_openai_client_class",
        return_value=FakeOpenAIClient,
      ):
        with self.assertRaises(OpenAICallRunnerError) as error:
          run_openai_call(
            provider_payload_path=provider_payload_path,
            output_path=temp_root / "raw_model_response.json",
          )

      self.assertIn("mocked client path only", str(error.exception))

  def test_runner_fails_clearly_if_openai_sdk_is_not_installed(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      provider_payload_path = build_agent_routed_provider_payload(temp_root)

      with patch(
        "harness.providers.openai.openai_call_runner._load_openai_client_class",
        side_effect=OpenAICallRunnerError("OpenAI SDK is not installed."),
      ):
        with self.assertRaises(OpenAICallRunnerError) as error:
          run_openai_call(
            provider_payload_path=provider_payload_path,
            output_path=temp_root / "raw_model_response.json",
          )

      self.assertEqual(str(error.exception), "OpenAI SDK is not installed.")
      self.assertFalse((temp_root / "raw_model_response.json").exists())

  def test_runner_direct_script_supports_provider_payload_and_output(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      provider_payload_path = build_agent_routed_provider_payload(temp_root)
      output_path = temp_root / "raw_model_response.json"
      fake_openai_root = temp_root / "fake_openai"
      fake_openai_root.mkdir()
      (fake_openai_root / "openai.py").write_text(
        textwrap.dedent(
          """
          class _Response:
            id = "resp_script"
            model = "gpt-5.4"
            status = "completed"
            output_text = "{\\"ok\\":true}"

            def model_dump(self, mode="json"):
              return {
                "id": self.id,
                "model": self.model,
                "status": self.status,
                "output_text": self.output_text,
              }

          class _Responses:
            def create(self, **kwargs):
              return _Response()

          class OpenAI:
            def __init__(self):
              self.responses = _Responses()
          """
        ),
        encoding="utf-8",
      )
      script_path = HARNESS_ROOT / "providers" / "openai" / "openai_call_runner.py"
      env = dict(os.environ)
      env["PYTHONDONTWRITEBYTECODE"] = "1"
      env["PYTHONPATH"] = str(fake_openai_root)
      remove_ledger_artifact()
      try:
        completed = subprocess.run(
          [
            sys.executable,
            str(script_path),
            "--provider-payload",
            str(provider_payload_path),
            "--output",
            str(output_path),
          ],
          cwd=script_path.parent,
          capture_output=True,
          text=True,
          check=False,
          env=env,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(output_path.is_file())
        self.assertIn("PASS: OpenAI raw response written", completed.stdout)
        self.assertIn("Response id: resp_script", completed.stdout)
        self.assertTrue(LEDGER_PATH.exists())
        ledger_lines = LEDGER_PATH.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(ledger_lines), 1)
        ledger_record = json.loads(ledger_lines[0])
        self.assertEqual(ledger_record["route"], "openai_call_runner")
        self.assertEqual(ledger_record["agent"], "unknown")
        self.assertEqual(ledger_record["schema_name"], "project_manager_report")
        self.assertEqual(ledger_record["openai_response_id"], "resp_script")
        self.assertTrue(ledger_record["validation_passed"])
      finally:
        remove_ledger_artifact()

  def test_emitted_raw_response_validates_against_generated_schema(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      provider_payload_path = build_agent_routed_provider_payload(temp_root)
      FakeOpenAIClient.next_response = FakeResponse(
        raw_response={"id": "resp_123", "model": "gpt-5.4", "status": "completed"}
      )

      with patch(
        "harness.providers.openai.openai_call_runner._load_openai_client_class",
        return_value=FakeOpenAIClient,
      ):
        artifact = run_openai_call(
          provider_payload_path=provider_payload_path,
          output_path=temp_root / "raw_model_response.json",
        )

      schema = load_json(RAW_RESPONSE_SCHEMA_PATH)
      Draft202012Validator.check_schema(schema)
      Draft202012Validator(schema).validate(
        load_json(temp_root / "raw_model_response.json")
      )
      self.assertIsInstance(artifact, OpenAIRawResponse)


if __name__ == "__main__":
  unittest.main()
