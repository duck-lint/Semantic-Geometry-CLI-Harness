from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from harness.agents.agent_context_compiler import compile_agent_context_packet
from harness.runtime.api_call_packet_builder import build_api_call_packet
from harness.runtime.api_call_ledger import DEFAULT_RUNTIME_CALL_LEDGER_PATH
from harness.runtime.model_resolution import (
  EffectiveModelSelection,
  ModelResolutionError,
  resolve_effective_model,
  write_effective_model_selection,
)
from harness.runtime.provider_runtime_policy import ProviderRuntimePolicy
from harness.runtime.runtime_budget_policy import RuntimeBudgetPolicy
from harness.runtime.task import task_from_cli
from tests.agent_test_support import create_test_project_manager_agent


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = REPO_ROOT / "harness"
MANIFEST_PATH = HARNESS_ROOT / "project_spec" / "static_context_packet.manifest.json"
LIVE_AGENT_PATH = HARNESS_ROOT / "agents" / "project_manager.agent.json"
AGENT_PATH = create_test_project_manager_agent(LIVE_AGENT_PATH)
PROVIDER_POLICY_PATH = HARNESS_ROOT / "runtime" / "provider_runtime.policy.json"
PROVIDER_POLICY_SCHEMA_PATH = (
  HARNESS_ROOT / "runtime" / "ProviderRuntimePolicy.schema.json"
)
EFFECTIVE_MODEL_SELECTION_SCHEMA_PATH = (
  HARNESS_ROOT / "runtime" / "EffectiveModelSelection.schema.json"
)
RUNTIME_BUDGET_PATH = HARNESS_ROOT / "runtime" / "runtime_budget.policy.json"


def load_json(path: Path) -> dict:
  return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
  path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def ensure_ledger_artifact() -> None:
  ledger_path = DEFAULT_RUNTIME_CALL_LEDGER_PATH
  ledger_path.parent.mkdir(parents=True, exist_ok=True)
  ledger_path.write_text("", encoding="utf-8")


def write_agent_variant(temp_root: Path, *, model: str) -> Path:
  agent_path = temp_root / "project_manager.agent.json"
  agent_data = load_json(AGENT_PATH)
  agent_data["model"] = model
  write_json(agent_path, agent_data)
  return agent_path


def build_provider_policy_data(
  *,
  default_direct_model: str = "gpt-5.4-mini",
  allowed_models: list[str] | None = None,
  fallback_strategy: str = "fail",
  fallback_models: list[dict] | None = None,
) -> dict:
  return {
    "$schema": "ProviderRuntimePolicy.schema.json",
    "metadata": {
      "document_id": "provider_runtime.policy.json",
      "title": "Provider Runtime Policy",
      "purpose": (
        "Defines provider/model defaults, allowed models, and fallback behavior "
        "for provider rendering."
      ),
      "source_format": "json",
      "document_authority": "runtime_policy",
    },
    "default_direct_model": {
      "provider": "openai",
      "model": default_direct_model,
    },
    "allowed_models": {
      "openai": allowed_models or ["gpt-5.4-mini", "gpt-5.4"],
    },
    "fallback_strategy": fallback_strategy,
    "fallback_models": fallback_models or [],
  }


def build_agent_routed_api_call_packet(
  temp_root: Path,
  *,
  agent_path: Path = AGENT_PATH,
):
  ensure_ledger_artifact()
  try:
    agent_context_path = temp_root / "agent_context_packet.json"
    static_context_path = temp_root / "static_context_packet.json"
    agent_context_packet = compile_agent_context_packet(
      agent_path=agent_path,
      output_path=agent_context_path,
      manifest_path=MANIFEST_PATH,
      harness_root=REPO_ROOT,
      target_repo_root=REPO_ROOT,
      static_context_output_path=static_context_path,
    )
    return build_api_call_packet(
      task=task_from_cli("Review the current project trajectory."),
      call_mode="agent_routed",
      agent_context_packet=agent_context_packet,
      output_path=temp_root / "api_call_packet.json",
    )
  finally:
    ledger_path = DEFAULT_RUNTIME_CALL_LEDGER_PATH
    if ledger_path.exists():
      ledger_path.unlink()
      try:
        ledger_path.parent.rmdir()
      except OSError:
        pass


def build_direct_api_call_packet(
  temp_root: Path,
  *,
  runtime_budget: RuntimeBudgetPolicy | None = None,
):
  return build_api_call_packet(
    task=task_from_cli("Review the current project trajectory."),
    call_mode="direct",
    runtime_budget=runtime_budget,
    output_path=temp_root / "api_call_packet.json",
  )


class ModelResolutionTests(unittest.TestCase):
  def test_agent_routed_call_uses_agent_contract_model(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      packet = build_agent_routed_api_call_packet(temp_root)
      policy = ProviderRuntimePolicy.model_validate(load_json(PROVIDER_POLICY_PATH))
      agent_model = load_json(AGENT_PATH)["model"]

      selection = resolve_effective_model(
        api_call_packet=packet,
        provider_runtime_policy=policy,
      )

      self.assertEqual(selection.provider, "openai")
      self.assertEqual(selection.model, agent_model)
      self.assertEqual(selection.source, "agent_contract")
      self.assertFalse(selection.fallback_used)
      self.assertEqual(selection.requested_model, agent_model)

  def test_direct_call_uses_provider_runtime_policy_default_direct_model(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      packet = build_direct_api_call_packet(temp_root)
      policy = ProviderRuntimePolicy.model_validate(load_json(PROVIDER_POLICY_PATH))

      selection = resolve_effective_model(
        api_call_packet=packet,
        provider_runtime_policy=policy,
      )

      self.assertEqual(selection.provider, "openai")
      self.assertEqual(selection.model, "gpt-5.4-mini")
      self.assertEqual(
        selection.source,
        "provider_runtime_policy.default_direct_model",
      )
      self.assertFalse(selection.fallback_used)

  def test_runtime_budget_does_not_affect_model_selection(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      runtime_budget = RuntimeBudgetPolicy.model_validate(
        load_json(RUNTIME_BUDGET_PATH)
      )
      packet_with_budget = build_direct_api_call_packet(
        temp_root / "with_budget",
        runtime_budget=runtime_budget,
      )
      packet_without_budget = build_direct_api_call_packet(
        temp_root / "without_budget"
      )
      policy = ProviderRuntimePolicy.model_validate(load_json(PROVIDER_POLICY_PATH))

      selection_with_budget = resolve_effective_model(
        api_call_packet=packet_with_budget,
        provider_runtime_policy=policy,
      )
      selection_without_budget = resolve_effective_model(
        api_call_packet=packet_without_budget,
        provider_runtime_policy=policy,
      )

      self.assertEqual(
        selection_with_budget.model,
        selection_without_budget.model,
      )
      self.assertEqual(
        selection_with_budget.source,
        selection_without_budget.source,
      )

  def test_agent_model_disallowed_fail_strategy_raises(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      agent_path = write_agent_variant(temp_root, model="gpt-5.4")
      packet = build_agent_routed_api_call_packet(temp_root, agent_path=agent_path)
      policy = ProviderRuntimePolicy.model_validate(
        build_provider_policy_data(
          allowed_models=["gpt-5.4-mini"],
          fallback_strategy="fail",
        )
      )

      with self.assertRaises(ModelResolutionError):
        resolve_effective_model(
          api_call_packet=packet,
          provider_runtime_policy=policy,
        )

  def test_agent_model_disallowed_fallback_strategy_selects_first_allowed_fallback(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      agent_path = write_agent_variant(temp_root, model="gpt-5.4")
      packet = build_agent_routed_api_call_packet(temp_root, agent_path=agent_path)
      policy = ProviderRuntimePolicy.model_validate(
        build_provider_policy_data(
          allowed_models=["gpt-5.4-mini"],
          fallback_strategy="first_allowed_fallback",
          fallback_models=[
            {"provider": "openai", "model": "gpt-5.4-mini"},
          ],
        )
      )

      selection = resolve_effective_model(
        api_call_packet=packet,
        provider_runtime_policy=policy,
      )

      self.assertEqual(selection.model, "gpt-5.4-mini")
      self.assertEqual(selection.source, "provider_runtime_policy.fallback_model")
      self.assertTrue(selection.fallback_used)
      self.assertEqual(selection.requested_model, "gpt-5.4")

  def test_direct_default_model_disallowed_fail_strategy_raises(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      packet = build_direct_api_call_packet(temp_root)
      policy = ProviderRuntimePolicy.model_validate(
        build_provider_policy_data(
          default_direct_model="gpt-5.4-mini",
          allowed_models=["gpt-5.4"],
          fallback_strategy="fail",
        )
      )

      with self.assertRaises(ModelResolutionError):
        resolve_effective_model(
          api_call_packet=packet,
          provider_runtime_policy=policy,
        )

  def test_direct_default_model_disallowed_fallback_strategy_selects_first_allowed_fallback(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      packet = build_direct_api_call_packet(temp_root)
      policy = ProviderRuntimePolicy.model_validate(
        build_provider_policy_data(
          default_direct_model="gpt-5.4-mini",
          allowed_models=["gpt-5.4"],
          fallback_strategy="first_allowed_fallback",
          fallback_models=[
            {"provider": "openai", "model": "gpt-5.4"},
          ],
        )
      )

      selection = resolve_effective_model(
        api_call_packet=packet,
        provider_runtime_policy=policy,
      )

      self.assertEqual(selection.model, "gpt-5.4")
      self.assertEqual(selection.source, "provider_runtime_policy.fallback_model")
      self.assertTrue(selection.fallback_used)
      self.assertEqual(selection.requested_model, "gpt-5.4-mini")

  def test_effective_model_selection_artifact_validates_against_generated_schema(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      api_call_packet = build_direct_api_call_packet(temp_root)
      policy_path = temp_root / "provider_runtime.policy.json"
      output_path = temp_root / "effective_model_selection.json"
      write_json(policy_path, build_provider_policy_data())

      selection = write_effective_model_selection(
        api_call_packet_path=temp_root / "api_call_packet.json",
        provider_runtime_policy_path=policy_path,
        output_path=output_path,
      )

      schema = load_json(EFFECTIVE_MODEL_SELECTION_SCHEMA_PATH)
      Draft202012Validator.check_schema(schema)
      Draft202012Validator(schema).validate(load_json(output_path))
      self.assertIsInstance(selection, EffectiveModelSelection)
      self.assertEqual(api_call_packet.call_mode, "direct")

  def test_provider_runtime_policy_json_validates_against_generated_schema(self) -> None:
    schema = load_json(PROVIDER_POLICY_SCHEMA_PATH)
    policy = load_json(PROVIDER_POLICY_PATH)

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(policy)

  def test_model_resolution_script_supports_direct_execution_without_provider_rendering(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      build_direct_api_call_packet(temp_root)
      policy_path = temp_root / "provider_runtime.policy.json"
      write_json(policy_path, build_provider_policy_data())
      output_path = temp_root / "effective_model_selection.json"
      script_path = HARNESS_ROOT / "runtime" / "model_resolution.py"
      openai_loaded_before = "openai" in sys.modules

      completed = subprocess.run(
        [
          sys.executable,
          str(script_path),
          "--api-call-packet",
          str(temp_root / "api_call_packet.json"),
          "--provider-runtime-policy",
          str(policy_path),
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
      self.assertFalse((temp_root / "provider_payload.json").exists())
      self.assertEqual("openai" in sys.modules, openai_loaded_before)


if __name__ == "__main__":
  unittest.main()
