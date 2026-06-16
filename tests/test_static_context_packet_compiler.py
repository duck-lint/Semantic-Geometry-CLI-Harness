from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from harness.project_spec.static_context_packet import StaticContextPacket
from harness.project_spec.static_context_packet_compiler import (
  StaticContextCompilationError,
  compile_static_context_packet,
  enforce_cardinality,
  load_and_validate_manifest,
)
from harness.project_spec.static_context_packet_manifest import Source
from harness.runtime.governance_primitives import GovernancePrimitives


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = REPO_ROOT / "harness"
MANIFEST_PATH = HARNESS_ROOT / "project_spec" / "static_context_packet.manifest.json"
PACKET_SCHEMA_PATH = HARNESS_ROOT / "project_spec" / "StaticContextPacket.schema.json"


def copy_target_sources(
  destination: Path,
  *,
  include_known_failures: bool = True,
  include_active_implementation: bool = False,
) -> Path:
  target_root = destination / "target"
  project_spec_root = target_root / "harness" / "project_spec"
  project_spec_root.mkdir(parents=True)

  source_names = ["project_spec.json", "open_decisions.json"]
  if include_known_failures:
    source_names.append("known_failures.json")

  for source_name in source_names:
    shutil.copy2(
      HARNESS_ROOT / "project_spec" / source_name,
      project_spec_root / source_name,
    )

  if include_active_implementation:
    active_root = target_root / "harness" / "implementations" / "active"
    active_root.mkdir(parents=True)
    (active_root / "implementation_plan_01.json").write_text(
      json.dumps({"fixture": "implementation_plan"}),
      encoding="utf-8",
    )
    (active_root / "implementation_tracker_01.json").write_text(
      json.dumps({"fixture": "implementation_tracker"}),
      encoding="utf-8",
    )

  return target_root


class StaticContextPacketCompilerTests(unittest.TestCase):
  def test_current_manifest_validates(self) -> None:
    manifest = load_and_validate_manifest(MANIFEST_PATH)

    self.assertEqual(len(manifest.sources), 6)
    self.assertEqual(
      {source.source_id for source in manifest.sources},
      {
        "governance_primitives",
        "project_spec",
        "known_failures",
        "open_decisions",
        "active_implementation_plan",
        "active_implementation_tracker",
      },
    )

  def test_compiler_emits_valid_packet_with_all_live_sources(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      target_root = copy_target_sources(
        temp_root,
        include_active_implementation=True,
      )
      output_path = temp_root / "static_context_packet.json"

      packet = compile_static_context_packet(
        MANIFEST_PATH,
        REPO_ROOT,
        target_root,
        output_path,
      )

      self.assertTrue(output_path.is_file())
      self.assertEqual(len(packet.source_coverage), 6)
      self.assertTrue(
        all(entry.status == "included" for entry in packet.source_coverage)
      )
      self.assertTrue(
        all(
          entry.validation.status == "passed"
          and entry.validation.validator == "json_parse"
          and entry.validation.parsed_content_available
          for entry in packet.source_coverage
        )
      )
      self.assertEqual(packet.missing_sources, [])
      self.assertEqual(
        packet.sources["governance_primitives"]["metadata"]["document_id"],
        "governance_primitives.json",
      )
      governance = GovernancePrimitives.model_validate(
        packet.sources["governance_primitives"]
      )
      self.assertIn(
        "ledger_artifact",
        governance.evidence_classes.runtime_evidence,
      )
      self.assertEqual(
        packet.sources["project_spec"]["metadata"]["document_id"],
        "project_spec.json",
      )
      self.assertEqual(
        packet.sources["known_failures"]["metadata"]["document_id"],
        "known_failures.json",
      )
      self.assertEqual(
        packet.sources["open_decisions"]["metadata"]["document_id"],
        "open_decisions.json",
      )
      self.assertIsNotNone(packet.sources["active_implementation_plan"])
      self.assertIsNotNone(packet.sources["active_implementation_tracker"])

      emitted = json.loads(output_path.read_text(encoding="utf-8"))
      self.assertEqual(StaticContextPacket.model_validate(emitted), packet)

  def test_optional_active_sources_are_recorded_absent(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      target_root = copy_target_sources(temp_root)
      output_path = temp_root / "static_context_packet.json"

      packet = compile_static_context_packet(
        MANIFEST_PATH,
        REPO_ROOT,
        target_root,
        output_path,
      )

      self.assertIsNone(packet.sources["active_implementation_plan"])
      self.assertIsNone(packet.sources["active_implementation_tracker"])
      self.assertEqual(
        {entry.source_id for entry in packet.missing_sources},
        {
          "active_implementation_plan",
          "active_implementation_tracker",
        },
      )
      self.assertTrue(
        all(
          entry.effect == "recorded_absent"
          for entry in packet.missing_sources
        )
      )
      self.assertEqual(
        {
          entry.source_id
          for entry in packet.source_coverage
          if entry.status == "missing"
        },
        {
          "active_implementation_plan",
          "active_implementation_tracker",
        },
      )
      self.assertTrue(
        all(
          entry.validation.status == "not_run"
          and not entry.validation.parsed_content_available
          for entry in packet.source_coverage
          if entry.status == "missing"
        )
      )

  def test_invalid_optional_source_blocks_compilation(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      target_root = copy_target_sources(
        temp_root,
        include_active_implementation=True,
      )
      plan_path = (
        target_root
        / "harness"
        / "implementations"
        / "active"
        / "implementation_plan_01.json"
      )
      plan_path.write_text("{", encoding="utf-8")
      output_path = temp_root / "static_context_packet.json"

      with self.assertRaises(StaticContextCompilationError) as error:
        compile_static_context_packet(
          MANIFEST_PATH,
          REPO_ROOT,
          target_root,
          output_path,
        )

      self.assertFalse(output_path.exists())
      self.assertTrue(
        any(
          entry.source_id == "active_implementation_plan"
          and entry.status == "invalid"
          and entry.validation.status == "failed"
          and not entry.validation.parsed_content_available
          for entry in error.exception.source_coverage
        )
      )

  def test_missing_required_source_blocks_without_emitting_packet(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      target_root = copy_target_sources(
        temp_root,
        include_known_failures=False,
      )
      output_path = temp_root / "static_context_packet.json"

      with self.assertRaises(StaticContextCompilationError) as error:
        compile_static_context_packet(
          MANIFEST_PATH,
          REPO_ROOT,
          target_root,
          output_path,
        )

      self.assertFalse(output_path.exists())
      self.assertTrue(
        any(
          entry.source_id == "known_failures"
          and entry.effect == "blocks_compilation"
          for entry in error.exception.missing_sources
        )
      )

  def test_invalid_required_source_blocks_without_raw_dict_inclusion(
    self,
  ) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      target_root = copy_target_sources(temp_root)
      known_failures_path = (
        target_root / "harness" / "project_spec" / "known_failures.json"
      )
      known_failures_path.write_text("{", encoding="utf-8")
      output_path = temp_root / "static_context_packet.json"

      with self.assertRaises(StaticContextCompilationError) as error:
        compile_static_context_packet(
          MANIFEST_PATH,
          REPO_ROOT,
          target_root,
          output_path,
        )

      self.assertFalse(output_path.exists())
      self.assertTrue(
        any(
          entry.source_id == "known_failures"
          and entry.status == "invalid"
          and entry.validation.status == "failed"
          for entry in error.exception.source_coverage
        )
      )

  def test_manifest_source_ids_drive_packet_shape_without_python_changes(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      manifest_path = temp_root / "manifest.json"
      output_path = temp_root / "static_context_packet.json"
      manifest = {
        "$schema": "./StaticContextPacketManifest.schema.json",
        "metadata": {
          "id": "static_context_packet.manifest.json",
          "name": "Static Context Packet Manifest",
        },
        "sources": [
          {
            "source_id": "renamed_project_authority",
            "scope": "target_repo",
            "required": True,
            "document_authority": "harness_target",
            "document": "harness/project_spec/project_spec.json",
            "schema_id": "project_spec",
            "cardinality": "exactly_one",
          }
        ],
      }
      manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

      packet = compile_static_context_packet(
        manifest_path,
        REPO_ROOT,
        REPO_ROOT,
        output_path,
      )

      self.assertEqual(set(packet.sources), {"renamed_project_authority"})
      self.assertEqual(
        packet.sources["renamed_project_authority"]["metadata"]["document_id"],
        "project_spec.json",
      )
      self.assertEqual(
        [entry.source_id for entry in packet.source_coverage],
        ["renamed_project_authority"],
      )

  def test_multi_document_cardinality_emits_a_list(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      source_root = temp_root / "harness" / "project_spec"
      source_root.mkdir(parents=True)
      for index in range(2):
        shutil.copy2(
          HARNESS_ROOT / "project_spec" / "open_decisions.json",
          source_root / f"open_decisions_{index}.json",
        )

      manifest_path = temp_root / "manifest.json"
      manifest_path.write_text(
        json.dumps(
          {
            "$schema": "./StaticContextPacketManifest.schema.json",
            "metadata": {
              "id": "static_context_packet.manifest.json",
              "name": "Static Context Packet Manifest",
            },
            "sources": [
              {
                "source_id": "decision_documents",
                "scope": "target_repo",
                "required": True,
                "document_authority": "harness_target",
                "document_glob": "harness/project_spec/open_decisions_*.json",
                "schema_id": "open_decisions",
                "cardinality": "one_or_more",
              }
            ],
          }
        ),
        encoding="utf-8",
      )

      packet = compile_static_context_packet(
        manifest_path,
        REPO_ROOT,
        temp_root,
        temp_root / "static_context_packet.json",
      )

      self.assertEqual(len(packet.sources["decision_documents"]), 2)
      self.assertEqual(packet.source_coverage[0].status, "included")

  def test_manifest_may_declare_opaque_json_without_a_python_model(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)
      source_path = temp_root / "opaque.json"
      source_path.write_text(json.dumps(["alpha", 2, False]), encoding="utf-8")
      manifest_path = temp_root / "manifest.json"
      manifest_path.write_text(
        json.dumps(
          {
            "$schema": "./StaticContextPacketManifest.schema.json",
            "metadata": {
              "id": "static_context_packet.manifest.json",
              "name": "Static Context Packet Manifest",
            },
            "sources": [
              {
                "source_id": "opaque_payload",
                "scope": "target_repo",
                "required": True,
                "document_authority": "operational_state",
                "document": "opaque.json",
                "schema_id": "opaque_json",
                "cardinality": "exactly_one",
              }
            ],
          }
        ),
        encoding="utf-8",
      )

      packet = compile_static_context_packet(
        manifest_path,
        REPO_ROOT,
        temp_root,
        temp_root / "static_context_packet.json",
      )

      self.assertEqual(packet.sources["opaque_payload"], ["alpha", 2, False])
      self.assertEqual(packet.source_coverage[0].schema_id, "opaque_json")

  def test_emitted_packet_validates_against_generated_schema(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      output_path = Path(temp_directory) / "static_context_packet.json"
      compile_static_context_packet(
        MANIFEST_PATH,
        REPO_ROOT,
        REPO_ROOT,
        output_path,
      )

      schema = json.loads(PACKET_SCHEMA_PATH.read_text(encoding="utf-8"))
      emitted = json.loads(output_path.read_text(encoding="utf-8"))

      Draft202012Validator.check_schema(schema)
      Draft202012Validator(schema).validate(emitted)

  def test_generated_schema_has_no_invalid_sources_category(self) -> None:
    schema = json.loads(PACKET_SCHEMA_PATH.read_text(encoding="utf-8"))

    self.assertNotIn("InvalidSourceEntry", schema.get("$defs", {}))
    self.assertNotIn("invalid_sources", schema["properties"])

  def test_all_cardinality_modes(self) -> None:
    cases = [
      ("exactly_one", 1, True),
      ("exactly_one", 0, False),
      ("zero_or_one", 0, True),
      ("zero_or_one", 2, False),
      ("zero_or_more", 0, True),
      ("zero_or_more", 2, True),
      ("one_or_more", 1, True),
      ("one_or_more", 0, False),
    ]

    with tempfile.TemporaryDirectory() as temp_directory:
      temp_root = Path(temp_directory)

      for cardinality, count, is_valid in cases:
        with self.subTest(cardinality=cardinality, count=count):
          source = Source.model_construct(
            source_id="project_spec",
            cardinality=cardinality,
          )
          paths = [temp_root / f"{index}.json" for index in range(count)]

          if is_valid:
            enforce_cardinality(source, paths)
          else:
            with self.assertRaises(ValueError):
              enforce_cardinality(source, paths)

  def test_compiler_supports_direct_script_execution(self) -> None:
    with tempfile.TemporaryDirectory() as temp_directory:
      output_path = Path(temp_directory) / "static_context_packet.json"
      script_path = (
        HARNESS_ROOT / "project_spec" / "static_context_packet_compiler.py"
      )

      completed = subprocess.run(
        [
          sys.executable,
          str(script_path),
          "--output",
          str(output_path),
        ],
        cwd=script_path.parent,
        capture_output=True,
        text=True,
        check=False,
      )

      self.assertEqual(completed.returncode, 0, completed.stderr)
      self.assertTrue(output_path.is_file())
      self.assertIn("PASS: Static context packet written", completed.stdout)


if __name__ == "__main__":
  unittest.main()
