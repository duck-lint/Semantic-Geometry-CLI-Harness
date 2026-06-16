from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import get_args

from jsonschema import Draft202012Validator

from harness.contracts.project_manager_report import SourceCoverageDisposition
from harness.runtime.governance_primitives import (
  DerivedRuntimeArtifact,
  GovernancePrimitives,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_PATH = REPO_ROOT / "harness" / "runtime" / "governance_primitives.json"
GOVERNANCE_SCHEMA_PATH = (
  REPO_ROOT / "harness" / "runtime" / "GovernancePrimitives.schema.json"
)


def load_json(path: Path) -> dict:
  return json.loads(path.read_text(encoding="utf-8"))


def runtime_evidence_keys(governance: GovernancePrimitives) -> set[str]:
  runtime_evidence = governance.document_authority_classes.runtime_evidence
  return set(type(runtime_evidence).model_fields)


class GovernancePrimitivesTests(unittest.TestCase):
  def test_current_governance_primitives_validate_with_model_and_schema(self) -> None:
    data = load_json(GOVERNANCE_PATH)
    schema = load_json(GOVERNANCE_SCHEMA_PATH)

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(data)
    governance = GovernancePrimitives.model_validate(data)

    self.assertEqual(
      runtime_evidence_keys(governance),
      set(get_args(DerivedRuntimeArtifact)),
    )

  def test_evidence_taxonomy_is_distinct_from_report_disposition(self) -> None:
    governance = GovernancePrimitives.model_validate(load_json(GOVERNANCE_PATH))
    evidence_classes = runtime_evidence_keys(governance)
    dispositions = set(get_args(SourceCoverageDisposition))

    self.assertTrue(evidence_classes.isdisjoint(dispositions))
    self.assertEqual(
      set(governance.document_authority_classes.runtime_evidence.ledger_artifact),
      {
        "api_call_was_recorded",
        "route_invocation_was_recorded",
        "token_usage_was_recorded",
      },
    )
    self.assertIn(
      "do not create project authority",
      governance.boundary_rule,
    )


if __name__ == "__main__":
  unittest.main()
