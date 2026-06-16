from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import get_args

from harness.contracts.project_manager_report import (
  ProjectManagerReport,
  SourceCoverageDisposition,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_ROOT = REPO_ROOT / "tests" / "fixtures"
CANONICAL_REJECTED_UNBLOCKED_RESPONSE_PATH = (
  FIXTURES_ROOT / "raw_model_response_rejected_unblocked.json"
)


def load_json(path: Path) -> dict:
  return json.loads(path.read_text(encoding="utf-8"))


def build_report_data(
  *,
  report_status: str,
  blocked: bool,
  blocking_reason: str | None,
  missing_basis: list[str],
  constraint_conflicts: list[str],
  next_admissible_transformation: str | None,
  repo_snapshot_packet_consumed: bool = True,
  repo_snapshot_packet_disposition: str = "used",
  repo_snapshot_packet_basis: list[str] | None = None,
  git_context_consumed: bool = True,
  git_context_disposition: str = "used",
) -> dict:
  report_source_coverage: dict[str, dict[str, object]] = {
    "static_context_packet": {
      "consumed": True,
      "disposition": "used",
      "basis": [
        "Used project_spec, governance_primitives, open_decisions, active_implementation_plan, and active_implementation_tracker.",
      ],
    },
    "task": {
      "consumed": True,
      "disposition": "used",
      "basis": [
        "Used the task text as current task authority.",
      ],
    },
    "repo_snapshot_packet": {
      "consumed": repo_snapshot_packet_consumed,
      "disposition": repo_snapshot_packet_disposition,
      "basis": (
        repo_snapshot_packet_basis
        if repo_snapshot_packet_basis is not None
        else [
          "Used repo_snapshot_packet as historical artifact evidence for the current PM slice.",
          "Historical artifact paths named in the repo snapshot packet included harness/runs/20260612-214948-agent-route/project_manager_report.json and harness/runs/20260612-214948-agent-route/raw_model_response.json.",
        ]
      ),
    },
    "git_context": {
      "consumed": git_context_consumed,
      "disposition": git_context_disposition,
      "basis": [
        "Used git_context to distinguish current worktree provenance from saved run artifacts.",
      ],
    },
  }

  return {
    "metadata": {
      "document_id": "project_manager_report.json",
      "title": "Project Manager Report",
      "source_format": "json",
      "document_authority": "output_policy_artifact",
    },
    "report_status": report_status,
    "report_summary": "Short report summary.",
    "report_source_coverage": report_source_coverage,
    "trajectory_review": {
      "current_posture": "Concrete repo-state evidenced from the provided context.",
      "thesis_attractor": "Direction implied by project spec without invented roadmap.",
      "drift_detection": {
        "drift_detected": True,
        "description": "Description of drift if detected.",
      },
      "structural_tension": "Main actionable mismatch.",
    },
    "proof_frontier": {
      "constraint_conflicts": constraint_conflicts,
      "dominant_tension_justification": "Why this tension governs.",
      "blocked": blocked,
      "blocking_reason": blocking_reason,
      "missing_basis": missing_basis,
      "next_admissible_transformation": next_admissible_transformation,
      "affected_surfaces": [
        "Surface touched by next transition.",
      ],
      "non_affected_surfaces": [
        "Surface explicitly not touched.",
      ],
      "stop_conditions": [
        "Condition that should stop the work.",
      ],
      "authority_constraints": [
        "Constraint from task.",
      ],
    },
  }


class ProjectManagerReportValidationTests(unittest.TestCase):
  def test_admissible_requires_unblocked_frontier_and_next_move(self) -> None:
    report = ProjectManagerReport.model_validate(
      build_report_data(
        report_status="admissible",
        blocked=False,
        blocking_reason=None,
        missing_basis=[],
        constraint_conflicts=[],
        next_admissible_transformation="Produce the validated PM report.",
      )
    )

    self.assertEqual(report.report_status, "admissible")
    self.assertFalse(report.proof_frontier.blocked)
    self.assertIsNotNone(report.report_source_coverage.repo_snapshot_packet)
    self.assertTrue(
      any(
        "harness/runs/20260612-214948-agent-route/project_manager_report.json"
        in basis
        for basis in report.report_source_coverage.repo_snapshot_packet.basis
      )
    )
    self.assertEqual(
      report.proof_frontier.next_admissible_transformation,
      "Produce the validated PM report.",
    )

  def test_report_source_coverage_accepts_explicit_missing_repo_snapshot_packet(self) -> None:
    report = ProjectManagerReport.model_validate(
      build_report_data(
        report_status="needs_clarification",
        blocked=False,
        blocking_reason=None,
        missing_basis=["Clarify the target surface."],
        constraint_conflicts=[],
        next_admissible_transformation="Ask the user to name the target surface.",
        repo_snapshot_packet_consumed=False,
        repo_snapshot_packet_disposition="missing",
        repo_snapshot_packet_basis=[
          "No repo_snapshot_packet was supplied for this task.",
        ],
      )
    )

    self.assertFalse(report.report_source_coverage.repo_snapshot_packet.consumed)
    self.assertEqual(
      report.report_source_coverage.repo_snapshot_packet.disposition,
      "missing",
    )

  def test_report_source_coverage_rejects_empty_basis(self) -> None:
    dispositions = (
      ("used", True),
      ("inspected_insufficient", True),
      ("inspected_not_relevant", True),
      ("inspected_contradictory", True),
      ("missing", False),
      ("invalid", False),
      ("not_required_for_task", False),
    )
    for disposition, consumed in dispositions:
      with self.subTest(disposition=disposition):
        with self.assertRaises(ValueError):
          ProjectManagerReport.model_validate(
            build_report_data(
              report_status="needs_clarification",
              blocked=False,
              blocking_reason=None,
              missing_basis=["Clarify the target surface."],
              constraint_conflicts=[],
              next_admissible_transformation="Ask the user to name the target surface.",
              repo_snapshot_packet_consumed=consumed,
              repo_snapshot_packet_disposition=disposition,
              repo_snapshot_packet_basis=[],
            )
          )

  def test_report_source_coverage_requires_every_fixed_key(self) -> None:
    for source_id in (
      "static_context_packet",
      "task",
      "repo_snapshot_packet",
      "git_context",
    ):
      with self.subTest(source_id=source_id):
        report_data = build_report_data(
          report_status="needs_clarification",
          blocked=False,
          blocking_reason=None,
          missing_basis=["Clarify the target surface."],
          constraint_conflicts=[],
          next_admissible_transformation="Ask the user to name the target surface.",
        )
        del report_data["report_source_coverage"][source_id]

        with self.assertRaises(ValueError):
          ProjectManagerReport.model_validate(report_data)

  def test_report_source_coverage_requires_disposition(self) -> None:
    report_data = build_report_data(
      report_status="needs_clarification",
      blocked=False,
      blocking_reason=None,
      missing_basis=["Clarify the target surface."],
      constraint_conflicts=[],
      next_admissible_transformation="Ask the user to name the target surface.",
    )
    del report_data["report_source_coverage"]["repo_snapshot_packet"]["disposition"]

    with self.assertRaises(ValueError):
      ProjectManagerReport.model_validate(report_data)

  def test_report_source_coverage_rejects_unknown_disposition(self) -> None:
    report_data = build_report_data(
      report_status="needs_clarification",
      blocked=False,
      blocking_reason=None,
      missing_basis=["Clarify the target surface."],
      constraint_conflicts=[],
      next_admissible_transformation="Ask the user to name the target surface.",
    )
    report_data["report_source_coverage"]["repo_snapshot_packet"]["disposition"] = (
      "ignored"
    )

    with self.assertRaises(ValueError):
      ProjectManagerReport.model_validate(report_data)

  def test_report_source_coverage_accepts_inspected_dispositions(self) -> None:
    for disposition in (
      "inspected_insufficient",
      "inspected_not_relevant",
      "inspected_contradictory",
    ):
      with self.subTest(disposition=disposition):
        report = ProjectManagerReport.model_validate(
          build_report_data(
            report_status="needs_clarification",
            blocked=False,
            blocking_reason=None,
            missing_basis=["Clarify the target surface."],
            constraint_conflicts=[],
            next_admissible_transformation="Ask the user to name the target surface.",
            repo_snapshot_packet_disposition=disposition,
            repo_snapshot_packet_basis=[
              f"Reviewed the repo snapshot and classified it as {disposition}.",
            ],
          )
        )

        self.assertTrue(report.report_source_coverage.repo_snapshot_packet.consumed)
        self.assertEqual(
          report.report_source_coverage.repo_snapshot_packet.disposition,
          disposition,
        )

  def test_source_coverage_basis_is_opaque_free_text(self) -> None:
    report = ProjectManagerReport.model_validate(
      build_report_data(
        report_status="needs_clarification",
        blocked=False,
        blocking_reason=None,
        missing_basis=["Clarify the target surface."],
        constraint_conflicts=[],
        next_admissible_transformation="Ask the user to name the target surface.",
        repo_snapshot_packet_disposition="inspected_insufficient",
        repo_snapshot_packet_basis=[
          "Reviewed it carefully; this explanation intentionally contains no taxonomy terms.",
        ],
      )
    )

    self.assertEqual(
      report.report_source_coverage.repo_snapshot_packet.basis,
      [
        "Reviewed it carefully; this explanation intentionally contains no taxonomy terms.",
      ],
    )

  def test_used_may_describe_non_authoritative_derived_evidence(self) -> None:
    report = ProjectManagerReport.model_validate(
      build_report_data(
        report_status="admissible",
        blocked=False,
        blocking_reason=None,
        missing_basis=[],
        constraint_conflicts=[],
        next_admissible_transformation="Preserve the bounded call record.",
        repo_snapshot_packet_disposition="used",
        repo_snapshot_packet_basis=[
          "Used non-authoritative ledger_artifact evidence for the valid claim family api_call_was_recorded.",
        ],
      )
    )

    self.assertEqual(
      report.report_source_coverage.repo_snapshot_packet.disposition,
      "used",
    )
    self.assertEqual(report.report_status, "admissible")

  def test_inspected_contradictory_does_not_force_rejected_status(self) -> None:
    report = ProjectManagerReport.model_validate(
      build_report_data(
        report_status="admissible",
        blocked=False,
        blocking_reason=None,
        missing_basis=[],
        constraint_conflicts=[],
        next_admissible_transformation="Proceed using the controlling source.",
        repo_snapshot_packet_disposition="inspected_contradictory",
        repo_snapshot_packet_basis=[
          "The supplied derived artifact contradicted the requested claim family, but no authority conflict was established.",
        ],
      )
    )

    self.assertEqual(
      report.report_source_coverage.repo_snapshot_packet.disposition,
      "inspected_contradictory",
    )
    self.assertEqual(report.report_status, "admissible")

  def test_schema_describes_disposition_taxonomy_boundary(self) -> None:
    schema = ProjectManagerReport.model_json_schema()
    coverage_entry = schema["$defs"]["ReportSourceCoverageEntry"]
    disposition_description = coverage_entry["properties"]["disposition"][
      "description"
    ]
    basis_description = coverage_entry["properties"]["basis"]["description"]

    self.assertEqual(
      tuple(coverage_entry["properties"]["disposition"]["enum"]),
      get_args(SourceCoverageDisposition),
    )
    self.assertIn("not that it is binding authority", disposition_description)
    self.assertIn("did not substantiate the requested claim family", disposition_description)
    self.assertIn("without independently determining report_status", disposition_description)
    self.assertIn("not parsed for validation", basis_description)

  def test_report_source_coverage_accepts_unconsumed_dispositions(self) -> None:
    for disposition in ("missing", "invalid", "not_required_for_task"):
      with self.subTest(disposition=disposition):
        report = ProjectManagerReport.model_validate(
          build_report_data(
            report_status="needs_clarification",
            blocked=False,
            blocking_reason=None,
            missing_basis=["Clarify the target surface."],
            constraint_conflicts=[],
            next_admissible_transformation="Ask the user to name the target surface.",
            repo_snapshot_packet_consumed=False,
            repo_snapshot_packet_disposition=disposition,
            repo_snapshot_packet_basis=[
              f"The repo snapshot was classified as {disposition} for this task.",
            ],
          )
        )

        self.assertFalse(report.report_source_coverage.repo_snapshot_packet.consumed)
        self.assertEqual(
          report.report_source_coverage.repo_snapshot_packet.disposition,
          disposition,
        )

  def test_report_source_coverage_rejects_consumed_disposition_mismatches(self) -> None:
    cases = (
      (False, "used"),
      (False, "inspected_insufficient"),
      (False, "inspected_not_relevant"),
      (False, "inspected_contradictory"),
      (True, "missing"),
      (True, "invalid"),
      (True, "not_required_for_task"),
    )
    for consumed, disposition in cases:
      with self.subTest(consumed=consumed, disposition=disposition):
        with self.assertRaises(ValueError):
          ProjectManagerReport.model_validate(
            build_report_data(
              report_status="needs_clarification",
              blocked=False,
              blocking_reason=None,
              missing_basis=["Clarify the target surface."],
              constraint_conflicts=[],
              next_admissible_transformation="Ask the user to name the target surface.",
              repo_snapshot_packet_consumed=consumed,
              repo_snapshot_packet_disposition=disposition,
            )
          )

  def test_required_static_context_cannot_be_unconsumed(self) -> None:
    report_data = build_report_data(
      report_status="needs_clarification",
      blocked=False,
      blocking_reason=None,
      missing_basis=["Clarify the target surface."],
      constraint_conflicts=[],
      next_admissible_transformation="Ask the user to name the target surface.",
    )
    report_data["report_source_coverage"]["static_context_packet"] = {
      "consumed": False,
      "disposition": "not_required_for_task",
      "basis": ["Static context was incorrectly classified as not required."],
    }

    with self.assertRaises(ValueError):
      ProjectManagerReport.model_validate(report_data)

  def test_rejected_can_be_unblocked_with_next_move(self) -> None:
    raw_response = load_json(CANONICAL_REJECTED_UNBLOCKED_RESPONSE_PATH)
    report = ProjectManagerReport.model_validate(
      json.loads(raw_response["output_text"])
    )

    self.assertEqual(report.report_status, "rejected")
    self.assertFalse(report.proof_frontier.blocked)
    self.assertIsNone(report.proof_frontier.blocking_reason)
    self.assertIsNotNone(report.report_source_coverage.repo_snapshot_packet)
    self.assertTrue(
      any(
        "harness/runs/20260612-214948-agent-route/project_manager_report.json"
        in basis
        for basis in report.report_source_coverage.repo_snapshot_packet.basis
      )
    )
    self.assertEqual(
      report.proof_frontier.next_admissible_transformation,
      "Classify the ledger as evidence of a recorded API call, not as proof of runtime state; if runtime proof is needed, require the corresponding saved run artifacts and validation/probe outputs.",
    )

  def test_needs_clarification_can_be_unblocked_with_next_move(self) -> None:
    report = ProjectManagerReport.model_validate(
      build_report_data(
        report_status="needs_clarification",
        blocked=False,
        blocking_reason=None,
        missing_basis=["Clarify the target surface."],
        constraint_conflicts=[],
        next_admissible_transformation="Ask the user to name the target surface.",
      )
    )

    self.assertEqual(report.report_status, "needs_clarification")
    self.assertFalse(report.proof_frontier.blocked)
    self.assertIsNotNone(report.report_source_coverage.repo_snapshot_packet)
    self.assertEqual(report.proof_frontier.missing_basis, ["Clarify the target surface."])

  def test_admissibility_blocked_can_be_unblocked_with_next_move(self) -> None:
    report = ProjectManagerReport.model_validate(
      build_report_data(
        report_status="admissibility_blocked",
        blocked=False,
        blocking_reason=None,
        missing_basis=["Saved runtime artifacts are absent."],
        constraint_conflicts=[],
        next_admissible_transformation=(
          "Request the missing saved runtime artifacts before judging runtime state."
        ),
      )
    )

    self.assertEqual(report.report_status, "admissibility_blocked")
    self.assertFalse(report.proof_frontier.blocked)
    self.assertIsNotNone(report.report_source_coverage.repo_snapshot_packet)
    self.assertEqual(
      report.proof_frontier.next_admissible_transformation,
      "Request the missing saved runtime artifacts before judging runtime state.",
    )

  def test_blocked_frontier_requires_blocking_reason(self) -> None:
    with self.assertRaises(ValueError):
      ProjectManagerReport.model_validate(
        build_report_data(
          report_status="rejected",
          blocked=True,
          blocking_reason=None,
          missing_basis=[],
          constraint_conflicts=["Repo snapshot ledger attachment treated as runtime proof."],
          next_admissible_transformation=None,
        )
      )

  def test_blocked_frontier_can_still_name_next_move(self) -> None:
    report = ProjectManagerReport.model_validate(
      build_report_data(
        report_status="needs_clarification",
        blocked=True,
        blocking_reason="The task does not specify the target surface.",
        missing_basis=["Clarify the target surface."],
        constraint_conflicts=[],
        next_admissible_transformation=(
          "Ask the user to name the target surface before proceeding."
        ),
      )
    )

    self.assertEqual(report.report_status, "needs_clarification")
    self.assertTrue(report.proof_frontier.blocked)
    self.assertEqual(
      report.proof_frontier.blocking_reason,
      "The task does not specify the target surface.",
    )
    self.assertEqual(
      report.proof_frontier.next_admissible_transformation,
      "Ask the user to name the target surface before proceeding.",
    )

  def test_unblocked_frontier_forbids_blocking_reason(self) -> None:
    with self.assertRaises(ValueError):
      ProjectManagerReport.model_validate(
        build_report_data(
          report_status="needs_clarification",
          blocked=False,
          blocking_reason="should not be present",
          missing_basis=["Clarify the target surface."],
          constraint_conflicts=[],
          next_admissible_transformation="Ask the user to name the target surface.",
        )
      )


if __name__ == "__main__":
  unittest.main()
