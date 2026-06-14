from __future__ import annotations

import json
import unittest
from pathlib import Path

from harness.contracts.project_manager_report import ProjectManagerReport


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
  include_repo_snapshot_packet: bool = True,
  repo_snapshot_packet_basis: list[str] | None = None,
  include_git_context: bool = True,
) -> dict:
  report_source_coverage: dict[str, dict[str, object]] = {
    "static_context_packet": {
      "consumed": True,
      "basis": [
        "Used project_spec, governance_primitives, open_decisions, active_implementation_plan, and active_implementation_tracker.",
      ],
    },
    "task": {
      "consumed": True,
      "basis": [
        "Used the task text as current task authority.",
      ],
    },
  }

  if include_repo_snapshot_packet:
    report_source_coverage["repo_snapshot_packet"] = {
      "consumed": True,
      "basis": (
        repo_snapshot_packet_basis
        if repo_snapshot_packet_basis is not None
        else [
          "Used repo_snapshot_packet as historical artifact evidence for the current PM slice.",
          "Historical artifact paths named in the repo snapshot packet included harness/runs/20260612-214948-agent-route/project_manager_report.json and harness/runs/20260612-214948-agent-route/raw_model_response.json.",
        ]
      ),
    }

  if include_git_context:
    report_source_coverage["git_context"] = {
      "consumed": True,
      "basis": [
        "Used git_context to distinguish current worktree provenance from saved run artifacts.",
      ],
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

  def test_report_source_coverage_allows_omitting_repo_snapshot_packet(self) -> None:
    report = ProjectManagerReport.model_validate(
      build_report_data(
        report_status="needs_clarification",
        blocked=False,
        blocking_reason=None,
        missing_basis=["Clarify the target surface."],
        constraint_conflicts=[],
        next_admissible_transformation="Ask the user to name the target surface.",
        include_repo_snapshot_packet=False,
      )
    )

    self.assertIsNone(report.report_source_coverage.repo_snapshot_packet)

  def test_report_source_coverage_rejects_empty_basis(self) -> None:
    with self.assertRaises(ValueError):
      ProjectManagerReport.model_validate(
        build_report_data(
          report_status="needs_clarification",
          blocked=False,
          blocking_reason=None,
          missing_basis=["Clarify the target surface."],
          constraint_conflicts=[],
          next_admissible_transformation="Ask the user to name the target surface.",
          repo_snapshot_packet_basis=[],
        )
      )

  def test_report_source_coverage_rejects_unconsumed_supplied_source(self) -> None:
    report_data = build_report_data(
      report_status="needs_clarification",
      blocked=False,
      blocking_reason=None,
      missing_basis=["Clarify the target surface."],
      constraint_conflicts=[],
      next_admissible_transformation="Ask the user to name the target surface.",
    )
    report_data["report_source_coverage"]["repo_snapshot_packet"]["consumed"] = False

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
