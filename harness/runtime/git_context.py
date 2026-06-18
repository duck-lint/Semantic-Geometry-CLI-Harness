from __future__ import annotations

import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class GitContext(BaseModel):
  model_config = ConfigDict(extra="forbid")

  available: bool
  branch: str | None = None
  commit: str | None = None
  is_dirty: bool | None = None
  status_summary: list[str] = Field(default_factory=list)
  failure: str | None = None


class GitCommitSummary(BaseModel):
  model_config = ConfigDict(extra="forbid")

  commit: str
  summary: str


class GitChangedFile(BaseModel):
  model_config = ConfigDict(extra="forbid")

  path: str
  status: str


class GitDeltaContext(BaseModel):
  model_config = ConfigDict(extra="forbid")

  available: bool
  base_commit: str | None = None
  head_commit: str | None = None
  comparison_range: str | None = None
  base_is_ancestor_of_head: bool | None = None
  commits_since_base: list[GitCommitSummary] = Field(default_factory=list)
  changed_files_since_base: list[GitChangedFile] = Field(default_factory=list)
  diff_stat_since_base: str | None = None
  worktree_state: str | None = None
  uncommitted_status_summary: list[str] = Field(default_factory=list)
  lineage_limitation: str | None = None
  failure: str | None = None


def _run_git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
    ["git", *args],
    cwd=repo_root,
    capture_output=True,
    text=True,
    check=False,
    timeout=10,
  )


def collect_git_context(repo_root: Path) -> GitContext:
  try:
    inside_work_tree = _run_git(repo_root, "rev-parse", "--is-inside-work-tree")
  except (OSError, subprocess.SubprocessError) as error:
    return GitContext(available=False, failure=str(error))

  if (
    inside_work_tree.returncode != 0
    or inside_work_tree.stdout.strip() != "true"
  ):
    failure = inside_work_tree.stderr.strip() or "Directory is not a git worktree."
    return GitContext(available=False, failure=failure)

  try:
    branch_result = _run_git(repo_root, "branch", "--show-current")
    commit_result = _run_git(repo_root, "rev-parse", "HEAD")
    status_result = _run_git(repo_root, "status", "--porcelain=v1")
  except (OSError, subprocess.SubprocessError) as error:
    return GitContext(available=False, failure=str(error))

  failed_commands = [
    result.stderr.strip()
    for result in (branch_result, commit_result, status_result)
    if result.returncode != 0
  ]
  if failed_commands:
    return GitContext(
      available=False,
      failure="; ".join(message for message in failed_commands if message),
    )

  status_summary = [
    line for line in status_result.stdout.splitlines() if line.strip()
  ]
  return GitContext(
    available=True,
    branch=branch_result.stdout.strip() or None,
    commit=commit_result.stdout.strip() or None,
    is_dirty=bool(status_summary),
    status_summary=status_summary,
  )


def _git_error(result: subprocess.CompletedProcess[str]) -> str:
  return result.stderr.strip() or result.stdout.strip()


def _parse_commit_summaries(output: str) -> list[GitCommitSummary]:
  summaries: list[GitCommitSummary] = []
  for line in output.splitlines():
    stripped = line.strip()
    if not stripped:
      continue
    commit, _, summary = stripped.partition(" ")
    summaries.append(GitCommitSummary(commit=commit, summary=summary))
  return summaries


def _parse_changed_files(output: str) -> list[GitChangedFile]:
  changed_files: list[GitChangedFile] = []
  for line in output.splitlines():
    stripped = line.strip()
    if not stripped:
      continue
    status, _, path = stripped.partition("\t")
    if not path:
      status, _, path = stripped.partition(" ")
    changed_files.append(GitChangedFile(status=status, path=path.strip()))
  return changed_files


def collect_git_delta_context(repo_root: Path, base_commit: str) -> GitDeltaContext:
  try:
    inside_work_tree = _run_git(repo_root, "rev-parse", "--is-inside-work-tree")
  except (OSError, subprocess.SubprocessError) as error:
    return GitDeltaContext(available=False, failure=str(error))

  if (
    inside_work_tree.returncode != 0
    or inside_work_tree.stdout.strip() != "true"
  ):
    failure = inside_work_tree.stderr.strip() or "Directory is not a git worktree."
    return GitDeltaContext(available=False, failure=failure)

  try:
    base_result = _run_git(
      repo_root,
      "rev-parse",
      "--verify",
      f"{base_commit}^{{commit}}",
    )
    head_result = _run_git(repo_root, "rev-parse", "HEAD")
  except (OSError, subprocess.SubprocessError) as error:
    return GitDeltaContext(available=False, failure=str(error))

  if base_result.returncode != 0:
    return GitDeltaContext(
      available=False,
      failure=_git_error(base_result)
      or f"Base commit does not resolve in this repository: {base_commit}",
    )

  if head_result.returncode != 0:
    return GitDeltaContext(
      available=False,
      failure=_git_error(head_result) or "Could not resolve HEAD.",
    )

  resolved_base_commit = base_result.stdout.strip()
  head_commit = head_result.stdout.strip()
  comparison_range = f"{resolved_base_commit}..{head_commit}"

  try:
    ancestor_result = _run_git(
      repo_root,
      "merge-base",
      "--is-ancestor",
      resolved_base_commit,
      head_commit,
    )
    commits_result = _run_git(
      repo_root,
      "rev-list",
      "--oneline",
      comparison_range,
    )
    files_result = _run_git(
      repo_root,
      "diff",
      "--name-status",
      comparison_range,
    )
    diff_stat_result = _run_git(repo_root, "diff", "--stat", comparison_range)
    status_result = _run_git(repo_root, "status", "--porcelain=v1")
  except (OSError, subprocess.SubprocessError) as error:
    return GitDeltaContext(
      available=False,
      base_commit=resolved_base_commit,
      head_commit=head_commit,
      comparison_range=comparison_range,
      failure=str(error),
    )

  if ancestor_result.returncode == 0:
    base_is_ancestor = True
    lineage_limitation = None
  elif ancestor_result.returncode == 1:
    base_is_ancestor = False
    lineage_limitation = (
      "Base commit is not an ancestor of HEAD; this comparison is not a clean "
      "archive lineage."
    )
  else:
    return GitDeltaContext(
      available=False,
      base_commit=resolved_base_commit,
      head_commit=head_commit,
      comparison_range=comparison_range,
      failure=_git_error(ancestor_result)
      or "Could not determine whether base commit is an ancestor of HEAD.",
    )

  failed_commands = [
    _git_error(result)
    for result in (commits_result, files_result, diff_stat_result, status_result)
    if result.returncode != 0
  ]
  if failed_commands:
    return GitDeltaContext(
      available=False,
      base_commit=resolved_base_commit,
      head_commit=head_commit,
      comparison_range=comparison_range,
      base_is_ancestor_of_head=base_is_ancestor,
      lineage_limitation=lineage_limitation,
      failure="; ".join(message for message in failed_commands if message),
    )

  uncommitted_status_summary = [
    line for line in status_result.stdout.splitlines() if line.strip()
  ]
  return GitDeltaContext(
    available=True,
    base_commit=resolved_base_commit,
    head_commit=head_commit,
    comparison_range=comparison_range,
    base_is_ancestor_of_head=base_is_ancestor,
    commits_since_base=_parse_commit_summaries(commits_result.stdout),
    changed_files_since_base=_parse_changed_files(files_result.stdout),
    diff_stat_since_base=diff_stat_result.stdout.strip() or None,
    worktree_state="dirty" if uncommitted_status_summary else "clean",
    uncommitted_status_summary=uncommitted_status_summary,
    lineage_limitation=lineage_limitation,
  )
