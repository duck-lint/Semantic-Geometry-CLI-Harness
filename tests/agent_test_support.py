from __future__ import annotations

import atexit
import json
import tempfile
from pathlib import Path


_TEMP_DIRECTORIES: list[tempfile.TemporaryDirectory[str]] = []


def create_test_project_manager_agent(live_agent_path: Path) -> Path:
  temp_directory = tempfile.TemporaryDirectory()
  _TEMP_DIRECTORIES.append(temp_directory)

  agent_data = json.loads(live_agent_path.read_text(encoding="utf-8"))
  for entry in agent_data["agent_input_policy"]:
    if entry["input_id"] == "repo_snapshot_packet":
      entry["resolution"] = {
        "mode": "paths",
        "paths": ["tests/fixtures/raw_model_response.json"],
        "include_harness": False,
      }
      break

  output_path = Path(temp_directory.name) / "project_manager.agent.json"
  output_path.write_text(
    json.dumps(agent_data, indent=2) + "\n",
    encoding="utf-8",
  )
  return output_path


def create_test_archive_manager_agent(live_agent_path: Path) -> Path:
  temp_directory = tempfile.TemporaryDirectory()
  _TEMP_DIRECTORIES.append(temp_directory)

  agent_data = json.loads(live_agent_path.read_text(encoding="utf-8"))
  for entry in agent_data["agent_input_policy"]:
    if entry["input_id"] == "repo_snapshot_packet":
      entry["resolution"] = {
        "mode": "paths",
        "paths": ["harness/contracts/archive_manager_report.example.json"],
        "include_harness": False,
      }
      break

  output_path = Path(temp_directory.name) / "archive_manager.agent.json"
  output_path.write_text(
    json.dumps(agent_data, indent=2) + "\n",
    encoding="utf-8",
  )
  return output_path


@atexit.register
def _cleanup_temp_directories() -> None:
  for temp_directory in _TEMP_DIRECTORIES:
    temp_directory.cleanup()
