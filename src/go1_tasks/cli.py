"""Console entry points that register the Go1 tasks before running mjlab."""

from __future__ import annotations

import go1_tasks
from mjlab.scripts.play import main as _play_main
from mjlab.scripts.train import main as _train_main


def train() -> None:
  """Run ``mjlab``'s training CLI with the Go1 tasks registered."""
  _train_main()


def play() -> None:
  """Run ``mjlab``'s play CLI with the Go1 tasks registered."""
  _play_main()


def eval_cts() -> None:
  """Evaluate a Go1 CTS checkpoint on flat ground."""
  from go1_tasks.eval_cts import main

  main()


def list_go1_tasks() -> None:
  """Print the registered Go1 task IDs."""
  for key, package_id in go1_tasks.TASK_IDS.items():
    print(f"{key:6s} {go1_tasks.task_id(key):40s} (package id: {package_id})")
