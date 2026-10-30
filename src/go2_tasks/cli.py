"""Console entry points that register the Go2 tasks."""

from __future__ import annotations

import go2_tasks
from mjlab.scripts.play import main as _play_main
from mjlab.scripts.train import main as _train_main


def train() -> None:
  _train_main()


def play() -> None:
  _play_main()


def list_go2_tasks() -> None:
  for key, task_id in go2_tasks.TASK_IDS.items():
    print(f"{key:6s} {task_id}")
