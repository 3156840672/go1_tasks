"""Shared fixtures for the quadruped task tests."""

import pytest
import warp as wp

import go1_tasks  # noqa: F401  Register Go1 tasks before test collection.
import go2_tasks  # noqa: F401  Register Go2 tasks before test collection.


@pytest.fixture(scope="session", autouse=True)
def configure_test_environment() -> None:
  """Configure test environment settings automatically for all tests."""
  wp.config.quiet = True
