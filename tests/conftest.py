"""Shared fixtures for the Go1 task tests."""

import pytest
import warp as wp

import go1_tasks  # noqa: F401  Register tasks before test collection.


@pytest.fixture(scope="session", autouse=True)
def configure_test_environment() -> None:
  """Configure test environment settings automatically for all tests."""
  wp.config.quiet = True
