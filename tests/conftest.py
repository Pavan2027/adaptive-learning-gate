"""
Shared pytest fixtures for the ALG test suite.
"""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def default_config_path(repo_root: Path) -> Path:
    """Absolute path to the default YAML configuration file."""
    return repo_root / "configs" / "default.yaml"
