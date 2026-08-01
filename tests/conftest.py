from pathlib import Path

import pytest

from src.config.config import load_config


@pytest.fixture(scope="session")
def config():
    """
    Load the project configuration once for all tests.
    """
    return load_config(
        Path("configs/config.yaml")
    )
