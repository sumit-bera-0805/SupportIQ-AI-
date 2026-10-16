import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_sample  # noqa: E402


@pytest.fixture(scope="session")
def sample():
    return load_sample()[1]
