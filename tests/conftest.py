import os
from pathlib import Path

import pytest

# allow "import src..." when running pytest from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault


@pytest.fixture(scope="session")
def sample_n():
    # keep tests fast
    return 2000