from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def serialization_dir():
    # Every round-trip test writes into <repo>/serialization/, which is gitignored and so absent
    # on a fresh checkout. The mini_test runner creates it; plain `pytest` must too.
    (Path(__file__).resolve().parent / "serialization").mkdir(parents=True, exist_ok=True)
