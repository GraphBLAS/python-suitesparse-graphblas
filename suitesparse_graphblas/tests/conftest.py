import pytest
from suitesparse_graphblas import initialize


@pytest.fixture(scope="session", autouse=True)
def intialize_suitesparse_graphblas():
    initialize()
