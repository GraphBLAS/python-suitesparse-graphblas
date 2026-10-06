import pytest

from suitesparse_graphblas import initialize

try:
    from numpy._core._multiarray_umath import _set_numpy_warn_if_no_mem_policy
except ImportError:  # pragma: no cover
    try:  # NumPy < 2
        from numpy.core._multiarray_umath import _set_numpy_warn_if_no_mem_policy
    except ImportError:  # private, so it may go away
        _set_numpy_warn_if_no_mem_policy = None


@pytest.fixture(scope="session", autouse=True)
def intialize_suitesparse_graphblas():
    initialize()


@pytest.fixture(scope="session", autouse=True)
def warn_if_no_memory_handler():
    # NumPy frees the data of an array that owns it but has no memory handler with libc
    # `free`, which is only right by luck (see utils.pyx).  Make NumPy warn when it does;
    # the warning fails the test (see `filterwarnings` in pyproject.toml).
    if _set_numpy_warn_if_no_mem_policy is None:  # pragma: no cover
        yield
        return
    old = _set_numpy_warn_if_no_mem_policy(True)
    yield
    _set_numpy_warn_if_no_mem_policy(old)
