import threading

import pytest

from suitesparse_graphblas import ffi, initialize, lib

try:
    from numpy._core._multiarray_umath import _set_numpy_warn_if_no_mem_policy
except ImportError:  # pragma: no cover
    try:  # NumPy < 2
        from numpy.core._multiarray_umath import _set_numpy_warn_if_no_mem_policy
    except ImportError:  # private, so it may go away
        _set_numpy_warn_if_no_mem_policy = None


def pytest_configure(config):
    # The marker tells pytest-run-parallel (`--parallel-threads`) to run a test in one thread.
    # The plugin registers it; without the plugin, an unknown marker fails `--strict-markers`
    # and `-W error`.
    if not config.pluginmanager.hasplugin("run-parallel"):
        config.addinivalue_line(
            "markers", "thread_unsafe: run the test in one thread under pytest-run-parallel"
        )


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


class _FreedInThisThread(threading.local):
    # Every thread starts with an empty list, so a free in any thread is recorded: an
    # AttributeError in the cffi callback would only be printed, and the free not seen.
    def __init__(self):
        self.addresses = []


# GraphBLAS may call the recording arena's functions until the process ends
_recording_arena_callbacks = []


@pytest.fixture(scope="session")
def recording_arena():
    """An arena that records the addresses it frees: ``(k, freed)``.

    `k` is its index.  Tests may run in several threads at once, so each thread records only
    its own frees, in ``freed.addresses``.  An arena can't be changed, so it's made once.  It
    needs cffi callbacks, which Pyodide doesn't have.
    """
    blocks = {}
    freed = _FreedInThisThread()

    @ffi.callback("void *(size_t)")
    def malloc(size):
        block = ffi.new("char[]", size or 1)
        blocks[int(ffi.cast("uintptr_t", block))] = block
        return block

    @ffi.callback("void (void *)")
    def free(ptr):
        address = int(ffi.cast("uintptr_t", ptr))
        blocks.pop(address, None)
        freed.addresses.append(address)

    k = lib.GxB_NARENAS - 2  # not 5: test_can_unclaim_buffer needs it uninitialized
    assert lib.GxB_arena_init(k, malloc, ffi.NULL, ffi.NULL, free) == lib.GrB_SUCCESS
    _recording_arena_callbacks.extend([malloc, free])
    return k, freed
