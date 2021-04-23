from ._graphblas import ffi, lib  # noqa
from . import utils
from ._version import get_versions


def is_initialized():
    """Is GraphBLAS initialized via GrB_init or GxB_init?"""
    return lib.GxB_Global_Option_get(lib.GxB_MODE, ffi.new("GrB_Mode*")) != lib.GrB_PANIC


def initialize(*, blocking=False, memory_manager="numpy"):
    """Initialize GraphBLAS via GrB_init or GxB_init.

    This must be called before any other GraphBLAS functions are called.
    A RuntimeError will be raised if called more than once.

    Parameters
    ----------
    blocking : bool, optional
        Whether to call init with GrB_BLOCKING or GrB_NONBLOCKING.
        Default is False.
    memory_manager : {'numpy', 'c'}, optional
        Choose which malloc/free functions to use.  'numpy' uses numpy's
        allocators, which makes it safe to perform zero-copy to and from numpy,
        and allows Python to track memory usage via tracemalloc (if enabled).
        'c' uses the default allocators.  Default is 'numpy'.

    The global variable `suitesparse_graphblas.is_initialized` indicates whether
    GraphBLAS has been initialized.
    """
    if is_initialized():
        raise RuntimeError("GraphBLAS is already initialized!  Unable to initialize again.")
    blocking = lib.GrB_BLOCKING if blocking else lib.GrB_NONBLOCKING
    memory_manager = memory_manager.lower()
    if memory_manager == "numpy":
        utils.call_gxb_init(ffi, lib, blocking)
    elif memory_manager == "c":
        lib.GrB_init(blocking)
    else:
        raise ValueError(f'memory_manager argument must be "numpy" or "c"; got: {memory_manager!r}')


__version__ = get_versions()["version"]
del get_versions
