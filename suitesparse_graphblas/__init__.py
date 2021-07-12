from ._graphblas import ffi, lib  # noqa
from . import utils
from ._version import get_versions
from . import exceptions as ex


def is_initialized():
    """Is GraphBLAS initialized via GrB_init or GxB_init?"""
    return lib.GxB_Global_Option_get(lib.GxB_MODE, ffi.new("GrB_Mode*")) != lib.GrB_PANIC


def supports_complex():
    """Does this package support complex numbers?"""
    return hasattr(lib, "GrB_FC64") or hasattr(lib, "GxB_FC64")


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


def libget(name):
    """Helper to get items from GraphBLAS which might be GrB or GxB"""
    try:
        return getattr(lib, name)
    except AttributeError:
        ext_name = f"GxB_{name[4:]}"
        try:
            return getattr(lib, ext_name)
        except AttributeError:
            pass
        raise


bool_types = frozenset((lib.GrB_BOOL,))

signed_integer_types = frozenset(
    (
        lib.GrB_INT8,
        lib.GrB_INT16,
        lib.GrB_INT32,
        lib.GrB_INT64,
    )
)

unsigned_integer_types = frozenset(
    (
        lib.GrB_UINT8,
        lib.GrB_UINT16,
        lib.GrB_UINT32,
        lib.GrB_UINT64,
    )
)

integer_types = signed_integer_types | unsigned_integer_types

real_types = frozenset(
    (
        lib.GrB_FP32,
        lib.GrB_FP64,
    )
)

if supports_complex():
    complex_types = frozenset(
        (
            lib.GxB_FC32,
            lib.GxB_FC64,
        )
    )
else:
    complex_types = frozenset()


grb_types = bool_types | integer_types | real_types | complex_types


_error_code_lookup = {
    # Warning
    lib.GrB_NO_VALUE: ex.NoValue,
    # API Errors
    lib.GrB_UNINITIALIZED_OBJECT: ex.UninitializedObject,
    lib.GrB_INVALID_OBJECT: ex.InvalidObject,
    lib.GrB_NULL_POINTER: ex.NullPointer,
    lib.GrB_INVALID_VALUE: ex.InvalidValue,
    lib.GrB_INVALID_INDEX: ex.InvalidIndex,
    lib.GrB_DOMAIN_MISMATCH: ex.DomainMismatch,
    lib.GrB_DIMENSION_MISMATCH: ex.DimensionMismatch,
    lib.GrB_OUTPUT_NOT_EMPTY: ex.OutputNotEmpty,
    # Execution Errors
    lib.GrB_OUT_OF_MEMORY: ex.OutOfMemory,
    lib.GrB_INSUFFICIENT_SPACE: ex.InsufficientSpace,
    lib.GrB_INDEX_OUT_OF_BOUNDS: ex.IndexOutOfBound,
    lib.GrB_PANIC: ex.Panic,
}
GrB_SUCCESS = lib.GrB_SUCCESS
GrB_NO_VALUE = lib.GrB_NO_VALUE


_error_func_lookup = {
    "struct GB_Type_opaque *": lib.GrB_Type_error,
    "struct GB_UnaryOp_opaque *": lib.GrB_UnaryOp_error,
    "struct GB_BinaryOp_opaque *": lib.GrB_BinaryOp_error,
    "struct GB_SelectOp_opaque *": lib.GxB_SelectOp_error,
    "struct GB_Monoid_opaque *": lib.GrB_Monoid_error,
    "struct GB_Semiring_opaque *": lib.GrB_Semiring_error,
    "struct GB_Scalar_opaque *": lib.GxB_Scalar_error,
    "struct GB_Matrix_opaque *": lib.GrB_Matrix_error,
    "struct GB_Vector_opaque *": lib.GrB_Vector_error,
    "struct GB_Descriptor_opaque *": lib.GrB_Descriptor_error,
}


def check_status(obj, response_code):
    """Check the return code of the GraphBLAS function.

    If the operation was successful, return None.

    If the operation returned no value return `exceptions.NoValue`.

    Otherwise it is an error, lookup the exception and the error
    description, and throw the exception.

    """
    if response_code == GrB_SUCCESS:
        return
    if response_code == GrB_NO_VALUE:
        return ex.NoValue

    if ffi.typeof(obj).item.kind == "pointer":
        obj = obj[0]

    cname = ffi.typeof(obj).cname
    error_func = _error_func_lookup.get(cname)
    if error_func is None:
        raise TypeError(f"Unknown cname {cname} looking up error string.")

    string = ffi.new("char**")
    error_func(string, obj)
    text = ffi.string(string[0]).decode()
    raise _error_code_lookup[response_code](text)
