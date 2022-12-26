from . import _version
from . import exceptions as ex
from . import utils
from ._graphblas import ffi, lib

import struct
import platform

_is_osx_arm64 = platform.machine() == "arm64"
_is_ppc64le = platform.machine() == "ppc64le"
_c_float = ffi.typeof("float")
_c_double = ffi.typeof("double")


# It is strongly recommended to use the non-variadic version of functions to be
# compatible with the most number of architectures. For example, you should use
# GxB_Matrix_Option_get_INT32 instead of GxB_Matrix_Option_get.
if _is_osx_arm64 or _is_ppc64le:

    def vararg(val):
        # Interpret float as int32 and double as int64
        # https://devblogs.microsoft.com/oldnewthing/20220823-00/?p=107041
        tov = ffi.typeof(val)
        if tov == _c_float:
            val = struct.unpack("l", struct.pack("f", val))[0]
            val = ffi.cast("int64_t", val)
        elif tov == _c_double:
            val = struct.unpack("q", struct.pack("d", val))[0]
            val = ffi.cast("int64_t", val)
        # Cast variadic argument as char * to force it onto the stack where ARM64 expects it
        # https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms
        #
        # The same fix *may* work for ppc64le
        return ffi.cast("char *", val)

else:

    def vararg(val):
        return val


def is_initialized():
    """Is GraphBLAS initialized via GrB_init or GxB_init?"""
    mode = ffi.new("int32_t*")
    return lib.GxB_Global_Option_get_INT32(lib.GxB_MODE, mode) != lib.GrB_PANIC


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
    # See: https://github.com/GraphBLAS/python-suitesparse-graphblas/issues/40
    for attr in dir(lib):
        getattr(lib, attr)


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
    lib.GrB_EMPTY_OBJECT: ex.EmptyObject,
    # Execution Errors
    lib.GrB_OUT_OF_MEMORY: ex.OutOfMemory,
    lib.GrB_INSUFFICIENT_SPACE: ex.InsufficientSpace,
    lib.GrB_INDEX_OUT_OF_BOUNDS: ex.IndexOutOfBound,
    lib.GrB_PANIC: ex.Panic,
    lib.GrB_NOT_IMPLEMENTED: ex.NotImplementedException,
    # GxB Errors
    lib.GxB_EXHAUSTED: StopIteration,
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


class burble:
    """Control diagnostic output, and may be used as a context manager.

    Set up and simple usage:

    >>> from suitesparse_graphblas import burble, lib, matrix
    >>>
    >>> A = matrix.new(lib.GrB_BOOL, 3, 3)
    >>> burble.is_enabled
    False
    >>> burble.enable()
    >>> burble.is_enabled
    True
    >>> burble.disable()

    Example with explicit enable and disable:

    >>> burble.enable()
    >>> n = matrix.nvals(A)
      [ GrB_Matrix_nvals
         1.91e-06 sec ]
    >>> burble.disable()

    Example as a context manager:

    >>> with burble():
    >>>     n = matrix.nvals(A)
      [ GrB_Matrix_nvals
         1.91e-06 sec ]

    """

    def __init__(self):
        self._states = []

    @property
    def is_enabled(self):
        """Is burble enabled?"""
        val_ptr = ffi.new("int32_t*")
        info = lib.GxB_Global_Option_get_INT32(lib.GxB_BURBLE, val_ptr)
        if info != lib.GrB_SUCCESS:
            raise _error_code_lookup[info](
                "Failed to get burble status (has GraphBLAS been initialized?"
            )
        return val_ptr[0]

    def enable(self):
        """Enable diagnostic output"""
        info = lib.GxB_Global_Option_set_INT32(lib.GxB_BURBLE, ffi.cast("int32_t", 1))
        if info != lib.GrB_SUCCESS:
            raise _error_code_lookup[info](
                "Failed to enable burble (has GraphBLAS been initialized?"
            )

    def disable(self):
        """Disable diagnostic output"""
        info = lib.GxB_Global_Option_set_INT32(lib.GxB_BURBLE, ffi.cast("int32_t", 0))
        if info != lib.GrB_SUCCESS:
            raise _error_code_lookup[info](
                "Failed to disable burble (has GraphBLAS been initialized?"
            )

    def __enter__(self):
        is_enabled = self.is_enabled
        if not is_enabled:
            self.enable()
        self._states.append(is_enabled)
        return self

    def __exit__(self, type_, value, traceback):
        is_enabled = self._states.pop()
        if not is_enabled:
            self.disable()

    def __reduce__(self):
        return "burble"

    def __repr__(self):
        return f"<burble is_enabled={self.is_enabled}>"


burble = burble()


__version__ = _version.get_versions()["version"]
