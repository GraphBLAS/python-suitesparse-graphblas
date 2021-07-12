from suitesparse_graphblas import (
    lib,
    ffi,
    check_status,
    exceptions,
)


def free(v):
    """Free a scalar."""
    check_status(v, lib.GxB_Scalar_free(v))


def new(T, *, free=free):
    """Create a new `GxB_Scalar` of type `T` and initialize it.

    The `free` argument is called when the object is garbage
    collected, the default is `scalar.free()`.  If `free` is None then
    there is no automatic garbage collection and it is up to the user
    to free the scalar.

    >>> S = new(lib.GrB_UINT8)

    """
    s = ffi.new("GxB_Scalar*")
    check_status(s, lib.GxB_Scalar_new(s, T))
    if free:
        return ffi.gc(s, free)
    return s


def type(s):
    """Return the GraphBLAS type of the scalar.

    >>> S = new(lib.GrB_UINT8)
    >>> type(S) == lib.GrB_UINT8
    True

    """
    T = ffi.new("GrB_Type*")
    check_status(s, lib.GxB_Scalar_type(T, s[0]))
    return T[0]


def set_bool(s, value):
    """Set a boolean value to the scalar.

    >>> s = new(lib.GrB_BOOL)
    >>> set_bool(s, True)
    >>> bool(s) == True
    True

    """
    check_status(s, lib.GxB_Scalar_setElement_BOOL(s[0], value))


def bool(s):
    """Get a boolean value from the scalar.

    >>> s = new(lib.GrB_BOOL)
    >>> set_bool(s, True)
    >>> bool(s) == True
    True

    """
    value = ffi.new("bool*")
    res = check_status(s, lib.GxB_Scalar_extractElement_BOOL(value, s[0]))
    if res == exceptions.NoValue:
        return None
    return value[0]
