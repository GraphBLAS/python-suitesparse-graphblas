from suitesparse_graphblas import (
    lib,
    ffi,
    check_status,
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
    v = ffi.new("GxB_Scalar*")
    check_status(v, lib.GxB_Scalar_new(v, T))
    if free:
        return ffi.gc(v, free)
    return v


def type(v):
    """Return the GraphBLAS type of the scalar.

    >>> S = new(lib.GrB_UINT8)
    >>> type(S) == lib.GrB_UINT8
    True

    """
    T = ffi.new("GrB_Type*")
    check_status(v, lib.GxB_Scalar_type(T, v[0]))
    return T[0]
