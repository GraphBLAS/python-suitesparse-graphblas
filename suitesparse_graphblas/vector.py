from suitesparse_graphblas import (
    lib,
    ffi,
    check_status,
)


def free(v):
    """Free a vector."""
    check_status(v, lib.GrB_Vector_free(v))


def new(T, size=lib.GxB_INDEX_MAX, *, free=free):
    """Create a new `GrB_Vector` of type `T` and initialize it.

    >>> A = new(lib.GrB_UINT8, 2)
    >>> size(A)
    2

    The default `size` is `lib.GxB_INDEX_MAX`.

    >>> A = new(lib.GrB_UINT8)
    >>> size(A) == lib.GxB_INDEX_MAX
    True

    The `free` argument is called when the object is garbage
    collected, the default is `vector.free()`.  If `free` is None then
    there is no automatic garbage collection and it is up to the user
    to free the vector.
    """
    v = ffi.new("GrB_Vector*")
    check_status(v, lib.GrB_Vector_new(v, T, size))
    if free:
        return ffi.gc(v, free)
    return v


def type(v):
    """Return the GraphBLAS type of the vector.

    >>> v = new(lib.GrB_UINT8, 2)
    >>> type(v) == lib.GrB_UINT8
    True


    """
    T = ffi.new("GrB_Type*")
    check_status(v, lib.GxB_Vector_type(T, v[0]))
    return T[0]


def size(v):
    """Return the size of the vector.

    >>> v = new(lib.GrB_UINT8, 2)
    >>> size(v) == 2
    True

    """
    n = ffi.new("GrB_Index*")
    check_status(v, lib.GrB_Vector_size(n, v[0]))
    return n[0]


def nvals(v):
    """Return the number of stored elements in the vector.

    >>> v = new(lib.GrB_BOOL, 2)
    >>> nvals(v)
    0
    >>> set_bool(v, True, 1)
    >>> nvals(v)
    1

    """
    n = ffi.new("GrB_Index*")
    check_status(v, lib.GrB_Vector_nvals(n, v[0]))
    return n[0]


def set_bool(v, value, i):
    """Set a boolean value to the vector at position `i`.

    >>> v = new(lib.GrB_BOOL, 3)
    >>> set_bool(v, True, 2)
    >>> bool(v, 2) == True
    True

    """
    check_status(v, lib.GrB_Vector_setElement_BOOL(v[0], value, i))


def bool(v, i):
    """Get a boolean value from the vector at position `i`.

    >>> v = new(lib.GrB_BOOL, 3)
    >>> set_bool(v, True, 2)
    >>> bool(v, 2) == True
    True

    """
    value = ffi.new("bool*")
    check_status(v, lib.GrB_Vector_extractElement_BOOL(value, v[0], i))
    return value[0]
