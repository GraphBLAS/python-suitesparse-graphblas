from suitesparse_graphblas import check_status, ffi, lib, supports_complex

from .io.serialize import deserialize_vector as deserialize  # noqa: F401
from .io.serialize import serialize_vector as serialize  # noqa: F401


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
    >>> get_bool(v, 2) == True
    True

    """
    check_status(v, lib.GrB_Vector_setElement_BOOL(v[0], value, i))


def get_bool(v, i):
    """Get a boolean value from the vector at position `i`.

    >>> v = new(lib.GrB_BOOL, 3)
    >>> set_bool(v, True, 2)
    >>> get_bool(v, 2) == True
    True

    """
    value = ffi.new("bool*")
    check_status(v, lib.GrB_Vector_extractElement_BOOL(value, v[0], i))
    return value[0]


def set_int8(v, value, i):
    """Set an int8 value to the vector at position `i`.

    >>> v = new(lib.GrB_INT8, 3)
    >>> set_int8(v, 7, 2)
    >>> get_int8(v, 2) == 7
    True

    """
    check_status(v, lib.GrB_Vector_setElement_INT8(v[0], value, i))


def get_int8(v, i):
    """Get an int8 value from the vector at position `i`.

    >>> v = new(lib.GrB_INT8, 3)
    >>> set_int8(v, 7, 2)
    >>> get_int8(v, 2) == 7
    True

    """
    value = ffi.new("int8_t*")
    check_status(v, lib.GrB_Vector_extractElement_INT8(value, v[0], i))
    return value[0]


def set_int16(v, value, i):
    """Set an int16 value to the vector at position `i`.

    >>> v = new(lib.GrB_INT16, 3)
    >>> set_int16(v, 7, 2)
    >>> get_int16(v, 2) == 7
    True

    """
    check_status(v, lib.GrB_Vector_setElement_INT16(v[0], value, i))


def get_int16(v, i):
    """Get an int16 value from the vector at position `i`.

    >>> v = new(lib.GrB_INT16, 3)
    >>> set_int16(v, 7, 2)
    >>> get_int16(v, 2) == 7
    True

    """
    value = ffi.new("int16_t*")
    check_status(v, lib.GrB_Vector_extractElement_INT16(value, v[0], i))
    return value[0]


def set_int32(v, value, i):
    """Set an int32 value to the vector at position `i`.

    >>> v = new(lib.GrB_INT32, 3)
    >>> set_int32(v, 7, 2)
    >>> get_int32(v, 2) == 7
    True

    """
    check_status(v, lib.GrB_Vector_setElement_INT32(v[0], value, i))


def get_int32(v, i):
    """Get an int32 value from the vector at position `i`.

    >>> v = new(lib.GrB_INT32, 3)
    >>> set_int32(v, 7, 2)
    >>> get_int32(v, 2) == 7
    True

    """
    value = ffi.new("int32_t*")
    check_status(v, lib.GrB_Vector_extractElement_INT32(value, v[0], i))
    return value[0]


def set_int64(v, value, i):
    """Set an int64 value to the vector at position `i`.

    >>> v = new(lib.GrB_INT64, 3)
    >>> set_int64(v, 7, 2)
    >>> get_int64(v, 2) == 7
    True

    """
    check_status(v, lib.GrB_Vector_setElement_INT64(v[0], value, i))


def get_int64(v, i):
    """Get an int64 value from the vector at position `i`.

    >>> v = new(lib.GrB_INT64, 3)
    >>> set_int64(v, 7, 2)
    >>> get_int64(v, 2) == 7
    True

    """
    value = ffi.new("int64_t*")
    check_status(v, lib.GrB_Vector_extractElement_INT64(value, v[0], i))
    return value[0]


def set_uint8(v, value, i):
    """Set a uint8 value to the vector at position `i`.

    >>> v = new(lib.GrB_UINT8, 3)
    >>> set_uint8(v, 7, 2)
    >>> get_uint8(v, 2) == 7
    True

    """
    check_status(v, lib.GrB_Vector_setElement_UINT8(v[0], value, i))


def get_uint8(v, i):
    """Get a uint8 value from the vector at position `i`.

    >>> v = new(lib.GrB_UINT8, 3)
    >>> set_uint8(v, 7, 2)
    >>> get_uint8(v, 2) == 7
    True

    """
    value = ffi.new("uint8_t*")
    check_status(v, lib.GrB_Vector_extractElement_UINT8(value, v[0], i))
    return value[0]


def set_uint16(v, value, i):
    """Set a uint16 value to the vector at position `i`.

    >>> v = new(lib.GrB_UINT16, 3)
    >>> set_uint16(v, 7, 2)
    >>> get_uint16(v, 2) == 7
    True

    """
    check_status(v, lib.GrB_Vector_setElement_UINT16(v[0], value, i))


def get_uint16(v, i):
    """Get a uint16 value from the vector at position `i`.

    >>> v = new(lib.GrB_UINT16, 3)
    >>> set_uint16(v, 7, 2)
    >>> get_uint16(v, 2) == 7
    True

    """
    value = ffi.new("uint16_t*")
    check_status(v, lib.GrB_Vector_extractElement_UINT16(value, v[0], i))
    return value[0]


def set_uint32(v, value, i):
    """Set a uint32 value to the vector at position `i`.

    >>> v = new(lib.GrB_UINT32, 3)
    >>> set_uint32(v, 7, 2)
    >>> get_uint32(v, 2) == 7
    True

    """
    check_status(v, lib.GrB_Vector_setElement_UINT32(v[0], value, i))


def get_uint32(v, i):
    """Get a uint32 value from the vector at position `i`.

    >>> v = new(lib.GrB_UINT32, 3)
    >>> set_uint32(v, 7, 2)
    >>> get_uint32(v, 2) == 7
    True

    """
    value = ffi.new("uint32_t*")
    check_status(v, lib.GrB_Vector_extractElement_UINT32(value, v[0], i))
    return value[0]


def set_uint64(v, value, i):
    """Set a uint64 value to the vector at position `i`.

    >>> v = new(lib.GrB_UINT64, 3)
    >>> set_uint64(v, 7, 2)
    >>> get_uint64(v, 2) == 7
    True

    """
    check_status(v, lib.GrB_Vector_setElement_UINT64(v[0], value, i))


def get_uint64(v, i):
    """Get a uint64 value from the vector at position `i`.

    >>> v = new(lib.GrB_UINT64, 3)
    >>> set_uint64(v, 7, 2)
    >>> get_uint64(v, 2) == 7
    True

    """
    value = ffi.new("uint64_t*")
    check_status(v, lib.GrB_Vector_extractElement_UINT64(value, v[0], i))
    return value[0]


def set_fp32(v, value, i):
    """Set an fp32 value to the vector at position `i`.

    >>> v = new(lib.GrB_FP32, 3)
    >>> set_fp32(v, 1.5, 2)
    >>> get_fp32(v, 2) == 1.5
    True

    """
    check_status(v, lib.GrB_Vector_setElement_FP32(v[0], value, i))


def get_fp32(v, i):
    """Get an fp32 value from the vector at position `i`.

    >>> v = new(lib.GrB_FP32, 3)
    >>> set_fp32(v, 1.5, 2)
    >>> get_fp32(v, 2) == 1.5
    True

    """
    value = ffi.new("float*")
    check_status(v, lib.GrB_Vector_extractElement_FP32(value, v[0], i))
    return value[0]


def set_fp64(v, value, i):
    """Set an fp64 value to the vector at position `i`.

    >>> v = new(lib.GrB_FP64, 3)
    >>> set_fp64(v, 1.5, 2)
    >>> get_fp64(v, 2) == 1.5
    True

    """
    check_status(v, lib.GrB_Vector_setElement_FP64(v[0], value, i))


def get_fp64(v, i):
    """Get an fp64 value from the vector at position `i`.

    >>> v = new(lib.GrB_FP64, 3)
    >>> set_fp64(v, 1.5, 2)
    >>> get_fp64(v, 2) == 1.5
    True

    """
    value = ffi.new("double*")
    check_status(v, lib.GrB_Vector_extractElement_FP64(value, v[0], i))
    return value[0]


if supports_complex():

    def set_fc32(v, value, i):
        """Set an fc32 value to the vector at position `i`.

        >>> v = new(lib.GxB_FC32, 3)
        >>> set_fc32(v, 2+3j, 2)
        >>> get_fc32(v, 2) == 2+3j
        True

        """
        check_status(v, lib.GxB_Vector_setElement_FC32(v[0], value, i))

    def get_fc32(v, i):
        """Get an fc32 value from the vector at position `i`.

        >>> v = new(lib.GxB_FC32, 3)
        >>> set_fc32(v, 2+3j, 2)
        >>> get_fc32(v, 2) == 2+3j
        True

        """
        value = ffi.new("GxB_FC32_t*")
        check_status(v, lib.GxB_Vector_extractElement_FC32(value, v[0], i))
        return value[0]

    def set_fc64(v, value, i):
        """Set an fc64 value to the vector at position `i`.

        >>> v = new(lib.GxB_FC64, 3)
        >>> set_fc64(v, 2+3j, 2)
        >>> get_fc64(v, 2) == 2+3j
        True

        """
        check_status(v, lib.GxB_Vector_setElement_FC64(v[0], value, i))

    def get_fc64(v, i):
        """Get an fc64 value from the vector at position `i`.

        >>> v = new(lib.GxB_FC64, 3)
        >>> set_fc64(v, 2+3j, 2)
        >>> get_fc64(v, 2) == 2+3j
        True

        """
        value = ffi.new("GxB_FC64_t*")
        check_status(v, lib.GxB_Vector_extractElement_FC64(value, v[0], i))
        return value[0]
