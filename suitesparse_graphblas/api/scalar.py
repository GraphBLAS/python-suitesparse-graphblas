"""Create, manipulate, and query GxB_Scalar objects."""

from suitesparse_graphblas import check_status, exceptions, ffi, lib, supports_complex

from .utils import _capture_c_output  # noqa: F401


def scalar_free(v):
    """Free a scalar."""
    check_status(v, lib.GxB_Scalar_free(v))


def scalar_new(T, *, free=scalar_free):
    """Create a new `GxB_Scalar` of type `T` and initialize it.

    The `free` argument is called when the object is garbage
    collected, the default is `scalar.scalar_free()`.  If `free` is None then
    there is no automatic garbage collection and it is up to the user
    to free the scalar.

    >>> S = scalar_new(lib.GrB_UINT8)

    """
    s = ffi.new("GxB_Scalar*")
    check_status(s, lib.GxB_Scalar_new(s, T))
    if free:
        return ffi.gc(s, free)
    return s


def scalar_type(s):
    """Return the GraphBLAS type of the scalar.

    >>> S = scalar_new(lib.GrB_UINT8)
    >>> scalar_type(S) == lib.GrB_UINT8
    True

    """
    T = ffi.new("GrB_Type*")
    check_status(s, lib.GxB_Scalar_type(T, s[0]))
    return T[0]


# ---------------------------------------------------------------------------
# GraphBLAS operations
# ---------------------------------------------------------------------------


def scalar_dup(s, *, free=scalar_free):
    """Duplicate a scalar.

    >>> s = scalar_new(lib.GrB_INT64)
    >>> set_int64(s, 42)
    >>> t = scalar_dup(s)
    >>> get_int64(t) == 42
    True

    """
    t = ffi.new("GxB_Scalar*")
    check_status(t, lib.GrB_Scalar_dup(t, s[0]))
    if free:
        return ffi.gc(t, free)
    return t


def scalar_clear(s):
    """Remove the entry from a scalar.

    >>> s = scalar_new(lib.GrB_INT64)
    >>> set_int64(s, 42)
    >>> get_int64(s) == 42
    True
    >>> scalar_clear(s)
    >>> get_int64(s) is None
    True

    """
    check_status(s, lib.GrB_Scalar_clear(s[0]))


def scalar_wait(s, waitmode=lib.GrB_COMPLETE):
    """Wait for a scalar to complete pending operations.

    >>> s = scalar_new(lib.GrB_INT64)
    >>> scalar_wait(s)

    """
    check_status(s, lib.GrB_Scalar_wait(s[0], waitmode))


def scalar_print(s, name="", level=lib.GxB_COMPLETE):
    """Print a scalar to stdout.

    ``level`` controls verbosity: ``lib.GxB_SILENT``, ``lib.GxB_SUMMARY``,
    ``lib.GxB_SHORT``, ``lib.GxB_COMPLETE``, ``lib.GxB_SHORT_VERBOSE``,
    or ``lib.GxB_COMPLETE_VERBOSE``.

    >>> s = scalar_new(lib.GrB_INT64)
    >>> set_int64(s, 42)
    >>> out = _capture_c_output(scalar_print, s, 's', lib.GxB_SHORT)
    >>> 'int64_t scalar' in out
    True
    >>> '(0,0)   42' in out
    True

    """
    check_status(s, lib.GxB_Scalar_fprint(
        s[0], name.encode() if isinstance(name, str) else name,
        level, ffi.NULL,
    ))


def scalar_fprint(s, f, name="", level=lib.GxB_COMPLETE):
    """Print a scalar to a C FILE* stream.

    Pass ``ffi.NULL`` for ``f`` to print to stdout.
    ``level`` controls verbosity (see :func:`scalar_print`).

    >>> s = scalar_new(lib.GrB_INT64)
    >>> set_int64(s, 7)
    >>> out = _capture_c_output(scalar_fprint, s, ffi.NULL, 's', lib.GxB_SHORT)
    >>> 'int64_t scalar' in out
    True
    >>> '(0,0)   7' in out
    True

    """
    check_status(s, lib.GxB_Scalar_fprint(
        s[0], name.encode() if isinstance(name, str) else name,
        level, f,
    ))


def set_bool(s, value):
    """Set a boolean value to the scalar.

    >>> s = scalar_new(lib.GrB_BOOL)
    >>> set_bool(s, True)
    >>> get_bool(s) == True
    True

    """
    check_status(s, lib.GxB_Scalar_setElement_BOOL(s[0], value))


def get_bool(s):
    """Get a boolean value from the scalar.

    >>> s = scalar_new(lib.GrB_BOOL)
    >>> set_bool(s, True)
    >>> get_bool(s) == True
    True

    """
    value = ffi.new("bool*")
    res = check_status(s, lib.GxB_Scalar_extractElement_BOOL(value, s[0]))
    if res == exceptions.NoValue:
        return None
    return value[0]


def set_int8(s, value):
    """Set an int8 value to the scalar.

    >>> s = scalar_new(lib.GrB_INT8)
    >>> set_int8(s, 7)
    >>> get_int8(s) == 7
    True

    """
    check_status(s, lib.GxB_Scalar_setElement_INT8(s[0], value))


def get_int8(s):
    """Get an int8 value from the scalar.

    >>> s = scalar_new(lib.GrB_INT8)
    >>> set_int8(s, 7)
    >>> get_int8(s) == 7
    True

    """
    value = ffi.new("int8_t*")
    res = check_status(s, lib.GxB_Scalar_extractElement_INT8(value, s[0]))
    if res == exceptions.NoValue:
        return None
    return value[0]


def set_int16(s, value):
    """Set an int16 value to the scalar.

    >>> s = scalar_new(lib.GrB_INT16)
    >>> set_int16(s, 7)
    >>> get_int16(s) == 7
    True

    """
    check_status(s, lib.GxB_Scalar_setElement_INT16(s[0], value))


def get_int16(s):
    """Get an int16 value from the scalar.

    >>> s = scalar_new(lib.GrB_INT16)
    >>> set_int16(s, 7)
    >>> get_int16(s) == 7
    True

    """
    value = ffi.new("int16_t*")
    res = check_status(s, lib.GxB_Scalar_extractElement_INT16(value, s[0]))
    if res == exceptions.NoValue:
        return None
    return value[0]


def set_int32(s, value):
    """Set an int32 value to the scalar.

    >>> s = scalar_new(lib.GrB_INT32)
    >>> set_int32(s, 7)
    >>> get_int32(s) == 7
    True

    """
    check_status(s, lib.GxB_Scalar_setElement_INT32(s[0], value))


def get_int32(s):
    """Get an int32 value from the scalar.

    >>> s = scalar_new(lib.GrB_INT32)
    >>> set_int32(s, 7)
    >>> get_int32(s) == 7
    True

    """
    value = ffi.new("int32_t*")
    res = check_status(s, lib.GxB_Scalar_extractElement_INT32(value, s[0]))
    if res == exceptions.NoValue:
        return None
    return value[0]


def set_int64(s, value):
    """Set an int64 value to the scalar.

    >>> s = scalar_new(lib.GrB_INT64)
    >>> set_int64(s, 7)
    >>> get_int64(s) == 7
    True

    """
    check_status(s, lib.GxB_Scalar_setElement_INT64(s[0], value))


def get_int64(s):
    """Get an int64 value from the scalar.

    >>> s = scalar_new(lib.GrB_INT64)
    >>> set_int64(s, 7)
    >>> get_int64(s) == 7
    True

    """
    value = ffi.new("int64_t*")
    res = check_status(s, lib.GxB_Scalar_extractElement_INT64(value, s[0]))
    if res == exceptions.NoValue:
        return None
    return value[0]


def set_uint8(s, value):
    """Set a uint8 value to the scalar.

    >>> s = scalar_new(lib.GrB_UINT8)
    >>> set_uint8(s, 7)
    >>> get_uint8(s) == 7
    True

    """
    check_status(s, lib.GxB_Scalar_setElement_UINT8(s[0], value))


def get_uint8(s):
    """Get a uint8 value from the scalar.

    >>> s = scalar_new(lib.GrB_UINT8)
    >>> set_uint8(s, 7)
    >>> get_uint8(s) == 7
    True

    """
    value = ffi.new("uint8_t*")
    res = check_status(s, lib.GxB_Scalar_extractElement_UINT8(value, s[0]))
    if res == exceptions.NoValue:
        return None
    return value[0]


def set_uint16(s, value):
    """Set a uint16 value to the scalar.

    >>> s = scalar_new(lib.GrB_UINT16)
    >>> set_uint16(s, 7)
    >>> get_uint16(s) == 7
    True

    """
    check_status(s, lib.GxB_Scalar_setElement_UINT16(s[0], value))


def get_uint16(s):
    """Get a uint16 value from the scalar.

    >>> s = scalar_new(lib.GrB_UINT16)
    >>> set_uint16(s, 7)
    >>> get_uint16(s) == 7
    True

    """
    value = ffi.new("uint16_t*")
    res = check_status(s, lib.GxB_Scalar_extractElement_UINT16(value, s[0]))
    if res == exceptions.NoValue:
        return None
    return value[0]


def set_uint32(s, value):
    """Set a uint32 value to the scalar.

    >>> s = scalar_new(lib.GrB_UINT32)
    >>> set_uint32(s, 7)
    >>> get_uint32(s) == 7
    True

    """
    check_status(s, lib.GxB_Scalar_setElement_UINT32(s[0], value))


def get_uint32(s):
    """Get a uint32 value from the scalar.

    >>> s = scalar_new(lib.GrB_UINT32)
    >>> set_uint32(s, 7)
    >>> get_uint32(s) == 7
    True

    """
    value = ffi.new("uint32_t*")
    res = check_status(s, lib.GxB_Scalar_extractElement_UINT32(value, s[0]))
    if res == exceptions.NoValue:
        return None
    return value[0]


def set_uint64(s, value):
    """Set a uint64 value to the scalar.

    >>> s = scalar_new(lib.GrB_UINT64)
    >>> set_uint64(s, 7)
    >>> get_uint64(s) == 7
    True

    """
    check_status(s, lib.GxB_Scalar_setElement_UINT64(s[0], value))


def get_uint64(s):
    """Get a uint64 value from the scalar.

    >>> s = scalar_new(lib.GrB_UINT64)
    >>> set_uint64(s, 7)
    >>> get_uint64(s) == 7
    True

    """
    value = ffi.new("uint64_t*")
    res = check_status(s, lib.GxB_Scalar_extractElement_UINT64(value, s[0]))
    if res == exceptions.NoValue:
        return None
    return value[0]


def set_fp32(s, value):
    """Set an fp32 value to the scalar.

    >>> s = scalar_new(lib.GrB_FP32)
    >>> set_fp32(s, 1.5)
    >>> get_fp32(s) == 1.5
    True

    """
    check_status(s, lib.GxB_Scalar_setElement_FP32(s[0], value))


def get_fp32(s):
    """Get an fp32 value from the scalar.

    >>> s = scalar_new(lib.GrB_FP32)
    >>> set_fp32(s, 1.5)
    >>> get_fp32(s) == 1.5
    True

    """
    value = ffi.new("float*")
    res = check_status(s, lib.GxB_Scalar_extractElement_FP32(value, s[0]))
    if res == exceptions.NoValue:
        return None
    return value[0]


def set_fp64(s, value):
    """Set an fp64 value to the scalar.

    >>> s = scalar_new(lib.GrB_FP64)
    >>> set_fp64(s, 1.5)
    >>> get_fp64(s) == 1.5
    True

    """
    check_status(s, lib.GxB_Scalar_setElement_FP64(s[0], value))


def get_fp64(s):
    """Get an fp64 value from the scalar.

    >>> s = scalar_new(lib.GrB_FP64)
    >>> set_fp64(s, 1.5)
    >>> get_fp64(s) == 1.5
    True

    """
    value = ffi.new("double*")
    res = check_status(s, lib.GxB_Scalar_extractElement_FP64(value, s[0]))
    if res == exceptions.NoValue:
        return None
    return value[0]


if supports_complex():

    def set_fc32(s, value):
        """Set an fc32 value to the scalar.

        >>> s = scalar_new(lib.GxB_FC32)
        >>> set_fc32(s, 2+3j)
        >>> get_fc32(s) == 2+3j
        True

        """
        check_status(s, lib.GxB_Scalar_setElement_FC32(s[0], value))

    def get_fc32(s):
        """Get an fc32 value from the scalar.

        >>> s = scalar_new(lib.GxB_FC32)
        >>> set_fc32(s, 2+3j)
        >>> get_fc32(s) == 2+3j
        True

        """
        value = ffi.new("GxB_FC32_t*")
        res = check_status(s, lib.GxB_Scalar_extractElement_FC32(value, s[0]))
        if res == exceptions.NoValue:
            return None
        return value[0]

    def set_fc64(s, value):
        """Set an fc64 value to the scalar.

        >>> s = scalar_new(lib.GxB_FC64)
        >>> set_fc64(s, 2+3j)
        >>> get_fc64(s) == 2+3j
        True

        """
        check_status(s, lib.GxB_Scalar_setElement_FC64(s[0], value))

    def get_fc64(s):
        """Get an fc64 value from the scalar.

        >>> s = scalar_new(lib.GxB_FC64)
        >>> set_fc64(s, 2+3j)
        >>> get_fc64(s) == 2+3j
        True

        """
        value = ffi.new("GxB_FC64_t*")
        res = check_status(s, lib.GxB_Scalar_extractElement_FC64(value, s[0]))
        if res == exceptions.NoValue:
            return None
        return value[0]
