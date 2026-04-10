from suitesparse_graphblas import check_status, ffi, lib, supports_complex

from .utils import _capture_c_output  # noqa: F401


def monoid_free(m):
    """Free a monoid.

    >>> m = monoid_new_int64(lib.GrB_PLUS_INT64, 0)
    >>> monoid_free(m)

    """
    check_status(m, lib.GrB_Monoid_free(m))


# ---------------------------------------------------------------------------
# Typed constructors
# ---------------------------------------------------------------------------


def monoid_new_bool(op, identity, *, free=monoid_free):
    """Create a new GrB_Monoid with a boolean identity.

    >>> m = monoid_new_bool(lib.GrB_LAND, True)
    >>> monoid_operator(m) == lib.GrB_LAND
    True

    """
    m = ffi.new("GrB_Monoid*")
    check_status(m, lib.GrB_Monoid_new_BOOL(m, op, identity))
    if free:
        return ffi.gc(m, free)
    return m


def monoid_new_int8(op, identity, *, free=monoid_free):
    """Create a new GrB_Monoid with an int8 identity.

    >>> m = monoid_new_int8(lib.GrB_PLUS_INT8, 0)
    >>> monoid_operator(m) == lib.GrB_PLUS_INT8
    True

    """
    m = ffi.new("GrB_Monoid*")
    check_status(m, lib.GrB_Monoid_new_INT8(m, op, identity))
    if free:
        return ffi.gc(m, free)
    return m


def monoid_new_int16(op, identity, *, free=monoid_free):
    """Create a new GrB_Monoid with an int16 identity.

    >>> m = monoid_new_int16(lib.GrB_PLUS_INT16, 0)
    >>> monoid_operator(m) == lib.GrB_PLUS_INT16
    True

    """
    m = ffi.new("GrB_Monoid*")
    check_status(m, lib.GrB_Monoid_new_INT16(m, op, identity))
    if free:
        return ffi.gc(m, free)
    return m


def monoid_new_int32(op, identity, *, free=monoid_free):
    """Create a new GrB_Monoid with an int32 identity.

    >>> m = monoid_new_int32(lib.GrB_PLUS_INT32, 0)
    >>> monoid_operator(m) == lib.GrB_PLUS_INT32
    True

    """
    m = ffi.new("GrB_Monoid*")
    check_status(m, lib.GrB_Monoid_new_INT32(m, op, identity))
    if free:
        return ffi.gc(m, free)
    return m


def monoid_new_int64(op, identity, *, free=monoid_free):
    """Create a new GrB_Monoid with an int64 identity.

    >>> m = monoid_new_int64(lib.GrB_PLUS_INT64, 0)
    >>> monoid_operator(m) == lib.GrB_PLUS_INT64
    True

    """
    m = ffi.new("GrB_Monoid*")
    check_status(m, lib.GrB_Monoid_new_INT64(m, op, identity))
    if free:
        return ffi.gc(m, free)
    return m


def monoid_new_uint8(op, identity, *, free=monoid_free):
    """Create a new GrB_Monoid with a uint8 identity.

    >>> m = monoid_new_uint8(lib.GrB_PLUS_UINT8, 0)
    >>> monoid_operator(m) == lib.GrB_PLUS_UINT8
    True

    """
    m = ffi.new("GrB_Monoid*")
    check_status(m, lib.GrB_Monoid_new_UINT8(m, op, identity))
    if free:
        return ffi.gc(m, free)
    return m


def monoid_new_uint16(op, identity, *, free=monoid_free):
    """Create a new GrB_Monoid with a uint16 identity.

    >>> m = monoid_new_uint16(lib.GrB_PLUS_UINT16, 0)
    >>> monoid_operator(m) == lib.GrB_PLUS_UINT16
    True

    """
    m = ffi.new("GrB_Monoid*")
    check_status(m, lib.GrB_Monoid_new_UINT16(m, op, identity))
    if free:
        return ffi.gc(m, free)
    return m


def monoid_new_uint32(op, identity, *, free=monoid_free):
    """Create a new GrB_Monoid with a uint32 identity.

    >>> m = monoid_new_uint32(lib.GrB_PLUS_UINT32, 0)
    >>> monoid_operator(m) == lib.GrB_PLUS_UINT32
    True

    """
    m = ffi.new("GrB_Monoid*")
    check_status(m, lib.GrB_Monoid_new_UINT32(m, op, identity))
    if free:
        return ffi.gc(m, free)
    return m


def monoid_new_uint64(op, identity, *, free=monoid_free):
    """Create a new GrB_Monoid with a uint64 identity.

    >>> m = monoid_new_uint64(lib.GrB_PLUS_UINT64, 0)
    >>> monoid_operator(m) == lib.GrB_PLUS_UINT64
    True

    """
    m = ffi.new("GrB_Monoid*")
    check_status(m, lib.GrB_Monoid_new_UINT64(m, op, identity))
    if free:
        return ffi.gc(m, free)
    return m


def monoid_new_fp32(op, identity, *, free=monoid_free):
    """Create a new GrB_Monoid with an fp32 identity.

    >>> m = monoid_new_fp32(lib.GrB_PLUS_FP32, 0.0)
    >>> monoid_operator(m) == lib.GrB_PLUS_FP32
    True

    """
    m = ffi.new("GrB_Monoid*")
    check_status(m, lib.GrB_Monoid_new_FP32(m, op, identity))
    if free:
        return ffi.gc(m, free)
    return m


def monoid_new_fp64(op, identity, *, free=monoid_free):
    """Create a new GrB_Monoid with an fp64 identity.

    >>> m = monoid_new_fp64(lib.GrB_PLUS_FP64, 0.0)
    >>> monoid_operator(m) == lib.GrB_PLUS_FP64
    True

    """
    m = ffi.new("GrB_Monoid*")
    check_status(m, lib.GrB_Monoid_new_FP64(m, op, identity))
    if free:
        return ffi.gc(m, free)
    return m


def monoid_new_udt(op, identity, *, free=monoid_free):
    """Create a new GrB_Monoid with a user-defined type identity.

    The ``identity`` argument must be a ``void*`` pointer.

    """
    m = ffi.new("GrB_Monoid*")
    check_status(m, lib.GrB_Monoid_new_UDT(m, op, identity))
    if free:
        return ffi.gc(m, free)
    return m


# ---------------------------------------------------------------------------
# Terminal constructors
# ---------------------------------------------------------------------------


def monoid_terminal_new_bool(op, identity, terminal, *, free=monoid_free):
    """Create a new GxB_Monoid with a boolean identity and terminal.

    >>> m = monoid_terminal_new_bool(lib.GrB_LAND, True, False)
    >>> monoid_operator(m) == lib.GrB_LAND
    True

    """
    m = ffi.new("GrB_Monoid*")
    check_status(m, lib.GxB_Monoid_terminal_new_BOOL(m, op, identity, terminal))
    if free:
        return ffi.gc(m, free)
    return m


def monoid_terminal_new_int8(op, identity, terminal, *, free=monoid_free):
    """Create a new GxB_Monoid with an int8 identity and terminal.

    >>> m = monoid_terminal_new_int8(lib.GrB_PLUS_INT8, 0, 0)
    >>> monoid_operator(m) == lib.GrB_PLUS_INT8
    True

    """
    m = ffi.new("GrB_Monoid*")
    check_status(m, lib.GxB_Monoid_terminal_new_INT8(m, op, identity, terminal))
    if free:
        return ffi.gc(m, free)
    return m


def monoid_terminal_new_int16(op, identity, terminal, *, free=monoid_free):
    """Create a new GxB_Monoid with an int16 identity and terminal.

    >>> m = monoid_terminal_new_int16(lib.GrB_PLUS_INT16, 0, 0)
    >>> monoid_operator(m) == lib.GrB_PLUS_INT16
    True

    """
    m = ffi.new("GrB_Monoid*")
    check_status(m, lib.GxB_Monoid_terminal_new_INT16(m, op, identity, terminal))
    if free:
        return ffi.gc(m, free)
    return m


def monoid_terminal_new_int32(op, identity, terminal, *, free=monoid_free):
    """Create a new GxB_Monoid with an int32 identity and terminal.

    >>> m = monoid_terminal_new_int32(lib.GrB_PLUS_INT32, 0, 0)
    >>> monoid_operator(m) == lib.GrB_PLUS_INT32
    True

    """
    m = ffi.new("GrB_Monoid*")
    check_status(m, lib.GxB_Monoid_terminal_new_INT32(m, op, identity, terminal))
    if free:
        return ffi.gc(m, free)
    return m


def monoid_terminal_new_int64(op, identity, terminal, *, free=monoid_free):
    """Create a new GxB_Monoid with an int64 identity and terminal.

    >>> m = monoid_terminal_new_int64(lib.GrB_PLUS_INT64, 0, 0)
    >>> monoid_operator(m) == lib.GrB_PLUS_INT64
    True

    """
    m = ffi.new("GrB_Monoid*")
    check_status(m, lib.GxB_Monoid_terminal_new_INT64(m, op, identity, terminal))
    if free:
        return ffi.gc(m, free)
    return m


def monoid_terminal_new_uint8(op, identity, terminal, *, free=monoid_free):
    """Create a new GxB_Monoid with a uint8 identity and terminal.

    >>> m = monoid_terminal_new_uint8(lib.GrB_PLUS_UINT8, 0, 0)
    >>> monoid_operator(m) == lib.GrB_PLUS_UINT8
    True

    """
    m = ffi.new("GrB_Monoid*")
    check_status(m, lib.GxB_Monoid_terminal_new_UINT8(m, op, identity, terminal))
    if free:
        return ffi.gc(m, free)
    return m


def monoid_terminal_new_uint16(op, identity, terminal, *, free=monoid_free):
    """Create a new GxB_Monoid with a uint16 identity and terminal.

    >>> m = monoid_terminal_new_uint16(lib.GrB_PLUS_UINT16, 0, 0)
    >>> monoid_operator(m) == lib.GrB_PLUS_UINT16
    True

    """
    m = ffi.new("GrB_Monoid*")
    check_status(m, lib.GxB_Monoid_terminal_new_UINT16(m, op, identity, terminal))
    if free:
        return ffi.gc(m, free)
    return m


def monoid_terminal_new_uint32(op, identity, terminal, *, free=monoid_free):
    """Create a new GxB_Monoid with a uint32 identity and terminal.

    >>> m = monoid_terminal_new_uint32(lib.GrB_PLUS_UINT32, 0, 0)
    >>> monoid_operator(m) == lib.GrB_PLUS_UINT32
    True

    """
    m = ffi.new("GrB_Monoid*")
    check_status(m, lib.GxB_Monoid_terminal_new_UINT32(m, op, identity, terminal))
    if free:
        return ffi.gc(m, free)
    return m


def monoid_terminal_new_uint64(op, identity, terminal, *, free=monoid_free):
    """Create a new GxB_Monoid with a uint64 identity and terminal.

    >>> m = monoid_terminal_new_uint64(lib.GrB_PLUS_UINT64, 0, 0)
    >>> monoid_operator(m) == lib.GrB_PLUS_UINT64
    True

    """
    m = ffi.new("GrB_Monoid*")
    check_status(m, lib.GxB_Monoid_terminal_new_UINT64(m, op, identity, terminal))
    if free:
        return ffi.gc(m, free)
    return m


def monoid_terminal_new_fp32(op, identity, terminal, *, free=monoid_free):
    """Create a new GxB_Monoid with an fp32 identity and terminal.

    >>> m = monoid_terminal_new_fp32(lib.GrB_PLUS_FP32, 0.0, 0.0)
    >>> monoid_operator(m) == lib.GrB_PLUS_FP32
    True

    """
    m = ffi.new("GrB_Monoid*")
    check_status(m, lib.GxB_Monoid_terminal_new_FP32(m, op, identity, terminal))
    if free:
        return ffi.gc(m, free)
    return m


def monoid_terminal_new_fp64(op, identity, terminal, *, free=monoid_free):
    """Create a new GxB_Monoid with an fp64 identity and terminal.

    >>> m = monoid_terminal_new_fp64(lib.GrB_PLUS_FP64, 0.0, 0.0)
    >>> monoid_operator(m) == lib.GrB_PLUS_FP64
    True

    """
    m = ffi.new("GrB_Monoid*")
    check_status(m, lib.GxB_Monoid_terminal_new_FP64(m, op, identity, terminal))
    if free:
        return ffi.gc(m, free)
    return m


def monoid_terminal_new_udt(op, identity, terminal, *, free=monoid_free):
    """Create a new GxB_Monoid with a user-defined type identity and terminal.

    The ``identity`` and ``terminal`` arguments must be ``void*`` pointers.

    """
    m = ffi.new("GrB_Monoid*")
    check_status(m, lib.GxB_Monoid_terminal_new_UDT(m, op, identity, terminal))
    if free:
        return ffi.gc(m, free)
    return m


if supports_complex():

    def monoid_new_fc32(op, identity, *, free=monoid_free):
        """Create a new GrB_Monoid with an fc32 identity.

        >>> m = monoid_new_fc32(lib.GxB_PLUS_FC32, 0 + 0j)
        >>> monoid_operator(m) == lib.GxB_PLUS_FC32
        True

        """
        m = ffi.new("GrB_Monoid*")
        check_status(m, lib.GxB_Monoid_new_FC32(m, op, identity))
        if free:
            return ffi.gc(m, free)
        return m

    def monoid_new_fc64(op, identity, *, free=monoid_free):
        """Create a new GrB_Monoid with an fc64 identity.

        >>> m = monoid_new_fc64(lib.GxB_PLUS_FC64, 0 + 0j)
        >>> monoid_operator(m) == lib.GxB_PLUS_FC64
        True

        """
        m = ffi.new("GrB_Monoid*")
        check_status(m, lib.GxB_Monoid_new_FC64(m, op, identity))
        if free:
            return ffi.gc(m, free)
        return m

    def monoid_terminal_new_fc32(op, identity, terminal, *, free=monoid_free):
        """Create a new GxB_Monoid with an fc32 identity and terminal.

        >>> m = monoid_terminal_new_fc32(lib.GxB_PLUS_FC32, 0 + 0j, 0 + 0j)
        >>> monoid_operator(m) == lib.GxB_PLUS_FC32
        True

        """
        m = ffi.new("GrB_Monoid*")
        check_status(m, lib.GxB_Monoid_terminal_new_FC32(m, op, identity, terminal))
        if free:
            return ffi.gc(m, free)
        return m

    def monoid_terminal_new_fc64(op, identity, terminal, *, free=monoid_free):
        """Create a new GxB_Monoid with an fc64 identity and terminal.

        >>> m = monoid_terminal_new_fc64(lib.GxB_PLUS_FC64, 0 + 0j, 0 + 0j)
        >>> monoid_operator(m) == lib.GxB_PLUS_FC64
        True

        """
        m = ffi.new("GrB_Monoid*")
        check_status(m, lib.GxB_Monoid_terminal_new_FC64(m, op, identity, terminal))
        if free:
            return ffi.gc(m, free)
        return m


# ---------------------------------------------------------------------------
# Wait
# ---------------------------------------------------------------------------


def monoid_wait(m, waitmode=lib.GrB_COMPLETE):
    """Wait for a monoid to complete pending operations.

    >>> m = ffi.new("GrB_Monoid*", lib.GrB_PLUS_MONOID_INT64)
    >>> monoid_wait(m)

    """
    check_status(m, lib.GrB_Monoid_wait(m[0], waitmode))


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------


def monoid_print(m, name="", level=lib.GxB_COMPLETE):
    """Print a monoid to stdout.

    ``level`` controls verbosity: ``lib.GxB_SILENT``, ``lib.GxB_SUMMARY``,
    ``lib.GxB_SHORT``, ``lib.GxB_COMPLETE``, ``lib.GxB_SHORT_VERBOSE``,
    or ``lib.GxB_COMPLETE_VERBOSE``.

    >>> m = ffi.new("GrB_Monoid*", lib.GrB_PLUS_MONOID_INT64)
    >>> out = _capture_c_output(monoid_print, m, 'm', lib.GxB_SHORT)
    >>> 'Monoid' in out
    True

    """
    check_status(m, lib.GxB_Monoid_fprint(
        m[0], name.encode() if isinstance(name, str) else name,
        level, ffi.NULL,
    ))


def monoid_fprint(m, f, name="", level=lib.GxB_COMPLETE):
    """Print a monoid to a C FILE* stream.

    Pass ``ffi.NULL`` for ``f`` to print to stdout.
    ``level`` controls verbosity (see :func:`monoid_print`).

    >>> m = ffi.new("GrB_Monoid*", lib.GrB_PLUS_MONOID_INT64)
    >>> out = _capture_c_output(monoid_fprint, m, ffi.NULL, 'm', lib.GxB_SHORT)
    >>> 'Monoid' in out
    True

    """
    check_status(m, lib.GxB_Monoid_fprint(
        m[0], name.encode() if isinstance(name, str) else name,
        level, f,
    ))


# ---------------------------------------------------------------------------
# Option get/set (typed)
# ---------------------------------------------------------------------------


def monoid_get_int32(m, field):
    """Get a monoid option as an int32.

    >>> m = ffi.new("GrB_Monoid*", lib.GrB_PLUS_MONOID_INT64)
    >>> monoid_get_int32(m, lib.GrB_INP0_TYPE_CODE) >= 0
    True

    """
    val = ffi.new("int32_t*")
    check_status(m, lib.GrB_Monoid_get_INT32(m[0], val, field))
    return val[0]


def monoid_set_int32(m, field, value):
    """Set a monoid option from an int32.

    """
    check_status(m, lib.GrB_Monoid_set_INT32(m[0], ffi.cast("int32_t", value), field))


def monoid_get_size(m, field):
    """Get a monoid option as a size_t.

    >>> m = ffi.new("GrB_Monoid*", lib.GrB_PLUS_MONOID_INT64)
    >>> monoid_get_size(m, lib.GrB_NAME) >= 0
    True

    """
    val = ffi.new("size_t*")
    check_status(m, lib.GrB_Monoid_get_SIZE(m[0], val, field))
    return val[0]


def monoid_get_string(m, field):
    """Get a monoid option as a string.

    >>> m = ffi.new("GrB_Monoid*", lib.GrB_PLUS_MONOID_INT64)
    >>> isinstance(monoid_get_string(m, lib.GrB_NAME), str)
    True

    """
    val = ffi.new("char[256]")
    check_status(m, lib.GrB_Monoid_get_String(m[0], val, field))
    return ffi.string(val).decode()


def monoid_set_string(m, field, value):
    """Set a monoid option from a string.

    """
    check_status(m, lib.GrB_Monoid_set_String(m[0], value.encode(), field))


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def monoid_operator(m):
    """Return the binary operator of a monoid.

    >>> m = ffi.new("GrB_Monoid*", lib.GrB_PLUS_MONOID_INT64)
    >>> monoid_operator(m) == lib.GrB_PLUS_INT64
    True

    """
    op = ffi.new("GrB_BinaryOp*")
    check_status(m, lib.GxB_Monoid_operator(op, m[0]))
    return op[0]


def monoid_identity(m):
    """Return a buffer containing the identity value of a monoid.

    >>> m = ffi.new("GrB_Monoid*", lib.GrB_PLUS_MONOID_INT64)
    >>> val = monoid_identity(m)

    """
    val = ffi.new("char[256]")
    check_status(m, lib.GxB_Monoid_identity(val, m[0]))
    return val


def monoid_terminal(m):
    """Return a (has_terminal, buffer) tuple for a monoid.

    ``has_terminal`` is a boolean indicating whether the monoid has a
    terminal value.

    >>> m = ffi.new("GrB_Monoid*", lib.GrB_PLUS_MONOID_INT64)
    >>> has_terminal, val = monoid_terminal(m)
    >>> isinstance(has_terminal, bool)
    True

    """
    has_terminal = ffi.new("bool*")
    val = ffi.new("char[256]")
    check_status(m, lib.GxB_Monoid_terminal(has_terminal, val, m[0]))
    return has_terminal[0], val
