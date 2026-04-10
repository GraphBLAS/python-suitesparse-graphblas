from suitesparse_graphblas import check_status, ffi, lib

from .utils import _capture_c_output  # noqa: F401


def semiring_free(s):
    """Free a semiring.

    >>> s = semiring_new(lib.GrB_PLUS_MONOID_INT64, lib.GrB_TIMES_INT64)
    >>> semiring_free(s)

    """
    check_status(s, lib.GrB_Semiring_free(s))


def semiring_new(add, multiply, *, free=semiring_free):
    """Create a new GrB_Semiring from a monoid and binary operator.

    ``add`` is a ``GrB_Monoid`` value and ``multiply`` is a
    ``GrB_BinaryOp`` value.

    The ``free`` argument is called when the object is garbage
    collected, the default is ``semiring.semiring_free()``.  If ``free``
    is None then there is no automatic garbage collection and it is up
    to the user to free the semiring.

    >>> s = semiring_new(lib.GrB_PLUS_MONOID_INT64, lib.GrB_TIMES_INT64)
    >>> semiring_multiply(s) == lib.GrB_TIMES_INT64
    True

    """
    s = ffi.new("GrB_Semiring*")
    check_status(s, lib.GrB_Semiring_new(s, add, multiply))
    if free:
        return ffi.gc(s, free)
    return s


# ---------------------------------------------------------------------------
# Wait
# ---------------------------------------------------------------------------


def semiring_wait(s, waitmode=lib.GrB_COMPLETE):
    """Wait for a semiring to complete pending operations.

    >>> s = ffi.new("GrB_Semiring*", lib.GrB_PLUS_TIMES_SEMIRING_INT64)
    >>> semiring_wait(s)

    """
    check_status(s, lib.GrB_Semiring_wait(s[0], waitmode))


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------


def semiring_print(s, name="", level=lib.GxB_COMPLETE):
    """Print a semiring to stdout.

    ``level`` controls verbosity: ``lib.GxB_SILENT``, ``lib.GxB_SUMMARY``,
    ``lib.GxB_SHORT``, ``lib.GxB_COMPLETE``, ``lib.GxB_SHORT_VERBOSE``,
    or ``lib.GxB_COMPLETE_VERBOSE``.

    >>> s = ffi.new("GrB_Semiring*", lib.GrB_PLUS_TIMES_SEMIRING_INT64)
    >>> out = _capture_c_output(semiring_print, s, 's', lib.GxB_SHORT)
    >>> 'Semiring' in out
    True

    """
    check_status(s, lib.GxB_Semiring_fprint(
        s[0], name.encode() if isinstance(name, str) else name,
        level, ffi.NULL,
    ))


def semiring_fprint(s, f, name="", level=lib.GxB_COMPLETE):
    """Print a semiring to a C FILE* stream.

    Pass ``ffi.NULL`` for ``f`` to print to stdout.
    ``level`` controls verbosity (see :func:`semiring_print`).

    >>> s = ffi.new("GrB_Semiring*", lib.GrB_PLUS_TIMES_SEMIRING_INT64)
    >>> out = _capture_c_output(semiring_fprint, s, ffi.NULL, 's', lib.GxB_SHORT)
    >>> 'Semiring' in out
    True

    """
    check_status(s, lib.GxB_Semiring_fprint(
        s[0], name.encode() if isinstance(name, str) else name,
        level, f,
    ))


# ---------------------------------------------------------------------------
# Option get/set (typed)
# ---------------------------------------------------------------------------


def semiring_get_int32(s, field):
    """Get a semiring option as an int32.

    >>> s = ffi.new("GrB_Semiring*", lib.GrB_PLUS_TIMES_SEMIRING_INT64)
    >>> semiring_get_int32(s, lib.GrB_INP0_TYPE_CODE) >= 0
    True

    """
    val = ffi.new("int32_t*")
    check_status(s, lib.GrB_Semiring_get_INT32(s[0], val, field))
    return val[0]


def semiring_set_int32(s, field, value):
    """Set a semiring option from an int32.

    """
    check_status(s, lib.GrB_Semiring_set_INT32(s[0], ffi.cast("int32_t", value), field))


def semiring_get_size(s, field):
    """Get a semiring option as a size_t.

    >>> s = ffi.new("GrB_Semiring*", lib.GrB_PLUS_TIMES_SEMIRING_INT64)
    >>> semiring_get_size(s, lib.GrB_NAME) >= 0
    True

    """
    val = ffi.new("size_t*")
    check_status(s, lib.GrB_Semiring_get_SIZE(s[0], val, field))
    return val[0]


def semiring_get_string(s, field):
    """Get a semiring option as a string.

    >>> s = ffi.new("GrB_Semiring*", lib.GrB_PLUS_TIMES_SEMIRING_INT64)
    >>> isinstance(semiring_get_string(s, lib.GrB_NAME), str)
    True

    """
    val = ffi.new("char[256]")
    check_status(s, lib.GrB_Semiring_get_String(s[0], val, field))
    return ffi.string(val).decode()


def semiring_set_string(s, field, value):
    """Set a semiring option from a string.

    """
    check_status(s, lib.GrB_Semiring_set_String(s[0], value.encode(), field))


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def semiring_add(s):
    """Return the additive monoid of a semiring.

    >>> s = ffi.new("GrB_Semiring*", lib.GrB_PLUS_TIMES_SEMIRING_INT64)
    >>> semiring_add(s) == lib.GrB_PLUS_MONOID_INT64
    True

    """
    m = ffi.new("GrB_Monoid*")
    check_status(s, lib.GxB_Semiring_add(m, s[0]))
    return m[0]


def semiring_multiply(s):
    """Return the multiply binary operator of a semiring.

    >>> s = ffi.new("GrB_Semiring*", lib.GrB_PLUS_TIMES_SEMIRING_INT64)
    >>> semiring_multiply(s) == lib.GrB_TIMES_INT64
    True

    """
    op = ffi.new("GrB_BinaryOp*")
    check_status(s, lib.GxB_Semiring_multiply(op, s[0]))
    return op[0]
