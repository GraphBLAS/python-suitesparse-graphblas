from suitesparse_graphblas import check_status, ffi, lib

from .utils import _capture_c_output  # noqa: F401


def binaryop_free(op):
    """Free a binary operator.

    >>> op = ffi.new("GrB_BinaryOp*", lib.GrB_PLUS_INT64)
    >>> binaryop_free(op)

    """
    check_status(op, lib.GrB_BinaryOp_free(op))


def binaryop_new(function, ztype, xtype, ytype, *, free=binaryop_free):
    """Create a new GrB_BinaryOp from a C function pointer.

    ``function`` must be a CFFI callback compatible with
    ``GxB_binary_function``.

    The ``free`` argument is called when the object is garbage
    collected, the default is ``binaryop.binaryop_free()``.  If ``free``
    is None then there is no automatic garbage collection and it is up
    to the user to free the operator.
    """
    op = ffi.new("GrB_BinaryOp*")
    check_status(op, lib.GrB_BinaryOp_new(op, function, ztype, xtype, ytype))
    if free:
        return ffi.gc(op, free)
    return op


def binaryop_new_named(function, ztype, xtype, ytype, name, defn, *, free=binaryop_free):
    """Create a new named GrB_BinaryOp (GxB extension).

    ``function`` must be a CFFI callback compatible with
    ``GxB_binary_function``.  ``name`` and ``defn`` are C strings
    describing the operator for JIT compilation.

    The ``free`` argument is called when the object is garbage
    collected, the default is ``binaryop.binaryop_free()``.  If ``free``
    is None then there is no automatic garbage collection and it is up
    to the user to free the operator.
    """
    op = ffi.new("GrB_BinaryOp*")
    check_status(
        op,
        lib.GxB_BinaryOp_new(
            op,
            function,
            ztype,
            xtype,
            ytype,
            name.encode() if isinstance(name, str) else name,
            defn.encode() if isinstance(defn, str) else defn,
        ),
    )
    if free:
        return ffi.gc(op, free)
    return op


def binaryop_new_indexop(idxbinop, theta, *, free=binaryop_free):
    """Create a new GrB_BinaryOp from a GxB_IndexBinaryOp and a scalar theta.

    The ``free`` argument is called when the object is garbage
    collected, the default is ``binaryop.binaryop_free()``.  If ``free``
    is None then there is no automatic garbage collection and it is up
    to the user to free the operator.
    """
    op = ffi.new("GrB_BinaryOp*")
    check_status(op, lib.GxB_BinaryOp_new_IndexOp(op, idxbinop, theta))
    if free:
        return ffi.gc(op, free)
    return op


def binaryop_wait(op, waitmode=lib.GrB_COMPLETE):
    """Wait for a binary operator to complete pending operations.

    >>> op = ffi.new("GrB_BinaryOp*", lib.GrB_PLUS_INT64)
    >>> binaryop_wait(op)

    """
    check_status(op, lib.GrB_BinaryOp_wait(op[0], waitmode))


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------


def binaryop_print(op, name="", level=lib.GxB_COMPLETE):
    """Print a binary operator to stdout.

    ``level`` controls verbosity: ``lib.GxB_SILENT``, ``lib.GxB_SUMMARY``,
    ``lib.GxB_SHORT``, ``lib.GxB_COMPLETE``, ``lib.GxB_SHORT_VERBOSE``,
    or ``lib.GxB_COMPLETE_VERBOSE``.

    >>> op = ffi.new("GrB_BinaryOp*", lib.GrB_PLUS_INT64)
    >>> out = _capture_c_output(binaryop_print, op, 'op', lib.GxB_SHORT)
    >>> 'plus' in out.lower()
    True

    """
    check_status(op, lib.GxB_BinaryOp_fprint(
        op[0], name.encode() if isinstance(name, str) else name,
        level, ffi.NULL,
    ))


def binaryop_fprint(op, f, name="", level=lib.GxB_COMPLETE):
    """Print a binary operator to a C FILE* stream.

    Pass ``ffi.NULL`` for ``f`` to print to stdout.
    ``level`` controls verbosity (see :func:`binaryop_print`).

    >>> op = ffi.new("GrB_BinaryOp*", lib.GrB_PLUS_INT64)
    >>> out = _capture_c_output(binaryop_fprint, op, ffi.NULL, 'op', lib.GxB_SHORT)
    >>> 'plus' in out.lower()
    True

    """
    check_status(op, lib.GxB_BinaryOp_fprint(
        op[0], name.encode() if isinstance(name, str) else name,
        level, f,
    ))


# ---------------------------------------------------------------------------
# Option get/set (typed)
# ---------------------------------------------------------------------------


def binaryop_get_int32(op, field):
    """Get an operator option as an int32.

    >>> op = ffi.new("GrB_BinaryOp*", lib.GrB_PLUS_INT64)
    >>> binaryop_get_int32(op, lib.GrB_OUTP_TYPE_CODE) >= 0
    True

    """
    val = ffi.new("int32_t*")
    check_status(op, lib.GrB_BinaryOp_get_INT32(op[0], val, field))
    return val[0]


def binaryop_set_int32(op, field, value):
    """Set an operator option from an int32."""
    check_status(op, lib.GrB_BinaryOp_set_INT32(op[0], ffi.cast("int32_t", value), field))


def binaryop_get_size(op, field):
    """Get an operator option as a size_t.

    >>> op = ffi.new("GrB_BinaryOp*", lib.GrB_PLUS_INT64)
    >>> binaryop_get_size(op, lib.GrB_NAME) > 0
    True

    """
    val = ffi.new("size_t*")
    check_status(op, lib.GrB_BinaryOp_get_SIZE(op[0], val, field))
    return val[0]


def binaryop_get_string(op, field):
    """Get an operator option as a string.

    >>> op = ffi.new("GrB_BinaryOp*", lib.GrB_PLUS_INT64)
    >>> 'plus' in binaryop_get_string(op, lib.GrB_NAME).lower()
    True

    """
    val = ffi.new("char[256]")
    check_status(op, lib.GrB_BinaryOp_get_String(op[0], val, field))
    return ffi.string(val).decode()


def binaryop_set_string(op, field, value):
    """Set an operator option from a string."""
    check_status(
        op,
        lib.GrB_BinaryOp_set_String(
            op[0], value.encode() if isinstance(value, str) else value, field
        ),
    )


# ---------------------------------------------------------------------------
# Type queries
# ---------------------------------------------------------------------------


def binaryop_xtype(op):
    """Return the first input (x) type of a binary operator.

    >>> op = ffi.new("GrB_BinaryOp*", lib.GrB_PLUS_INT64)
    >>> binaryop_xtype(op) == lib.GrB_INT64
    True

    """
    T = ffi.new("GrB_Type*")
    check_status(op, lib.GxB_BinaryOp_xtype(T, op[0]))
    return T[0]


def binaryop_ytype(op):
    """Return the second input (y) type of a binary operator.

    >>> op = ffi.new("GrB_BinaryOp*", lib.GrB_PLUS_INT64)
    >>> binaryop_ytype(op) == lib.GrB_INT64
    True

    """
    T = ffi.new("GrB_Type*")
    check_status(op, lib.GxB_BinaryOp_ytype(T, op[0]))
    return T[0]


def binaryop_ztype(op):
    """Return the output (z) type of a binary operator.

    >>> op = ffi.new("GrB_BinaryOp*", lib.GrB_PLUS_INT64)
    >>> binaryop_ztype(op) == lib.GrB_INT64
    True

    """
    T = ffi.new("GrB_Type*")
    check_status(op, lib.GxB_BinaryOp_ztype(T, op[0]))
    return T[0]


def binaryop_xtype_name(op):
    """Return the first input (x) type name of a binary operator as a string.

    >>> op = ffi.new("GrB_BinaryOp*", lib.GrB_PLUS_INT64)
    >>> binaryop_xtype_name(op)
    'int64_t'

    """
    name = ffi.new("char[256]")
    check_status(op, lib.GxB_BinaryOp_xtype_name(name, op[0]))
    return ffi.string(name).decode()


def binaryop_ytype_name(op):
    """Return the second input (y) type name of a binary operator as a string.

    >>> op = ffi.new("GrB_BinaryOp*", lib.GrB_PLUS_INT64)
    >>> binaryop_ytype_name(op)
    'int64_t'

    """
    name = ffi.new("char[256]")
    check_status(op, lib.GxB_BinaryOp_ytype_name(name, op[0]))
    return ffi.string(name).decode()


def binaryop_ztype_name(op):
    """Return the output (z) type name of a binary operator as a string.

    >>> op = ffi.new("GrB_BinaryOp*", lib.GrB_PLUS_INT64)
    >>> binaryop_ztype_name(op)
    'int64_t'

    """
    name = ffi.new("char[256]")
    check_status(op, lib.GxB_BinaryOp_ztype_name(name, op[0]))
    return ffi.string(name).decode()
