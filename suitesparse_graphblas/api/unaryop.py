from suitesparse_graphblas import check_status, ffi, lib

from .utils import _capture_c_output  # noqa: F401


def unaryop_free(op):
    """Free a unary operator.

    >>> op = ffi.new("GrB_UnaryOp*", lib.GrB_AINV_INT64)
    >>> unaryop_free(op)

    """
    check_status(op, lib.GrB_UnaryOp_free(op))


def unaryop_new(function, ztype, xtype, *, free=unaryop_free):
    """Create a new GrB_UnaryOp from a C function pointer.

    ``function`` must be a CFFI callback compatible with
    ``GxB_unary_function``.

    The ``free`` argument is called when the object is garbage
    collected, the default is ``unaryop.unaryop_free()``.  If ``free``
    is None then there is no automatic garbage collection and it is up
    to the user to free the operator.
    """
    op = ffi.new("GrB_UnaryOp*")
    check_status(op, lib.GrB_UnaryOp_new(op, function, ztype, xtype))
    if free:
        return ffi.gc(op, free)
    return op


def unaryop_new_named(function, ztype, xtype, name, defn, *, free=unaryop_free):
    """Create a new named GrB_UnaryOp (GxB extension).

    ``function`` must be a CFFI callback compatible with
    ``GxB_unary_function``.  ``name`` and ``defn`` are C strings
    describing the operator for JIT compilation.

    The ``free`` argument is called when the object is garbage
    collected, the default is ``unaryop.unaryop_free()``.  If ``free``
    is None then there is no automatic garbage collection and it is up
    to the user to free the operator.
    """
    op = ffi.new("GrB_UnaryOp*")
    check_status(
        op,
        lib.GxB_UnaryOp_new(
            op,
            function,
            ztype,
            xtype,
            name.encode() if isinstance(name, str) else name,
            defn.encode() if isinstance(defn, str) else defn,
        ),
    )
    if free:
        return ffi.gc(op, free)
    return op


def unaryop_wait(op, waitmode=lib.GrB_COMPLETE):
    """Wait for a unary operator to complete pending operations.

    >>> op = ffi.new("GrB_UnaryOp*", lib.GrB_AINV_INT64)
    >>> unaryop_wait(op)

    """
    check_status(op, lib.GrB_UnaryOp_wait(op[0], waitmode))


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------


def unaryop_print(op, name="", level=lib.GxB_COMPLETE):
    """Print a unary operator to stdout.

    ``level`` controls verbosity: ``lib.GxB_SILENT``, ``lib.GxB_SUMMARY``,
    ``lib.GxB_SHORT``, ``lib.GxB_COMPLETE``, ``lib.GxB_SHORT_VERBOSE``,
    or ``lib.GxB_COMPLETE_VERBOSE``.

    >>> op = ffi.new("GrB_UnaryOp*", lib.GrB_AINV_INT64)
    >>> out = _capture_c_output(unaryop_print, op, 'op', lib.GxB_SHORT)
    >>> 'ainv' in out.lower()
    True

    """
    check_status(op, lib.GxB_UnaryOp_fprint(
        op[0], name.encode() if isinstance(name, str) else name,
        level, ffi.NULL,
    ))


def unaryop_fprint(op, f, name="", level=lib.GxB_COMPLETE):
    """Print a unary operator to a C FILE* stream.

    Pass ``ffi.NULL`` for ``f`` to print to stdout.
    ``level`` controls verbosity (see :func:`unaryop_print`).

    >>> op = ffi.new("GrB_UnaryOp*", lib.GrB_AINV_INT64)
    >>> out = _capture_c_output(unaryop_fprint, op, ffi.NULL, 'op', lib.GxB_SHORT)
    >>> 'ainv' in out.lower()
    True

    """
    check_status(op, lib.GxB_UnaryOp_fprint(
        op[0], name.encode() if isinstance(name, str) else name,
        level, f,
    ))


# ---------------------------------------------------------------------------
# Option get/set (typed)
# ---------------------------------------------------------------------------


def unaryop_get_int32(op, field):
    """Get an operator option as an int32.

    >>> op = ffi.new("GrB_UnaryOp*", lib.GrB_AINV_INT64)
    >>> unaryop_get_int32(op, lib.GrB_OUTP_TYPE_CODE) >= 0
    True

    """
    val = ffi.new("int32_t*")
    check_status(op, lib.GrB_UnaryOp_get_INT32(op[0], val, field))
    return val[0]


def unaryop_set_int32(op, field, value):
    """Set an operator option from an int32."""
    check_status(op, lib.GrB_UnaryOp_set_INT32(op[0], ffi.cast("int32_t", value), field))


def unaryop_get_size(op, field):
    """Get an operator option as a size_t.

    >>> op = ffi.new("GrB_UnaryOp*", lib.GrB_AINV_INT64)
    >>> unaryop_get_size(op, lib.GrB_NAME) > 0
    True

    """
    val = ffi.new("size_t*")
    check_status(op, lib.GrB_UnaryOp_get_SIZE(op[0], val, field))
    return val[0]


def unaryop_get_string(op, field):
    """Get an operator option as a string.

    >>> op = ffi.new("GrB_UnaryOp*", lib.GrB_AINV_INT64)
    >>> 'ainv' in unaryop_get_string(op, lib.GrB_NAME).lower()
    True

    """
    val = ffi.new("char[256]")
    check_status(op, lib.GrB_UnaryOp_get_String(op[0], val, field))
    return ffi.string(val).decode()


def unaryop_set_string(op, field, value):
    """Set an operator option from a string."""
    check_status(
        op,
        lib.GrB_UnaryOp_set_String(
            op[0], value.encode() if isinstance(value, str) else value, field
        ),
    )


# ---------------------------------------------------------------------------
# Type queries
# ---------------------------------------------------------------------------


def unaryop_xtype(op):
    """Return the input (x) type of a unary operator.

    >>> op = ffi.new("GrB_UnaryOp*", lib.GrB_AINV_INT64)
    >>> unaryop_xtype(op) == lib.GrB_INT64
    True

    """
    T = ffi.new("GrB_Type*")
    check_status(op, lib.GxB_UnaryOp_xtype(T, op[0]))
    return T[0]


def unaryop_ztype(op):
    """Return the output (z) type of a unary operator.

    >>> op = ffi.new("GrB_UnaryOp*", lib.GrB_AINV_INT64)
    >>> unaryop_ztype(op) == lib.GrB_INT64
    True

    """
    T = ffi.new("GrB_Type*")
    check_status(op, lib.GxB_UnaryOp_ztype(T, op[0]))
    return T[0]


def unaryop_xtype_name(op):
    """Return the input (x) type name of a unary operator as a string.

    >>> op = ffi.new("GrB_UnaryOp*", lib.GrB_AINV_INT64)
    >>> unaryop_xtype_name(op)
    'int64_t'

    """
    name = ffi.new("char[256]")
    check_status(op, lib.GxB_UnaryOp_xtype_name(name, op[0]))
    return ffi.string(name).decode()


def unaryop_ztype_name(op):
    """Return the output (z) type name of a unary operator as a string.

    >>> op = ffi.new("GrB_UnaryOp*", lib.GrB_AINV_INT64)
    >>> unaryop_ztype_name(op)
    'int64_t'

    """
    name = ffi.new("char[256]")
    check_status(op, lib.GxB_UnaryOp_ztype_name(name, op[0]))
    return ffi.string(name).decode()
