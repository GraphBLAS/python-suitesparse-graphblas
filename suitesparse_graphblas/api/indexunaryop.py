from suitesparse_graphblas import check_status, ffi, lib

from .utils import _capture_c_output  # noqa: F401


def indexunaryop_free(op):
    """Free an index unary operator.

    >>> op = ffi.new("GrB_IndexUnaryOp*", lib.GrB_ROWINDEX_INT64)
    >>> indexunaryop_free(op)

    """
    check_status(op, lib.GrB_IndexUnaryOp_free(op))


def indexunaryop_new(function, ztype, xtype, ytype, *, free=indexunaryop_free):
    """Create a new GrB_IndexUnaryOp from a C function pointer.

    ``function`` must be a CFFI callback compatible with
    ``GxB_index_unary_function``.

    The ``free`` argument is called when the object is garbage
    collected, the default is ``indexunaryop.indexunaryop_free()``.  If
    ``free`` is None then there is no automatic garbage collection and
    it is up to the user to free the operator.
    """
    op = ffi.new("GrB_IndexUnaryOp*")
    check_status(op, lib.GrB_IndexUnaryOp_new(op, function, ztype, xtype, ytype))
    if free:
        return ffi.gc(op, free)
    return op


def indexunaryop_new_named(function, ztype, xtype, ytype, name, defn, *, free=indexunaryop_free):
    """Create a new named GrB_IndexUnaryOp (GxB extension).

    ``function`` must be a CFFI callback compatible with
    ``GxB_index_unary_function``.  ``name`` and ``defn`` are C strings
    describing the operator for JIT compilation.

    The ``free`` argument is called when the object is garbage
    collected, the default is ``indexunaryop.indexunaryop_free()``.  If
    ``free`` is None then there is no automatic garbage collection and
    it is up to the user to free the operator.
    """
    op = ffi.new("GrB_IndexUnaryOp*")
    check_status(
        op,
        lib.GxB_IndexUnaryOp_new(
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


def indexunaryop_wait(op, waitmode=lib.GrB_COMPLETE):
    """Wait for an index unary operator to complete pending operations.

    >>> op = ffi.new("GrB_IndexUnaryOp*", lib.GrB_ROWINDEX_INT64)
    >>> indexunaryop_wait(op)

    """
    check_status(op, lib.GrB_IndexUnaryOp_wait(op[0], waitmode))


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------


def indexunaryop_print(op, name="", level=lib.GxB_COMPLETE):
    """Print an index unary operator to stdout.

    ``level`` controls verbosity: ``lib.GxB_SILENT``, ``lib.GxB_SUMMARY``,
    ``lib.GxB_SHORT``, ``lib.GxB_COMPLETE``, ``lib.GxB_SHORT_VERBOSE``,
    or ``lib.GxB_COMPLETE_VERBOSE``.

    >>> op = ffi.new("GrB_IndexUnaryOp*", lib.GrB_ROWINDEX_INT64)
    >>> out = _capture_c_output(indexunaryop_print, op, 'op', lib.GxB_SHORT)
    >>> 'rowindex' in out.lower()
    True

    """
    check_status(op, lib.GxB_IndexUnaryOp_fprint(
        op[0], name.encode() if isinstance(name, str) else name,
        level, ffi.NULL,
    ))


def indexunaryop_fprint(op, f, name="", level=lib.GxB_COMPLETE):
    """Print an index unary operator to a C FILE* stream.

    Pass ``ffi.NULL`` for ``f`` to print to stdout.
    ``level`` controls verbosity (see :func:`indexunaryop_print`).

    >>> op = ffi.new("GrB_IndexUnaryOp*", lib.GrB_ROWINDEX_INT64)
    >>> out = _capture_c_output(indexunaryop_fprint, op, ffi.NULL, 'op', lib.GxB_SHORT)
    >>> 'rowindex' in out.lower()
    True

    """
    check_status(op, lib.GxB_IndexUnaryOp_fprint(
        op[0], name.encode() if isinstance(name, str) else name,
        level, f,
    ))


# ---------------------------------------------------------------------------
# Option get/set (typed)
# ---------------------------------------------------------------------------


def indexunaryop_get_int32(op, field):
    """Get an operator option as an int32.

    >>> op = ffi.new("GrB_IndexUnaryOp*", lib.GrB_ROWINDEX_INT64)
    >>> indexunaryop_get_int32(op, lib.GrB_OUTP_TYPE_CODE) >= 0
    True

    """
    val = ffi.new("int32_t*")
    check_status(op, lib.GrB_IndexUnaryOp_get_INT32(op[0], val, field))
    return val[0]


def indexunaryop_set_int32(op, field, value):
    """Set an operator option from an int32."""
    check_status(
        op, lib.GrB_IndexUnaryOp_set_INT32(op[0], ffi.cast("int32_t", value), field)
    )


def indexunaryop_get_size(op, field):
    """Get an operator option as a size_t.

    >>> op = ffi.new("GrB_IndexUnaryOp*", lib.GrB_ROWINDEX_INT64)
    >>> indexunaryop_get_size(op, lib.GrB_NAME) > 0
    True

    """
    val = ffi.new("size_t*")
    check_status(op, lib.GrB_IndexUnaryOp_get_SIZE(op[0], val, field))
    return val[0]


def indexunaryop_get_string(op, field):
    """Get an operator option as a string.

    >>> op = ffi.new("GrB_IndexUnaryOp*", lib.GrB_ROWINDEX_INT64)
    >>> 'rowindex' in indexunaryop_get_string(op, lib.GrB_NAME).lower()
    True

    """
    val = ffi.new("char[256]")
    check_status(op, lib.GrB_IndexUnaryOp_get_String(op[0], val, field))
    return ffi.string(val).decode()


def indexunaryop_set_string(op, field, value):
    """Set an operator option from a string."""
    check_status(
        op,
        lib.GrB_IndexUnaryOp_set_String(
            op[0], value.encode() if isinstance(value, str) else value, field
        ),
    )


# ---------------------------------------------------------------------------
# Type name queries
# ---------------------------------------------------------------------------


def indexunaryop_xtype_name(op):
    """Return the input (x) type name of an index unary operator as a string.

    >>> op = ffi.new("GrB_IndexUnaryOp*", lib.GrB_ROWINDEX_INT64)
    >>> indexunaryop_xtype_name(op)
    ''

    """
    name = ffi.new("char[256]")
    check_status(op, lib.GxB_IndexUnaryOp_xtype_name(name, op[0]))
    return ffi.string(name).decode()


def indexunaryop_ytype_name(op):
    """Return the thunk (y) type name of an index unary operator as a string.

    >>> op = ffi.new("GrB_IndexUnaryOp*", lib.GrB_ROWINDEX_INT64)
    >>> indexunaryop_ytype_name(op)
    'int64_t'

    """
    name = ffi.new("char[256]")
    check_status(op, lib.GxB_IndexUnaryOp_ytype_name(name, op[0]))
    return ffi.string(name).decode()


def indexunaryop_ztype_name(op):
    """Return the output (z) type name of an index unary operator as a string.

    >>> op = ffi.new("GrB_IndexUnaryOp*", lib.GrB_ROWINDEX_INT64)
    >>> indexunaryop_ztype_name(op)
    'int64_t'

    """
    name = ffi.new("char[256]")
    check_status(op, lib.GxB_IndexUnaryOp_ztype_name(name, op[0]))
    return ffi.string(name).decode()
