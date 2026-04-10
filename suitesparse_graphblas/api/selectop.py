from suitesparse_graphblas import _error_code_lookup, ffi, lib

from .utils import _capture_c_output  # noqa: F401


def selectop_xtype(op):
    """Return the xtype of a select operator.

    >>> op = ffi.new("GxB_SelectOp*", lib.GxB_TRIL)
    >>> selectop_xtype(op) == ffi.NULL
    True

    """
    T = ffi.new("GrB_Type*")
    info = lib.GxB_SelectOp_xtype(T, op[0])
    if info != lib.GrB_SUCCESS:
        raise _error_code_lookup.get(info, RuntimeError)(
            f"GxB_SelectOp_xtype failed with info={info}"
        )
    return T[0]


def selectop_ttype(op):
    """Return the ttype (thunk type) of a select operator.

    >>> op = ffi.new("GxB_SelectOp*", lib.GxB_TRIL)
    >>> selectop_ttype(op) == ffi.NULL
    True

    """
    T = ffi.new("GrB_Type*")
    info = lib.GxB_SelectOp_ttype(T, op[0])
    if info != lib.GrB_SUCCESS:
        raise _error_code_lookup.get(info, RuntimeError)(
            f"GxB_SelectOp_ttype failed with info={info}"
        )
    return T[0]


def selectop_print(op, name="", level=lib.GxB_COMPLETE):
    """Print a select operator to stdout.

    ``level`` controls verbosity: ``lib.GxB_SILENT``, ``lib.GxB_SUMMARY``,
    ``lib.GxB_SHORT``, ``lib.GxB_COMPLETE``, ``lib.GxB_SHORT_VERBOSE``,
    or ``lib.GxB_COMPLETE_VERBOSE``.

    >>> op = ffi.new("GxB_SelectOp*", lib.GxB_TRIL)
    >>> out = _capture_c_output(selectop_print, op, 'op', lib.GxB_SHORT)
    >>> 'tril' in out.lower()
    True

    """
    info = lib.GxB_SelectOp_fprint(
        op[0], name.encode() if isinstance(name, str) else name,
        level, ffi.NULL,
    )
    if info != lib.GrB_SUCCESS:
        raise _error_code_lookup.get(info, RuntimeError)(
            f"GxB_SelectOp_fprint failed with info={info}"
        )


def selectop_fprint(op, f, name="", level=lib.GxB_COMPLETE):
    """Print a select operator to a C FILE* stream.

    Pass ``ffi.NULL`` for ``f`` to print to stdout.
    ``level`` controls verbosity (see :func:`selectop_print`).

    >>> op = ffi.new("GxB_SelectOp*", lib.GxB_TRIL)
    >>> out = _capture_c_output(selectop_fprint, op, ffi.NULL, 'op', lib.GxB_SHORT)
    >>> 'tril' in out.lower()
    True

    """
    info = lib.GxB_SelectOp_fprint(
        op[0], name.encode() if isinstance(name, str) else name,
        level, f,
    )
    if info != lib.GrB_SUCCESS:
        raise _error_code_lookup.get(info, RuntimeError)(
            f"GxB_SelectOp_fprint failed with info={info}"
        )
