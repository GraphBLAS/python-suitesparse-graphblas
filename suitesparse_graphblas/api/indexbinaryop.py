"""GxB_IndexBinaryOp functional API.

There are no built-in GxB_IndexBinaryOp constants in the library, so most
functions in this module cannot have doctests without constructing an operator
from a CFFI callback.  Only ``indexbinaryop_new`` creates instances, and it
requires a C function pointer.
"""

from suitesparse_graphblas import check_status, ffi, lib

from .utils import _capture_c_output  # noqa: F401


def indexbinaryop_free(op):
    """Free an index binary operator."""
    check_status(op, lib.GxB_IndexBinaryOp_free(op))


def indexbinaryop_new(
    function, ztype, xtype, ytype, theta_type, name, defn, *, free=indexbinaryop_free
):
    """Create a new GxB_IndexBinaryOp from a C function pointer.

    ``function`` must be a CFFI callback compatible with
    ``GxB_index_binary_function``.  ``name`` and ``defn`` are C strings
    describing the operator for JIT compilation.

    The ``free`` argument is called when the object is garbage
    collected, the default is ``indexbinaryop.indexbinaryop_free()``.
    If ``free`` is None then there is no automatic garbage collection
    and it is up to the user to free the operator.
    """
    op = ffi.new("GxB_IndexBinaryOp*")
    check_status(
        op,
        lib.GxB_IndexBinaryOp_new(
            op,
            function,
            ztype,
            xtype,
            ytype,
            theta_type,
            name.encode() if isinstance(name, str) else name,
            defn.encode() if isinstance(defn, str) else defn,
        ),
    )
    if free:
        return ffi.gc(op, free)
    return op


def indexbinaryop_wait(op, waitmode=lib.GrB_COMPLETE):
    """Wait for an index binary operator to complete pending operations."""
    check_status(op, lib.GxB_IndexBinaryOp_wait(op[0], waitmode))


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------


def indexbinaryop_print(op, name="", level=lib.GxB_COMPLETE):
    """Print an index binary operator to stdout.

    ``level`` controls verbosity: ``lib.GxB_SILENT``, ``lib.GxB_SUMMARY``,
    ``lib.GxB_SHORT``, ``lib.GxB_COMPLETE``, ``lib.GxB_SHORT_VERBOSE``,
    or ``lib.GxB_COMPLETE_VERBOSE``.
    """
    check_status(op, lib.GxB_IndexBinaryOp_fprint(
        op[0], name.encode() if isinstance(name, str) else name,
        level, ffi.NULL,
    ))


def indexbinaryop_fprint(op, f, name="", level=lib.GxB_COMPLETE):
    """Print an index binary operator to a C FILE* stream.

    Pass ``ffi.NULL`` for ``f`` to print to stdout.
    ``level`` controls verbosity (see :func:`indexbinaryop_print`).
    """
    check_status(op, lib.GxB_IndexBinaryOp_fprint(
        op[0], name.encode() if isinstance(name, str) else name,
        level, f,
    ))


# ---------------------------------------------------------------------------
# Option get/set (typed)
# ---------------------------------------------------------------------------


def indexbinaryop_get_int32(op, field):
    """Get an operator option as an int32."""
    val = ffi.new("int32_t*")
    check_status(op, lib.GxB_IndexBinaryOp_get_INT32(op[0], val, field))
    return val[0]


def indexbinaryop_set_int32(op, field, value):
    """Set an operator option from an int32."""
    check_status(
        op, lib.GxB_IndexBinaryOp_set_INT32(op[0], ffi.cast("int32_t", value), field)
    )


def indexbinaryop_get_size(op, field):
    """Get an operator option as a size_t."""
    val = ffi.new("size_t*")
    check_status(op, lib.GxB_IndexBinaryOp_get_SIZE(op[0], val, field))
    return val[0]


def indexbinaryop_get_string(op, field):
    """Get an operator option as a string."""
    val = ffi.new("char[256]")
    check_status(op, lib.GxB_IndexBinaryOp_get_String(op[0], val, field))
    return ffi.string(val).decode()


def indexbinaryop_set_string(op, field, value):
    """Set an operator option from a string."""
    check_status(
        op,
        lib.GxB_IndexBinaryOp_set_String(
            op[0], value.encode() if isinstance(value, str) else value, field
        ),
    )
