from suitesparse_graphblas import check_status, ffi, lib

from .utils import _capture_c_output  # noqa: F401


def context_free(ctx):
    """Free a context.

    >>> ctx = context_new()
    >>> context_free(ctx)

    """
    check_status(ctx, lib.GxB_Context_free(ctx))


def context_new(*, free=context_free):
    """Create a new GxB_Context and initialize it.

    The ``free`` argument is called when the object is garbage
    collected, the default is ``context.context_free()``.  If ``free`` is None
    then there is no automatic garbage collection and it is up to the user
    to free the context.

    >>> ctx = context_new()

    """
    ctx = ffi.new("GxB_Context*")
    check_status(ctx, lib.GxB_Context_new(ctx))
    if free:
        return ffi.gc(ctx, free)
    return ctx


def context_wait(ctx, waitmode=lib.GrB_COMPLETE):
    """Wait for a context to complete pending operations.

    >>> ctx = context_new()
    >>> context_wait(ctx)

    """
    check_status(ctx, lib.GxB_Context_wait(ctx[0], waitmode))


# ---------------------------------------------------------------------------
# Engage / disengage
# ---------------------------------------------------------------------------


def context_engage(ctx):
    """Engage a context for the calling thread.

    An engaged context controls settings like nthreads and chunk for all
    GraphBLAS operations executed by the calling thread until the context
    is disengaged.

    >>> ctx = context_new()
    >>> context_engage(ctx)
    >>> context_disengage(ctx)

    """
    check_status(ctx, lib.GxB_Context_engage(ctx[0]))


def context_disengage(ctx):
    """Disengage a context from the calling thread.

    After disengaging, the calling thread reverts to the default
    (world) context settings.

    >>> ctx = context_new()
    >>> context_engage(ctx)
    >>> context_disengage(ctx)

    """
    check_status(ctx, lib.GxB_Context_disengage(ctx[0]))


# ---------------------------------------------------------------------------
# Option get/set (typed)
# ---------------------------------------------------------------------------


def context_get_int32(ctx, field):
    """Get a context option as an int32.

    >>> ctx = context_new()
    >>> context_get_int32(ctx, lib.GxB_CONTEXT_NTHREADS) >= 0
    True

    """
    val = ffi.new("int32_t*")
    check_status(ctx, lib.GxB_Context_get_INT32(ctx[0], field, val))
    return val[0]


def context_set_int32(ctx, field, value):
    """Set a context option from an int32.

    >>> ctx = context_new()
    >>> context_set_int32(ctx, lib.GxB_CONTEXT_NTHREADS, 2)
    >>> context_get_int32(ctx, lib.GxB_CONTEXT_NTHREADS)
    2

    """
    check_status(ctx, lib.GxB_Context_set_INT32(ctx[0], field, ffi.cast("int32_t", value)))


def context_get_fp64(ctx, field):
    """Get a context option as a float64.

    >>> ctx = context_new()
    >>> isinstance(context_get_fp64(ctx, lib.GxB_CONTEXT_CHUNK), float)
    True

    """
    val = ffi.new("double*")
    check_status(ctx, lib.GxB_Context_get_FP64(ctx[0], field, val))
    return val[0]


def context_set_fp64(ctx, field, value):
    """Set a context option from a float64.

    >>> ctx = context_new()
    >>> context_set_fp64(ctx, lib.GxB_CONTEXT_CHUNK, 4096.0)
    >>> context_get_fp64(ctx, lib.GxB_CONTEXT_CHUNK)
    4096.0

    """
    check_status(ctx, lib.GxB_Context_set_FP64(ctx[0], field, ffi.cast("double", value)))


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------


def context_get_nthreads(ctx):
    """Get the number of threads for a context.

    >>> ctx = context_new()
    >>> context_get_nthreads(ctx) >= 0
    True

    """
    return context_get_int32(ctx, lib.GxB_CONTEXT_NTHREADS)


def context_set_nthreads(ctx, nthreads):
    """Set the number of threads for a context.

    >>> ctx = context_new()
    >>> context_set_nthreads(ctx, 4)
    >>> context_get_nthreads(ctx)
    4

    """
    context_set_int32(ctx, lib.GxB_CONTEXT_NTHREADS, nthreads)


def context_get_chunk(ctx):
    """Get the chunk size for a context.

    The chunk size determines the minimum amount of work per thread.

    >>> ctx = context_new()
    >>> isinstance(context_get_chunk(ctx), float)
    True

    """
    return context_get_fp64(ctx, lib.GxB_CONTEXT_CHUNK)


def context_set_chunk(ctx, chunk):
    """Set the chunk size for a context.

    >>> ctx = context_new()
    >>> context_set_chunk(ctx, 4096.0)
    >>> context_get_chunk(ctx)
    4096.0

    """
    context_set_fp64(ctx, lib.GxB_CONTEXT_CHUNK, chunk)


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------


def context_print(ctx, name="", level=lib.GxB_COMPLETE):
    """Print a context to stdout.

    ``level`` controls verbosity: ``lib.GxB_SILENT``, ``lib.GxB_SUMMARY``,
    ``lib.GxB_SHORT``, ``lib.GxB_COMPLETE``, ``lib.GxB_SHORT_VERBOSE``,
    or ``lib.GxB_COMPLETE_VERBOSE``.

    >>> ctx = context_new()
    >>> out = _capture_c_output(context_print, ctx, 'ctx', lib.GxB_SHORT)
    >>> 'Context' in out
    True

    """
    check_status(ctx, lib.GxB_Context_fprint(
        ctx[0], name.encode() if isinstance(name, str) else name,
        level, ffi.NULL,
    ))


def context_fprint(ctx, f, name="", level=lib.GxB_COMPLETE):
    """Print a context to a C FILE* stream.

    Pass ``ffi.NULL`` for ``f`` to print to stdout.
    ``level`` controls verbosity (see :func:`context_print`).

    >>> ctx = context_new()
    >>> out = _capture_c_output(context_fprint, ctx, ffi.NULL, 'ctx', lib.GxB_SHORT)
    >>> 'Context' in out
    True

    """
    check_status(ctx, lib.GxB_Context_fprint(
        ctx[0], name.encode() if isinstance(name, str) else name,
        level, f,
    ))
