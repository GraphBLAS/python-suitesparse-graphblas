import operator
from contextlib import contextmanager

import numpy as np

from suitesparse_graphblas import check_status, ffi, lib
from suitesparse_graphblas.api.context import (
    context_disengage,
    context_engage,
    context_free,
    context_new,
)
from suitesparse_graphblas.api.global_options import global_get_int32
from suitesparse_graphblas.exceptions import InvalidValue
from suitesparse_graphblas.utils import claim_buffer


def _free_desc(desc):
    """Free a descriptor via a temporary pointer."""
    desc_ptr = ffi.new("GrB_Descriptor*", desc)
    lib.GrB_Descriptor_free(desc_ptr)


@contextmanager
def _nthreads_context(nthreads):
    """Engage a new GxB_Context that uses at most `nthreads` threads, for the body.

    GraphBLAS 10 takes the number of threads from the Context engaged on the calling thread,
    not from a descriptor.  Apart from the number of threads, the new Context is a copy of
    GxB_CONTEXT_WORLD: GxB_Context_new copies its chunk and GPUs, and the arenas are copied
    here, since they would otherwise be arena 0.  Nothing is done if `nthreads` is None, 0 or
    negative.
    """
    if nthreads is not None:
        nthreads = operator.index(nthreads)  # TypeError if it isn't an integer
    if nthreads is None or nthreads <= 0:
        yield
        return
    ctx = context_new(free=None)
    try:
        # A Context engaged on this thread would be lost: GraphBLAS can't say which one it
        # is, so it can't be engaged again afterwards.  But it can say whether there is one:
        # disengaging a Context that isn't engaged fails with GrB_INVALID_VALUE if another
        # one is, and changes nothing.
        info = lib.GxB_Context_disengage(ctx[0])
        if info == lib.GrB_INVALID_VALUE:
            raise InvalidValue(
                "nthreads can't be given while a GxB_Context is engaged on this thread; "
                "set GxB_CONTEXT_NTHREADS of that Context instead"
            )
        check_status(ctx, info)
        # OverflowError if `nthreads` doesn't fit in int32_t
        check_status(ctx, lib.GxB_Context_set_INT(ctx[0], nthreads, lib.GxB_CONTEXT_NTHREADS))
        header_arena = global_get_int32(lib.GxB_ARENA_HEADER)
        check_status(ctx, lib.GxB_Context_set_INT(ctx[0], header_arena, lib.GxB_ARENA_HEADER))
        data_arena = global_get_int32(lib.GxB_ARENA_DATA)
        check_status(ctx, lib.GxB_Context_set_INT(ctx[0], data_arena, lib.GxB_ARENA_DATA))
        context_engage(ctx)
        try:
            yield
        finally:
            context_disengage(ctx)
    finally:
        context_free(ctx)


def get_serialize_desc(compression=lib.GxB_COMPRESSION_DEFAULT, level=None, nthreads=None):
    """Create a descriptor for serializing or deserializing.

    This returns None (for NULL descriptor) or a GrB_Descriptor.

    The number of threads isn't a descriptor setting in GraphBLAS 10, which raises
    InvalidValue if `nthreads` is given.  The `nthreads` argument of the functions that
    serialize and deserialize sets it in a GxB_Context instead.
    """
    if nthreads is None and (compression is None or compression == lib.GxB_COMPRESSION_DEFAULT):
        return None
    desc_ptr = ffi.new("GrB_Descriptor*")
    check_status(desc_ptr, lib.GrB_Descriptor_new(desc_ptr))
    desc = ffi.gc(desc_ptr[0], _free_desc)
    if nthreads is not None:
        check_status(
            desc,
            lib.GxB_Desc_set_INT32(desc, lib.GxB_NTHREADS, ffi.cast("int32_t", nthreads)),
        )
    if compression is not None:
        if level is not None and compression in {
            lib.GxB_COMPRESSION_LZ4HC,
            lib.GxB_COMPRESSION_ZSTD,
        }:
            compression += level
        check_status(
            desc,
            lib.GxB_Desc_set_INT32(desc, lib.GxB_COMPRESSION, ffi.cast("int32_t", compression)),
        )
    return desc


def serialize_matrix(A, compression=lib.GxB_COMPRESSION_DEFAULT, level=None, *, nthreads=None):
    """Serialize a Matrix into an array of bytes.

    Parameters
    ----------
    compression : int, optional
        One of None, GxB_COMPRESSION_NONE, GxB_COMPRESSION_DEFAULT,
        GxB_COMPRESSION_LZ4, GxB_COMPRESSION_LZ4HC, or GxB_COMPRESSION_ZSTD
    level : int, optional
        For GxB_COMPRESSION_LZ4HC, should be between 1 and 9, where 9 is most compressed.
        For GxB_COMPRESSION_ZSTD, should be between 1 and 19, where 19 is most compressed.
    nthreads : int, optional
        The maximum number of OpenMP threads to use.  None (the default), 0 or negative
        means the number of threads of the GxB_Context engaged on this thread, if any, or
        else of GxB_CONTEXT_WORLD.  A positive number makes the call run under a new
        Context with that many threads, so it raises InvalidValue if a Context is engaged
        on this thread (which stays engaged: set its number of threads instead).
        Raises TypeError if `nthreads` isn't an integer, OverflowError if it doesn't fit
        in int32_t.
    """
    desc = get_serialize_desc(compression, level)
    data_ptr = ffi.new("void**")
    size_ptr = ffi.new("GrB_Index*")
    # Name the arena for the result, which is otherwise that of the engaged Context (if
    # any), so that `claim_buffer` knows which allocator must free it
    arena = global_get_int32(lib.GxB_ARENA_DATA)
    desc = ffi.NULL if desc is None else desc
    with _nthreads_context(nthreads):
        info = lib.GxB_Matrix_serialize_arena(data_ptr, size_ptr, A[0], arena, desc)
    check_status(A, info)
    return claim_buffer(ffi, data_ptr[0], size_ptr[0], np.dtype(np.uint8), arena)


def serialize_vector(v, compression=lib.GxB_COMPRESSION_DEFAULT, level=None, *, nthreads=None):
    """Serialize a Vector into an array of bytes.

    Parameters
    ----------
    compression : int, optional
        One of None, GxB_COMPRESSION_NONE, GxB_COMPRESSION_DEFAULT,
        GxB_COMPRESSION_LZ4, GxB_COMPRESSION_LZ4HC, or GxB_COMPRESSION_ZSTD
    level : int, optional
        For GxB_COMPRESSION_LZ4HC, should be between 1 and 9, where 9 is most compressed.
        For GxB_COMPRESSION_ZSTD, should be between 1 and 19, where 19 is most compressed.
    nthreads : int, optional
        The maximum number of OpenMP threads to use.  None (the default), 0 or negative
        means the number of threads of the GxB_Context engaged on this thread, if any, or
        else of GxB_CONTEXT_WORLD.  A positive number makes the call run under a new
        Context with that many threads, so it raises InvalidValue if a Context is engaged
        on this thread (which stays engaged: set its number of threads instead).
        Raises TypeError if `nthreads` isn't an integer, OverflowError if it doesn't fit
        in int32_t.
    """
    desc = get_serialize_desc(compression, level)
    data_ptr = ffi.new("void**")
    size_ptr = ffi.new("GrB_Index*")
    arena = global_get_int32(lib.GxB_ARENA_DATA)  # see `serialize_matrix`
    desc = ffi.NULL if desc is None else desc
    with _nthreads_context(nthreads):
        info = lib.GxB_Vector_serialize_arena(data_ptr, size_ptr, v[0], arena, desc)
    check_status(v, info)
    return claim_buffer(ffi, data_ptr[0], size_ptr[0], np.dtype(np.uint8), arena)


def deserialize_matrix(data, *, free=True, nthreads=None):
    """Deserialize a Matrix from bytes.

    The `free` argument is called when the object is garbage
    collected, the default is `matrix.matrix_free()`.  If `free` is None then
    there is no automatic garbage collection and it is up to the user
    to free the matrix.

    Parameters
    ----------
    nthreads : int, optional
        The maximum number of OpenMP threads to use.  None (the default), 0 or negative
        means the number of threads of the GxB_Context engaged on this thread, if any, or
        else of GxB_CONTEXT_WORLD.  A positive number makes the call run under a new
        Context with that many threads, so it raises InvalidValue if a Context is engaged
        on this thread (which stays engaged: set its number of threads instead).
        Raises TypeError if `nthreads` isn't an integer, OverflowError if it doesn't fit
        in int32_t.
    """
    data = np.frombuffer(data, np.uint8)
    A = ffi.new("GrB_Matrix*")
    with _nthreads_context(nthreads):
        info = lib.GxB_Matrix_deserialize(
            A,
            ffi.NULL,  # dtype; we don't check for now
            ffi.from_buffer("void*", data),
            data.nbytes,
            ffi.NULL,
        )
    check_status(A, info)
    if free:
        if callable(free):
            return ffi.gc(A, free)
        return ffi.gc(A, matrix.matrix_free)
    return A


def deserialize_vector(data, *, free=True, nthreads=None):
    """Deserialize a Vector from bytes.

    The `free` argument is called when the object is garbage
    collected, the default is `vector.vector_free()`.  If `free` is None then
    there is no automatic garbage collection and it is up to the user
    to free the vector.

    Parameters
    ----------
    nthreads : int, optional
        The maximum number of OpenMP threads to use.  None (the default), 0 or negative
        means the number of threads of the GxB_Context engaged on this thread, if any, or
        else of GxB_CONTEXT_WORLD.  A positive number makes the call run under a new
        Context with that many threads, so it raises InvalidValue if a Context is engaged
        on this thread (which stays engaged: set its number of threads instead).
        Raises TypeError if `nthreads` isn't an integer, OverflowError if it doesn't fit
        in int32_t.
    """
    data = np.frombuffer(data, np.uint8)
    v = ffi.new("GrB_Vector*")
    with _nthreads_context(nthreads):
        info = lib.GxB_Vector_deserialize(
            v,
            ffi.NULL,  # dtype; we don't check for now
            ffi.from_buffer("void*", data),
            data.nbytes,
            ffi.NULL,
        )
    check_status(v, info)
    if free:
        if callable(free):
            return ffi.gc(v, free)
        return ffi.gc(v, vector.vector_free)
    return v


from suitesparse_graphblas.api import matrix, vector  # noqa: E402 isort:skip
