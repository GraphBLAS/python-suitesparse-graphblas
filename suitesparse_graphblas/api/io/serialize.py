import operator
import warnings

import numpy as np

from suitesparse_graphblas import check_status, ffi, lib
from suitesparse_graphblas.api.global_options import global_get_int32
from suitesparse_graphblas.exceptions import InvalidObject, InvalidValue
from suitesparse_graphblas.utils import claim_buffer


def _free_desc(desc):
    """Free a descriptor via a temporary pointer."""
    desc_ptr = ffi.new("GrB_Descriptor*", desc)
    lib.GrB_Descriptor_free(desc_ptr)


def _call_with_nthreads(nthreads, func, *args):
    """Return ``func(*args)``, called under a new GxB_Context with `nthreads` threads.

    GraphBLAS 10 takes the number of threads from the Context engaged on the calling thread,
    not from a descriptor.  If `nthreads` is None, 0 or negative, `func` is just called.
    `nthreads` must be an integer, and not a bool.

    Apart from the number of threads, the new Context is a copy of GxB_CONTEXT_WORLD:
    GxB_Context_new copies its chunk and GPUs, and the arenas are copied here, since they
    would otherwise be arena 0.  WORLD is the Context the call would use without `nthreads`,
    because a Context engaged on this thread makes this raise.

    It all happens in this one frame, in try and finally blocks, rather than in a generator
    context manager.  A KeyboardInterrupt can come as the context manager's ``__exit__``
    begins, before the generator resumes, and that leaves disengaging and freeing the Context
    to garbage collection: later, and on PyPy perhaps in another thread.
    """
    if nthreads is not None:
        # NumPy 2.2 still lets `operator.index` take its bool, with a DeprecationWarning
        if isinstance(nthreads, (bool, np.bool_)):
            raise TypeError("nthreads must be an integer, not a bool")
        nthreads = operator.index(nthreads)  # TypeError if it isn't an integer
    if nthreads is None or nthreads <= 0:
        return func(*args)
    ctx = ffi.new("GxB_Context*")
    # A KeyboardInterrupt between here and `try` leaks the Context, which isn't engaged yet
    check_status(ctx, lib.GxB_Context_new(ctx))
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
        try:
            check_status(ctx, lib.GxB_Context_engage(ctx[0]))
            return func(*args)
        finally:
            lib.GxB_Context_disengage(ctx[0])
    finally:
        # Disengaging a Context that isn't engaged changes nothing, so disengage it again, in
        # case a KeyboardInterrupt stopped the `finally` above before it could (PyPy may stop
        # it at its first bytecode).  Freeing it while engaged would leave the thread with a
        # dangling Context.
        lib.GxB_Context_disengage(ctx[0])
        lib.GxB_Context_free(ctx)


def _check_deserialized(obj, fprint, free, describe):
    """Free `obj` and raise InvalidObject(`describe()`) if `fprint` finds it invalid.

    GraphBLAS checks only enough of a blob to deserialize it without reading past its end, so
    a corrupt blob can make an invalid object, and its User Guide says to check the result
    with a silent ``GxB_*_fprint``.
    """
    info = fprint(obj[0], b"", lib.GxB_SILENT, ffi.NULL)
    if info == lib.GrB_SUCCESS:
        return
    try:
        if info == lib.GrB_INVALID_OBJECT:
            raise InvalidObject(describe())
        check_status(obj, info)
    finally:
        free(obj)


def _describe_invalid_vector(v):
    # GxB_Vector_deserialize takes a Matrix blob too, and its Matrix makes an invalid Vector
    # unless it is n-by-1 and held by column, as a Vector is
    A = ffi.cast("GrB_Matrix", v[0])
    if lib.GxB_Matrix_fprint(A, b"", lib.GxB_SILENT, ffi.NULL) == lib.GrB_SUCCESS:
        nrows, ncols = ffi.new("GrB_Index*"), ffi.new("GrB_Index*")
        check_status(v, lib.GrB_Matrix_nrows(nrows, A))
        check_status(v, lib.GrB_Matrix_ncols(ncols, A))
        return f"data holds a {nrows[0]}x{ncols[0]} Matrix, not a Vector"
    return "data does not hold a valid Vector: GxB_Vector_fprint rejects what it deserializes to"


def get_serialize_desc(compression=lib.GxB_COMPRESSION_DEFAULT, level=None, nthreads=None):
    """Create a descriptor for serializing or deserializing.

    This returns None (for NULL descriptor) or a GrB_Descriptor.

    .. deprecated:: 10.5.3.0
        The `nthreads` argument.  The number of threads isn't a descriptor setting in
        GraphBLAS 10, which raises InvalidValue if `nthreads` is given.  Use the `nthreads`
        argument of serialize_matrix, serialize_vector, deserialize_matrix or
        deserialize_vector instead, which sets it in a GxB_Context.
    """
    if nthreads is not None:
        warnings.warn(
            "The `nthreads` argument of get_serialize_desc is deprecated: GraphBLAS 10 "
            "rejects the number of threads in a descriptor.  Use the `nthreads` argument "
            "of serialize_matrix, serialize_vector, deserialize_matrix or "
            "deserialize_vector instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        # OverflowError if it doesn't fit in int32_t, rather than wrapping around
        check_status(desc, lib.GxB_Desc_set_INT32(desc, lib.GxB_COMPRESSION, compression))
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
        The maximum number of OpenMP threads to use.  None (the default), 0 or any negative
        integer, however large, means the number of threads of the GxB_Context engaged on
        this thread, if any, or else of GxB_CONTEXT_WORLD.  A positive integer makes the
        call run under a new Context with that many threads.  It raises OverflowError if
        that is more than int32_t holds, and InvalidValue if a Context is engaged on this
        thread (which stays engaged: set its number of threads instead).  Raises TypeError
        if `nthreads` isn't an integer, or is a bool.
    """
    desc = get_serialize_desc(compression, level)
    data_ptr = ffi.new("void**")
    size_ptr = ffi.new("GrB_Index*")
    # Name the arena for the result, which is otherwise that of the engaged Context (if
    # any), so that `claim_buffer` knows which allocator must free it
    arena = global_get_int32(lib.GxB_ARENA_DATA)
    desc = ffi.NULL if desc is None else desc
    info = _call_with_nthreads(
        nthreads, lib.GxB_Matrix_serialize_arena, data_ptr, size_ptr, A[0], arena, desc
    )
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
        The maximum number of OpenMP threads to use.  None (the default), 0 or any negative
        integer, however large, means the number of threads of the GxB_Context engaged on
        this thread, if any, or else of GxB_CONTEXT_WORLD.  A positive integer makes the
        call run under a new Context with that many threads.  It raises OverflowError if
        that is more than int32_t holds, and InvalidValue if a Context is engaged on this
        thread (which stays engaged: set its number of threads instead).  Raises TypeError
        if `nthreads` isn't an integer, or is a bool.
    """
    desc = get_serialize_desc(compression, level)
    data_ptr = ffi.new("void**")
    size_ptr = ffi.new("GrB_Index*")
    arena = global_get_int32(lib.GxB_ARENA_DATA)  # see `serialize_matrix`
    desc = ffi.NULL if desc is None else desc
    info = _call_with_nthreads(
        nthreads, lib.GxB_Vector_serialize_arena, data_ptr, size_ptr, v[0], arena, desc
    )
    check_status(v, info)
    return claim_buffer(ffi, data_ptr[0], size_ptr[0], np.dtype(np.uint8), arena)


def deserialize_matrix(data, *, free=True, nthreads=None):
    """Deserialize a Matrix from bytes.

    Parameters
    ----------
    data : bytes-like
        A serialized Matrix, such as the result of `serialize_matrix`: bytes, a NumPy array, or
        anything else with the buffer protocol.
    free : callable or bool, optional
        Called with the result when it is garbage collected.  True (the default) means
        `matrix.matrix_free`.  If `free` is None or False, there is no automatic garbage
        collection, and it is up to the caller to free the Matrix.
    nthreads : int, optional
        The maximum number of OpenMP threads to use.  None (the default), 0 or any negative
        integer, however large, means the number of threads of the GxB_Context engaged on
        this thread, if any, or else of GxB_CONTEXT_WORLD.  A positive integer makes the
        call run under a new Context with that many threads.  It raises OverflowError if
        that is more than int32_t holds, and InvalidValue if a Context is engaged on this
        thread (which stays engaged: set its number of threads instead).  Raises TypeError
        if `nthreads` isn't an integer, or is a bool.

    Raises InvalidObject, after freeing what it deserialized, if the structure of the Matrix
    in `data` (its indices, pointers and bitmap) is invalid: GraphBLAS checks only that it can
    read the blob.  The check takes O(nvec + nvals) time, or O(nrows * ncols) for a bitmap.
    It doesn't check the values.
    """
    data = np.frombuffer(data, np.uint8)
    A = ffi.new("GrB_Matrix*")
    info = _call_with_nthreads(
        nthreads,
        lib.GxB_Matrix_deserialize,
        A,
        ffi.NULL,  # dtype; we don't check for now
        ffi.from_buffer("void*", data),
        data.nbytes,
        ffi.NULL,
    )
    check_status(A, info)
    _check_deserialized(
        A,
        lib.GxB_Matrix_fprint,
        lib.GrB_Matrix_free,
        lambda: "data does not hold a valid Matrix: GxB_Matrix_fprint rejects what it "
        "deserializes to",
    )
    if free:
        if callable(free):
            return ffi.gc(A, free)
        return ffi.gc(A, matrix.matrix_free)
    return A


def deserialize_vector(data, *, free=True, nthreads=None):
    """Deserialize a Vector from bytes.

    Parameters
    ----------
    data : bytes-like
        A serialized Vector, such as the result of `serialize_vector`: bytes, a NumPy array, or
        anything else with the buffer protocol.
    free : callable or bool, optional
        Called with the result when it is garbage collected.  True (the default) means
        `vector.vector_free`.  If `free` is None or False, there is no automatic garbage
        collection, and it is up to the caller to free the Vector.
    nthreads : int, optional
        The maximum number of OpenMP threads to use.  None (the default), 0 or any negative
        integer, however large, means the number of threads of the GxB_Context engaged on
        this thread, if any, or else of GxB_CONTEXT_WORLD.  A positive integer makes the
        call run under a new Context with that many threads.  It raises OverflowError if
        that is more than int32_t holds, and InvalidValue if a Context is engaged on this
        thread (which stays engaged: set its number of threads instead).  Raises TypeError
        if `nthreads` isn't an integer, or is a bool.

    Raises InvalidObject, after freeing what it deserialized, if the structure of the Vector
    in `data` (its indices and bitmap) is invalid: GraphBLAS checks only that it can read the
    blob.  A Matrix blob holds a valid Vector only if the Matrix is n-by-1 and held by column.
    The check takes O(nvals) time, or O(size) for a bitmap.  It doesn't check the values.
    """
    data = np.frombuffer(data, np.uint8)
    v = ffi.new("GrB_Vector*")
    info = _call_with_nthreads(
        nthreads,
        lib.GxB_Vector_deserialize,
        v,
        ffi.NULL,  # dtype; we don't check for now
        ffi.from_buffer("void*", data),
        data.nbytes,
        ffi.NULL,
    )
    check_status(v, info)
    _check_deserialized(
        v, lib.GxB_Vector_fprint, lib.GrB_Vector_free, lambda: _describe_invalid_vector(v)
    )
    if free:
        if callable(free):
            return ffi.gc(v, free)
        return ffi.gc(v, vector.vector_free)
    return v


from suitesparse_graphblas.api import matrix, vector  # noqa: E402 isort:skip
