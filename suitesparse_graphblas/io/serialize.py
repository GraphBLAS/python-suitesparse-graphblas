import numpy as np

from suitesparse_graphblas import check_status, ffi, lib
from suitesparse_graphblas.utils import claim_buffer


def free_desc(desc):
    """Free a descriptor."""
    check_status(desc, lib.GrB_Descriptor_free(desc))


def get_serialize_desc(compression=lib.GxB_COMPRESSION_DEFAULT, level=None, nthreads=None):
    """Create a descriptor for serializing or deserializing.

    This returns None (for NULL descriptor) or a pointer to a GrB_Descriptor.
    """
    if nthreads is None and (compression is None or compression == lib.GxB_COMPRESSION_DEFAULT):
        return None
    desc = ffi.new("GrB_Descriptor*")
    check_status(desc, lib.GrB_Descriptor_new(desc))
    desc = ffi.gc(desc, free_desc)
    if nthreads is not None:
        check_status(
            desc,
            lib.GxB_Desc_set_INT32(desc[0], lib.GxB_NTHREADS, ffi.cast("int32_t", nthreads)),
        )
    if compression is not None:
        if level is not None and compression in {
            lib.GxB_COMPRESSION_LZ4HC,
            lib.GxB_COMPRESSION_ZSTD,
        }:
            compression += level
        check_status(
            desc,
            lib.GxB_Desc_set_INT32(desc[0], lib.GxB_COMPRESSION, ffi.cast("int32_t", compression)),
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
        The maximum number of OpenMP threads to use.
    """
    desc = get_serialize_desc(compression, level, nthreads)
    data_ptr = ffi.new("void**")
    size_ptr = ffi.new("GrB_Index*")
    check_status(
        A, lib.GxB_Matrix_serialize(data_ptr, size_ptr, A[0], ffi.NULL if desc is None else desc[0])
    )
    return claim_buffer(ffi, data_ptr[0], size_ptr[0], np.dtype(np.uint8))


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
        The maximum number of OpenMP threads to use.
    """
    desc = get_serialize_desc(compression, level, nthreads)
    data_ptr = ffi.new("void**")
    size_ptr = ffi.new("GrB_Index*")
    check_status(
        v, lib.GxB_Vector_serialize(data_ptr, size_ptr, v[0], ffi.NULL if desc is None else desc[0])
    )
    return claim_buffer(ffi, data_ptr[0], size_ptr[0], np.dtype(np.uint8))


def deserialize_matrix(data, *, free=True, nthreads=None):
    """Deserialize a Matrix from bytes.

    The `free` argument is called when the object is garbage
    collected, the default is `matrix.free()`.  If `free` is None then
    there is no automatic garbage collection and it is up to the user
    to free the matrix.
    """
    data = np.frombuffer(data, np.uint8)
    desc = get_serialize_desc(None, nthreads)
    A = ffi.new("GrB_Matrix*")
    check_status(
        A,
        lib.GxB_Matrix_deserialize(
            A,
            ffi.NULL,  # dtype; we don't check for now
            ffi.from_buffer("void*", data),
            data.nbytes,
            ffi.NULL if desc is None else desc[0],
        ),
    )
    if free:
        if callable(free):
            return ffi.gc(A, free)
        return ffi.gc(A, matrix.free)
    return A


def deserialize_vector(data, *, free=True, nthreads=None):
    """Deserialize a Vector from bytes.

    The `free` argument is called when the object is garbage
    collected, the default is `vector.free()`.  If `free` is None then
    there is no automatic garbage collection and it is up to the user
    to free the vector.
    """
    data = np.frombuffer(data, np.uint8)
    desc = get_serialize_desc(None, nthreads)
    v = ffi.new("GrB_Vector*")
    check_status(
        v,
        lib.GxB_Vector_deserialize(
            v,
            ffi.NULL,  # dtype; we don't check for now
            ffi.from_buffer("void*", data),
            data.nbytes,
            ffi.NULL if desc is None else desc[0],
        ),
    )
    if free:
        if callable(free):
            return ffi.gc(v, free)
        return ffi.gc(v, vector.free)
    return v


from suitesparse_graphblas import matrix, vector  # noqa: E402 isort:skip
