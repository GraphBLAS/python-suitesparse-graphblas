from libc.stdint cimport uint64_t
from numpy cimport dtype as dtype_t
from numpy cimport ndarray, npy_intp


cdef extern from "numpy/arrayobject.h" nogil:
    # These aren't public (i.e., "extern"), but other projects use them too
    void *PyDataMem_NEW(size_t size)
    void *PyDataMem_NEW_ZEROED(size_t nmemb, size_t size)
    void *PyDataMem_RENEW(void *ptr, size_t size)
    void PyDataMem_FREE(void *ptr)
    # These are available in newer Cython versions
    void PyArray_ENABLEFLAGS(ndarray array, int flags)
    void PyArray_CLEARFLAGS(ndarray array, int flags)
    # Not exposed by Cython (b/c it steals a reference from dtype)
    ndarray PyArray_NewFromDescr(
        type subtype, dtype_t dtype, int nd, npy_intp *dims, npy_intp *strides, void *data, int flags, object obj
    )

ctypedef enum GrB_Mode:
    GrB_NONBLOCKING
    GrB_BLOCKING

ctypedef uint64_t (*GxB_init)(
    GrB_Mode,
    void *(*user_malloc_function)(size_t),
    void *(*user_calloc_function)(size_t, size_t),
    void *(*user_realloc_function)(void *, size_t),
    void (*user_free_function)(void *),
)

cpdef int call_gxb_init(object ffi, object lib, int mode)

cpdef ndarray claim_buffer(object ffi, object cdata, size_t size, dtype_t dtype)

cpdef ndarray claim_buffer_2d(
    object ffi, object cdata, size_t cdata_size, size_t nrows, size_t ncols, dtype_t dtype, bint is_c_order
)

cpdef unclaim_buffer(ndarray array)
