from cpython.ref cimport PyObject
from libc.stdint cimport int32_t, uint8_t
from numpy cimport dtype as dtype_t
from numpy cimport ndarray, npy_intp


cdef extern from "numpy/arrayobject.h" nogil:
    # These aren't public (i.e., "extern"), but other projects use them too
    void *PyDataMem_NEW(size_t size)
    void *PyDataMem_NEW_ZEROED(size_t nmemb, size_t size)
    void *PyDataMem_RENEW(void *ptr, size_t size)
    void PyDataMem_FREE(void *ptr)
    # The memory handler (a PyCapsule) whose allocator matches the functions above
    PyObject *PyDataMem_DefaultHandler
    # NumPy memory handlers (NEP 49); like `mem_handler`, these need
    # NPY_TARGET_VERSION >= NPY_1_22_API_VERSION (see setup.py)
    ctypedef struct PyDataMemAllocator:
        void *ctx
        void *(*malloc)(void *ctx, size_t size) noexcept nogil
        void *(*calloc)(void *ctx, size_t nelem, size_t elsize) noexcept nogil
        void *(*realloc)(void *ctx, void *ptr, size_t new_size) noexcept nogil
        void (*free)(void *ctx, void *ptr, size_t size) noexcept nogil
    ctypedef struct PyDataMem_Handler:
        char name[127]
        uint8_t version
        PyDataMemAllocator allocator
    ctypedef struct PyArrayObject_fields:
        PyObject *mem_handler
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

# GrB_Info is a C enum (int); errors are negative
ctypedef int (*GxB_init)(
    GrB_Mode,
    void *(*user_malloc_function)(size_t),
    void *(*user_calloc_function)(size_t, size_t),
    void *(*user_realloc_function)(void *, size_t),
    void (*user_free_function)(void *),
)
# GrB_Global_get_INT32
ctypedef int (*global_get_int32_t)(void *, int32_t *, int) noexcept nogil

cpdef int call_gxb_init(object ffi, object lib, int mode)

cpdef ndarray claim_buffer(object ffi, object cdata, size_t size, dtype_t dtype, object arena=*)

cpdef ndarray claim_buffer_2d(
    object ffi,
    object cdata,
    size_t cdata_size,
    size_t nrows,
    size_t ncols,
    dtype_t dtype,
    bint is_c_order,
    object arena=*,
)

cpdef bint can_unclaim_buffer(object array, object arena=*) except -1

cpdef unclaim_buffer(ndarray array, object arena=*)
