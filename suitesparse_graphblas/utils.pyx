# cython: freethreading_compatible=True
#
# We don't do anything special to support free-threading, but GraphBLAS C
# libraries are required to be thread-safe, so things should "just work".
# Of course, users writing multithreaded code can find many creative ways
# to fail, but python-suitesparse-graphblas shouldn't crash.
import numpy as np
from cpython.ref cimport Py_INCREF
from libc.stdint cimport uintptr_t
from numpy cimport NPY_ARRAY_F_CONTIGUOUS, NPY_ARRAY_OWNDATA, NPY_ARRAY_WRITEABLE
from numpy cimport dtype as dtype_t
from numpy cimport PyArray_CHKFLAGS, import_array, ndarray, npy_intp

import_array()


# Whether GraphBLAS was given NumPy's allocator (`call_gxb_init`) instead of libc's (`GrB_init`)
cdef bint uses_numpy_allocator = False


cdef inline void own_data(ndarray array):
    # Make `array` own (and eventually free) its data, which GraphBLAS allocated.
    # An array wrapped around existing data has no memory handler, and NumPy frees
    # the data of such an array with libc `free`.  That is the wrong deallocator for
    # data from `PyDataMem_NEW` whenever `PyDataMem_NEW` is not libc `malloc`:
    # NumPy >=2.5 uses `PyMem_RawMalloc`, which is mimalloc on free-threaded
    # Python >=3.15 and adds debug hooks under `PYTHONMALLOC=debug` or `python -X dev`.
    # NumPy's default handler frees with the allocator `PyDataMem_NEW` allocates with.
    cdef PyArrayObject_fields *fields = <PyArrayObject_fields *><void *>array
    if uses_numpy_allocator and fields.mem_handler == NULL:
        Py_INCREF(<object>PyDataMem_DefaultHandler)
        fields.mem_handler = PyDataMem_DefaultHandler
    PyArray_ENABLEFLAGS(array, NPY_ARRAY_OWNDATA)


cpdef int call_gxb_init(object ffi, object lib, int mode):
    # We need to call `GxB_init`, but we didn't compile Cython against GraphBLAS.  So, we get it from cffi.
    # Step 1: ffi.addressof(lib, "GxB_init")
    #    Return type: cffi.cdata object of a function pointer.  Can't cast to int.
    # Step 2: ffi.cast("uintptr_t", ...)
    #    Return type: cffi.cdata object of a uintptr_t type, an unsigned pointer.  Can cast to int.
    # Step 3: int(...)
    #    Return type: int.  The physical address of the function.
    # Step 4: <uintptr_t>(...)
    #    Return type: uintptr_t in Cython.  Cast Python int to Cython integer for pointers.
    # Step 5: <GsB_init>(...)
    #    Return: function pointer in Cython!

    global uses_numpy_allocator
    cdef GxB_init func = <GxB_init><uintptr_t>int(ffi.cast("uintptr_t", ffi.addressof(lib, "GxB_init")))
    cdef int info = func(<GrB_Mode>mode, PyDataMem_NEW, PyDataMem_NEW_ZEROED, PyDataMem_RENEW, PyDataMem_FREE)
    if info == 0:  # GrB_SUCCESS
        uses_numpy_allocator = True
    return info


cpdef ndarray claim_buffer(object ffi, object cdata, size_t size, dtype_t dtype):
    cdef:
        npy_intp dims = size
        uintptr_t ptr = int(ffi.cast("uintptr_t", cdata))
        ndarray array
    Py_INCREF(dtype)
    array = PyArray_NewFromDescr(
        ndarray, dtype, 1, &dims, NULL, <void*>ptr, NPY_ARRAY_WRITEABLE, <object>NULL
    )
    own_data(array)
    return array


cpdef ndarray claim_buffer_2d(
    object ffi, object cdata, size_t cdata_size, size_t nrows, size_t ncols, dtype_t dtype, bint is_c_order
):
    cdef:
        size_t size = nrows * ncols
        ndarray array
        uintptr_t ptr
        npy_intp dims[2]
        int flags = NPY_ARRAY_WRITEABLE
    if cdata_size == size:
        ptr = int(ffi.cast("uintptr_t", cdata))
        dims[0] = nrows
        dims[1] = ncols
        if not is_c_order:
            flags |= NPY_ARRAY_F_CONTIGUOUS
        Py_INCREF(dtype)
        array = PyArray_NewFromDescr(
            ndarray, dtype, 2, dims, NULL, <void*>ptr, flags, <object>NULL
        )
        own_data(array)
    elif cdata_size > size:  # pragma: no cover
        array = claim_buffer(ffi, cdata, cdata_size, dtype)
        if is_c_order:
            array = array[:size].reshape((nrows, ncols))
        else:
            array = array[:size].reshape((ncols, nrows)).T
    else:  # pragma: no cover
        raise ValueError(
            f"Buffer size too small: {cdata_size}.  "
            f"Unable to create matrix of size {nrows}x{ncols} = {size}"
        )
    return array


cpdef bint can_unclaim_buffer(ndarray array):
    """Whether GraphBLAS may take ownership of the data of ``array``.

    GraphBLAS frees data it owns with the allocator it was initialized with, so
    ``array`` must own its data and have allocated it with that same allocator.
    Arrays that NumPy allocated qualify (unless ``memory_manager="c"`` or a custom
    NumPy memory handler was in use), and so do arrays from ``claim_buffer``; views
    never do.  When this returns False, give GraphBLAS a copy that does qualify, or
    data it does not own (e.g. ``GxB_IS_READONLY``).

    This cannot see other references to the data: GraphBLAS may free or reallocate
    it at any time after the handover, so no view of ``array`` (nor memoryview or
    cffi pointer into it) may outlive the handover.
    """
    cdef PyArrayObject_fields *fields = <PyArrayObject_fields *><void *>array
    if not PyArray_CHKFLAGS(array, NPY_ARRAY_OWNDATA):
        return False
    if uses_numpy_allocator:
        return fields.mem_handler == PyDataMem_DefaultHandler
    # NumPy takes the data of an array without a memory handler to be from libc
    return fields.mem_handler == NULL


cpdef unclaim_buffer(ndarray array):
    """Make ``array`` a read-only view of data that GraphBLAS now owns.

    Call this once GraphBLAS has taken ownership of the data, which is only safe
    if ``can_unclaim_buffer(array)`` was True; this function cannot undo the handover.
    """
    PyArray_CLEARFLAGS(array, NPY_ARRAY_OWNDATA | NPY_ARRAY_WRITEABLE)
