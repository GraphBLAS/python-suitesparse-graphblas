import numpy as np
from cpython.ref cimport Py_INCREF
from libc.stdint cimport uintptr_t
from numpy cimport NPY_ARRAY_F_CONTIGUOUS, NPY_ARRAY_OWNDATA, NPY_ARRAY_WRITEABLE
from numpy cimport dtype as dtype_t
from numpy cimport import_array, ndarray, npy_intp

import_array()

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

    cdef GxB_init func = <GxB_init><uintptr_t>int(ffi.cast("uintptr_t", ffi.addressof(lib, "GxB_init")))
    return func(<GrB_Mode>mode, PyDataMem_NEW, PyDataMem_NEW_ZEROED, PyDataMem_RENEW, PyDataMem_FREE)


cpdef ndarray claim_buffer(object ffi, object cdata, size_t size, dtype_t dtype):
    cdef:
        npy_intp dims = size
        uintptr_t ptr = int(ffi.cast("uintptr_t", cdata))
        ndarray array
    Py_INCREF(dtype)
    array = PyArray_NewFromDescr(
        ndarray, dtype, 1, &dims, NULL, <void*>ptr, NPY_ARRAY_WRITEABLE, <object>NULL
    )
    PyArray_ENABLEFLAGS(array, NPY_ARRAY_OWNDATA)
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
        PyArray_ENABLEFLAGS(array, NPY_ARRAY_OWNDATA)
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


cpdef unclaim_buffer(ndarray array):
    PyArray_CLEARFLAGS(array, NPY_ARRAY_OWNDATA | NPY_ARRAY_WRITEABLE)
