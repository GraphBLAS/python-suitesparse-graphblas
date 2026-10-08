# cython: freethreading_compatible=True
#
# We don't do anything special to support free-threading, but GraphBLAS C
# libraries are required to be thread-safe, so things should "just work".
# Of course, users writing multithreaded code can find many creative ways
# to fail, but python-suitesparse-graphblas shouldn't crash.
"""Move memory between GraphBLAS and NumPy without copying.

Memory must be freed by the allocator that allocated it.  GraphBLAS frees the memory it
owns with the allocator of the *arena* the memory belongs to: arena 0 is set up by
``suitesparse_graphblas.initialize`` (NumPy's allocator for ``memory_manager="numpy"``,
libc's for ``"c"``), and ``GxB_arena_init`` can add more.  NumPy frees the data of an
array with the array's memory handler (NEP 49).  So:

- GraphBLAS -> NumPy: ``claim_buffer`` and ``claim_buffer_2d`` wrap memory GraphBLAS
  allocated in an array whose memory handler frees it with the same arena's allocator.
- NumPy -> GraphBLAS: ``give_buffer`` hands the data of an array to a GraphBLAS function
  that takes ownership of it, such as ``GxB_Matrix_pack_CSR`` or ``GxB_Vector_load``
  (copying it first if GraphBLAS can't own it).  ``can_unclaim_buffer`` and
  ``unclaim_buffer`` are the lower-level pieces.
- ``empty`` makes an array that GraphBLAS can always take ownership of.

Each function takes an optional ``arena``.  It defaults to GraphBLAS's global data arena
(``GxB_ARENA_DATA`` of ``GrB_GLOBAL``; 0 unless changed), which is where GraphBLAS
allocates, and where it expects memory it is given to be from, unless the calling thread
has engaged a ``GxB_Context`` with another data arena.  GraphBLAS can't be asked which
Context is engaged, so pass ``arena`` in that case.

Which arena a buffer from GraphBLAS is in: ``GxB_*_unpack_*`` and ``GxB_*_export_*`` first
move the object to the arena above, so their buffers are in it whatever arena the object
was made in; ``GxB_*_unload`` reports each buffer's arena (``handling - GrB_DEFAULT``).
"""
import operator
import threading
from contextlib import contextmanager

import numpy as np

from suitesparse_graphblas._graphblas import ffi as _ffi
from suitesparse_graphblas._graphblas import lib as _lib

from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_New
from cpython.pyport cimport PY_SSIZE_T_MAX
from cpython.ref cimport Py_INCREF
from libc.stdint cimport SIZE_MAX, int32_t, uintptr_t
from libc.stdio cimport snprintf
from libc.string cimport memset
from numpy cimport NPY_ARRAY_F_CONTIGUOUS, NPY_ARRAY_OWNDATA, NPY_ARRAY_WRITEABLE
from numpy cimport dtype as dtype_t
from numpy cimport PyArray_CHKFLAGS, PyArray_DATA, import_array, ndarray, npy_intp

import_array()


cdef enum:
    NARENAS = 8  # GxB_NARENAS

if _lib.GxB_NARENAS > NARENAS:  # pragma: no cover
    raise ImportError(f"Unsupported GxB_NARENAS: {_lib.GxB_NARENAS}")

ctypedef void *(*malloc_t)(size_t) noexcept nogil
ctypedef void *(*calloc_t)(size_t, size_t) noexcept nogil
ctypedef void *(*realloc_t)(void *, size_t) noexcept nogil
ctypedef void (*free_t)(void *) noexcept nogil

ctypedef struct Arena:
    malloc_t malloc
    calloc_t calloc  # may be NULL
    realloc_t realloc  # may be NULL
    free_t free

# The allocator of each GraphBLAS arena, and a NumPy memory handler that uses it
cdef Arena arenas[NARENAS]
cdef PyDataMem_Handler arena_handlers[NARENAS]
# The NumPy memory handler (a PyCapsule) for the memory of each arena, made on first use.
# It is NumPy's default handler if the arena uses NumPy's allocator.
cdef list _handlers = [None] * NARENAS
_handlers_lock = threading.Lock()


cdef uintptr_t address(object cdata) except? 0:
    # The address in a cffi pointer.  Cython isn't compiled against GraphBLAS, so this is how
    # it gets at what cffi's `lib` has: cast the result to the C type and use it directly.
    return int(_ffi.cast("uintptr_t", cdata))


# `GrB_Global_get_INT32(GrB_GLOBAL, ...)`, called directly rather than through cffi because
# it is on the path of every function below
cdef global_get_int32_t global_get_int32 = <global_get_int32_t>address(
    _ffi.addressof(_lib, "GrB_Global_get_INT32")
)
cdef void *GrB_GLOBAL = <void *>address(_lib.GrB_GLOBAL)
cdef int GxB_ARENA_DATA = _lib.GxB_ARENA_DATA


cdef void *arena_malloc(void *ctx, size_t size) noexcept nogil:
    return (<Arena *>ctx).malloc(size)


cdef void *arena_calloc(void *ctx, size_t nelem, size_t elsize) noexcept nogil:
    cdef Arena *arena = <Arena *>ctx
    cdef void *ptr
    if arena.calloc != NULL:
        return arena.calloc(nelem, elsize)
    if elsize != 0 and nelem > SIZE_MAX // elsize:
        return NULL
    ptr = arena.malloc(nelem * elsize)
    if ptr != NULL:
        memset(ptr, 0, nelem * elsize)
    return ptr


cdef void *arena_realloc(void *ctx, void *ptr, size_t new_size) noexcept nogil:
    cdef Arena *arena = <Arena *>ctx
    if arena.realloc == NULL:  # optional for GraphBLAS arenas; NumPy raises MemoryError
        return NULL
    return arena.realloc(ptr, new_size)


cdef void arena_free(void *ctx, void *ptr, size_t size) noexcept nogil:
    (<Arena *>ctx).free(ptr)


cdef int _resolve_arena(object arena) except -1:
    cdef int32_t k
    cdef int info
    if arena is None:
        info = global_get_int32(GrB_GLOBAL, &k, GxB_ARENA_DATA)
        if info != 0:  # GrB_SUCCESS
            raise RuntimeError(
                f"Unable to get GraphBLAS's data arena (info={info}); is GraphBLAS initialized?"
            )
        return k
    index = operator.index(arena)
    if not 0 <= index < NARENAS:
        raise ValueError(f"arena must be in range({NARENAS}); got {arena!r}")
    return index


cdef object _handler(int k):
    # The NumPy memory handler for memory from arena `k`
    handler = _handlers[k]
    if handler is None:
        with _handlers_lock:
            handler = _handlers[k]
            if handler is None:
                handler = _handlers[k] = _new_handler(k)
    return handler


cdef object _new_handler(int k):
    cdef uintptr_t funcs[4]
    flag = _ffi.new("int*")
    info = _lib.GxB_arena_initialized(flag, k)
    if info != _lib.GrB_SUCCESS:
        raise RuntimeError(
            f"Unable to query GraphBLAS arena {k} (info={info}); is GraphBLAS initialized?"
        )
    if not flag[0]:
        raise ValueError(f"GraphBLAS arena {k} has not been initialized (see GxB_arena_init)")
    func = _ffi.new("void**")
    for i, field in enumerate(
        [_lib.GxB_ARENA_MALLOC, _lib.GxB_ARENA_CALLOC, _lib.GxB_ARENA_REALLOC, _lib.GxB_ARENA_FREE]
    ):
        info = _lib.GrB_Global_get_VOID(_lib.GrB_GLOBAL, func, field + k)
        if info != _lib.GrB_SUCCESS:
            raise RuntimeError(f"Unable to get the allocator of GraphBLAS arena {k} (info={info})")
        funcs[i] = address(func[0])
    arenas[k].malloc = <malloc_t>funcs[0]
    arenas[k].calloc = <calloc_t>funcs[1]
    arenas[k].realloc = <realloc_t>funcs[2]
    arenas[k].free = <free_t>funcs[3]
    if funcs[3] == <uintptr_t><void *>PyDataMem_FREE:
        # NumPy's allocator (memory_manager="numpy"): use NumPy's own handler, so these
        # arrays are like any other NumPy array (including for `can_unclaim_buffer`).
        return <object>PyDataMem_DefaultHandler
    snprintf(arena_handlers[k].name, sizeof(arena_handlers[k].name), "suitesparse_graphblas_arena%d", k)
    arena_handlers[k].version = 1
    arena_handlers[k].allocator.ctx = &arenas[k]
    arena_handlers[k].allocator.malloc = arena_malloc
    arena_handlers[k].allocator.calloc = arena_calloc
    arena_handlers[k].allocator.realloc = arena_realloc
    arena_handlers[k].allocator.free = arena_free
    return PyCapsule_New(&arena_handlers[k], "mem_handler", NULL)


cdef str _handler_name(object handler):
    return (<PyDataMem_Handler *>PyCapsule_GetPointer(handler, "mem_handler")).name.decode()


cdef inline void own_data(ndarray array, object handler):
    # Make `array`, which was wrapped around existing data, own (and eventually free) that
    # data with `handler`.  Without a handler, NumPy frees with libc `free`, which is wrong
    # unless the data is from libc `malloc`.
    cdef PyArrayObject_fields *fields = <PyArrayObject_fields *><void *>array
    if fields.mem_handler == NULL:  # else NumPy allocated the data (it was given NULL)
        Py_INCREF(handler)
        fields.mem_handler = <PyObject *>handler
    PyArray_ENABLEFLAGS(array, NPY_ARRAY_OWNDATA)


cdef str _why_not_ownable(object array, object handler):
    # Why GraphBLAS can't take ownership of the data of `array` (to free with `handler`),
    # or None if it can.
    cdef ndarray arr
    cdef PyObject *mem_handler
    if not isinstance(array, ndarray):
        return f"it is a {type(array).__name__}, not a NumPy array"
    arr = array
    if not PyArray_CHKFLAGS(arr, NPY_ARRAY_OWNDATA):
        return "it does not own its data (e.g., it is a view, or is already being given)"
    if not PyArray_CHKFLAGS(arr, NPY_ARRAY_WRITEABLE):
        return "it is read-only"
    if arr.dtype.hasobject:
        return "it holds Python objects"
    mem_handler = (<PyArrayObject_fields *><void *>arr).mem_handler
    if mem_handler == <PyObject *>handler:
        return None
    if mem_handler == NULL:
        return "it has no NumPy memory handler, so its data came from an unknown allocator"
    return (
        f"its data came from NumPy memory handler {_handler_name(<object>mem_handler)!r}, "
        f"but GraphBLAS will free it as {_handler_name(handler)!r}"
    )


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


cpdef ndarray claim_buffer(object ffi, object cdata, size_t size, dtype_t dtype, object arena=None):
    """Return a 1-d array that owns ``size`` elements of GraphBLAS-allocated memory at ``cdata``.

    ``arena`` is the GraphBLAS arena that allocated the memory (default: the global data
    arena); NumPy will free the memory with that arena's allocator.
    """
    cdef:
        npy_intp dims = size
        uintptr_t ptr = int(ffi.cast("uintptr_t", cdata))
        ndarray array
    handler = _handler(_resolve_arena(arena))
    if ptr == 0 and size != 0:
        raise ValueError(f"Unable to claim {size} elements from a NULL pointer")
    Py_INCREF(dtype)
    array = PyArray_NewFromDescr(
        ndarray, dtype, 1, &dims, NULL, <void*>ptr, NPY_ARRAY_WRITEABLE, <object>NULL
    )
    own_data(array, handler)
    return array


cpdef ndarray claim_buffer_2d(
    object ffi,
    object cdata,
    size_t cdata_size,
    size_t nrows,
    size_t ncols,
    dtype_t dtype,
    bint is_c_order,
    object arena=None,
):
    """Like ``claim_buffer``, but return a 2-d ``nrows`` by ``ncols`` array."""
    cdef:
        size_t size = nrows * ncols
        ndarray array
        uintptr_t ptr
        npy_intp dims[2]
        int flags = NPY_ARRAY_WRITEABLE
    if cdata_size == size:
        handler = _handler(_resolve_arena(arena))
        ptr = int(ffi.cast("uintptr_t", cdata))
        if ptr == 0 and size != 0:
            raise ValueError(f"Unable to claim {size} elements from a NULL pointer")
        dims[0] = nrows
        dims[1] = ncols
        if not is_c_order:
            flags |= NPY_ARRAY_F_CONTIGUOUS
        Py_INCREF(dtype)
        array = PyArray_NewFromDescr(
            ndarray, dtype, 2, dims, NULL, <void*>ptr, flags, <object>NULL
        )
        own_data(array, handler)
    elif cdata_size > size:  # pragma: no cover
        array = claim_buffer(ffi, cdata, cdata_size, dtype, arena)
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


cpdef bint can_unclaim_buffer(object array, object arena=None) except -1:
    """Whether GraphBLAS may take ownership of the data of ``array``.

    True only if ``array`` owns its data, is writeable, does not hold Python objects, and
    its data came from the allocator of the GraphBLAS arena (default: the global data
    arena) that will free it.  So views never qualify, nor do arrays from a custom NumPy
    memory handler, nor (with ``memory_manager="c"``) arrays NumPy allocated.  Arrays from
    ``claim_buffer``, ``empty``, and (with ``memory_manager="numpy"``) NumPy's default
    allocator do.  Prefer ``give_buffer``, which also copies when this is False.

    This cannot see other references to the data: GraphBLAS may free or reallocate it at
    any time after the handover, so no view of ``array`` (nor memoryview or cffi pointer
    into it) may outlive the handover.
    """
    if not isinstance(array, ndarray) or not PyArray_CHKFLAGS(array, NPY_ARRAY_OWNDATA):
        return False
    return _why_not_ownable(array, _handler(_resolve_arena(arena))) is None


cpdef unclaim_buffer(ndarray array, object arena=None):
    """Make ``array`` a read-only view of its data, which GraphBLAS has taken ownership of.

    Call this *after* a GraphBLAS function took ownership, and only for an array that
    passed ``can_unclaim_buffer`` beforehand.  Prefer ``give_buffer``, which does both.

    Raises ValueError if GraphBLAS can't own the data (see ``can_unclaim_buffer``).  Since
    GraphBLAS already has it and will free it, ``array`` stops owning the data even then,
    so that NumPy does not free it too.
    """
    why_not = _why_not_ownable(array, _handler(_resolve_arena(arena)))
    if why_not is None:
        PyArray_CLEARFLAGS(array, NPY_ARRAY_OWNDATA | NPY_ARRAY_WRITEABLE)
        return
    PyArray_CLEARFLAGS(array, NPY_ARRAY_OWNDATA)
    raise ValueError(
        f"GraphBLAS cannot safely own the data of this array: {why_not}.  Check "
        "`can_unclaim_buffer` before handing data to GraphBLAS, or use `give_buffer`."
    )


def empty(shape, dtype=float, order="C", arena=None):
    """Return a new, uninitialized array whose data GraphBLAS can take ownership of.

    Like ``numpy.empty``, but allocated by the allocator of the GraphBLAS ``arena``
    (default: the global data arena) whatever the ``memory_manager`` or NumPy memory
    handler in use, so ``can_unclaim_buffer`` is True for it.
    """
    cdef:
        int k = _resolve_arena(arena)
        void *ptr
        ndarray array
        ndarray dims
    handler = _handler(k)
    dtype = np.dtype(dtype)
    if dtype.hasobject:
        raise TypeError("GraphBLAS cannot own arrays that hold Python objects")
    if order not in {"C", "F"}:
        raise ValueError(f"order must be 'C' or 'F'; got {order!r}")
    try:
        shape = (operator.index(shape),)
    except TypeError:
        try:
            shape = tuple([operator.index(dim) for dim in shape])
        except TypeError:
            raise TypeError(f"shape must be an int or a sequence of ints; got {shape!r}") from None
    nbytes = dtype.itemsize
    for dim in shape:
        if dim < 0:
            raise ValueError("negative dimensions are not allowed")
        nbytes *= dim
    if nbytes > PY_SSIZE_T_MAX:
        raise ValueError("array is too big")
    dims = np.array(shape, dtype=np.intp)
    ptr = arenas[k].malloc(nbytes or 1)  # NumPy never allocates 0 bytes either
    if ptr == NULL:
        raise MemoryError(f"Unable to allocate {nbytes} bytes")
    Py_INCREF(dtype)
    try:
        array = PyArray_NewFromDescr(
            ndarray,
            dtype,
            len(shape),
            <npy_intp *>PyArray_DATA(dims),
            NULL,
            ptr,
            NPY_ARRAY_WRITEABLE | (NPY_ARRAY_F_CONTIGUOUS if order == "F" else 0),
            <object>NULL,
        )
    except BaseException:
        arenas[k].free(ptr)
        raise
    own_data(array, handler)
    return array


@contextmanager
def give_buffer(array, ctype="void*", *, copy=None, arena=None):
    """Hand the data of ``array`` to a GraphBLAS function that takes ownership of it.

    Yields ``(ptr, nbytes, arena)``: ``ptr`` is a new ``ctype *`` that points to the data,
    to pass to, e.g., ``GxB_Matrix_pack_CSR`` (with ``nbytes`` as the size) or to
    ``GxB_Vector_load`` (with handling ``GrB_DEFAULT + arena``).  GraphBLAS sets ``*ptr``
    to NULL when it takes ownership; then, on exit, the array handed over becomes a
    read-only view of data it no longer owns.  Otherwise (e.g., the call failed) nothing
    changes.

    copy : bool, optional
        None (default): hand over the data of ``array`` itself if GraphBLAS can own it
        (see ``can_unclaim_buffer``), else a copy.  True: always hand over a copy, leaving
        ``array`` unchanged.  False: never copy; raise ValueError if GraphBLAS can't own it.
    arena : int, optional
        The GraphBLAS arena that will own the data (default: the global data arena).

    An array is handed over only once: to give the same array as two arguments of one
    call, nest two ``give_buffer`` and the second gets a copy.  Giving one array from two
    threads at once is a race, like any other change to an array.

    Examples
    --------
    >>> import numpy as np
    >>> from suitesparse_graphblas import check_status, ffi, lib, vector
    >>> from suitesparse_graphblas.utils import give_buffer
    >>> v = vector.vector_new(lib.GrB_INT64)
    >>> values = np.arange(3, dtype=np.int64)
    >>> with give_buffer(values) as (X, nbytes, arena):
    ...     info = lib.GxB_Vector_load(
    ...         v[0], X, lib.GrB_INT64, values.size, nbytes, lib.GrB_DEFAULT + arena, ffi.NULL
    ...     )
    ...     check_status(v, info)
    >>> vector.vector_nvals(v), values.flags.owndata
    (3, False)
    """
    cdef int k = _resolve_arena(arena)
    cdef ndarray buf
    if copy is not None and not isinstance(copy, bool):
        raise TypeError(f"copy must be None, True, or False; got {copy!r}")
    handler = _handler(k)
    array = np.asarray(array)
    if array.dtype.hasobject:
        raise TypeError("GraphBLAS cannot own arrays that hold Python objects")
    why_not = None if copy else _why_not_ownable(array, handler)
    if copy is False and why_not is not None:
        raise ValueError(
            f"GraphBLAS cannot take ownership of the data of this array without copying: {why_not}"
        )
    if copy or why_not is not None:
        order = "F" if array.flags.f_contiguous and not array.flags.c_contiguous else "C"
        buf = empty(array.shape, array.dtype, order, k)
        np.copyto(buf, array, casting="no")
    else:
        buf = array
    ptr = _ffi.new(f"{ctype}*", _ffi.cast(ctype, <uintptr_t>PyArray_DATA(buf)))
    # `buf` stops owning its data for now, so that giving the same array again copies it
    PyArray_CLEARFLAGS(buf, NPY_ARRAY_OWNDATA)
    try:
        yield ptr, buf.nbytes, k
    finally:
        if ptr[0] == _ffi.NULL:  # GraphBLAS owns the data now
            PyArray_CLEARFLAGS(buf, NPY_ARRAY_WRITEABLE)
        else:
            PyArray_ENABLEFLAGS(buf, NPY_ARRAY_OWNDATA)
