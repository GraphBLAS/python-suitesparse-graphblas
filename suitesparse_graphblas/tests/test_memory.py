import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from suitesparse_graphblas import check_status, exceptions, ffi, lib, matrix, vector
from suitesparse_graphblas.utils import (
    can_unclaim_buffer,
    claim_buffer,
    claim_buffer_2d,
    empty,
    give_buffer,
    unclaim_buffer,
)

# Move memory between GraphBLAS and NumPy every supported way, then free all of it.
# `binary` is not tested on Windows; see test_io.py.
SCRIPT = textwrap.dedent("""
    import gc
    import sys
    import threading
    from pathlib import Path

    import numpy as np

    import suitesparse_graphblas as ssgb
    from suitesparse_graphblas import check_status, ffi, lib, matrix, vector
    from suitesparse_graphblas.api import context
    from suitesparse_graphblas.utils import can_unclaim_buffer, claim_buffer, empty, give_buffer

    memory_manager, tmpdir = sys.argv[1], Path(sys.argv[2])
    ssgb.initialize(memory_manager=memory_manager)
    # NumPy warns (here, raises) when it frees an array without a memory handler (see the
    # subprocess environment), but from deallocation, where it can only report it
    unraisable = []
    sys.unraisablehook = unraisable.append
    int64 = np.dtype(np.int64)
    # NumPy keeps freed blocks under 1024 bytes for reuse, which would hide a wrong free
    N = 1000


    def load(array, **kwargs):
        # NumPy -> GraphBLAS, the way to use `give_buffer`
        v = vector.vector_new(lib.GrB_INT64)
        with give_buffer(array, **kwargs) as (X, nbytes, arena):
            info = lib.GxB_Vector_load(
                v[0], X, lib.GrB_INT64, array.size, nbytes, lib.GrB_DEFAULT + arena, ffi.NULL
            )
            check_status(v, info)
        assert vector.vector_nvals(v) == array.size
        return v


    def unload(v):
        # GraphBLAS -> NumPy, the way to use `claim_buffer`
        X = ffi.new("void**")
        n = ffi.new("uint64_t*")
        handling = ffi.new("int*")
        info = lib.GxB_Vector_unload(
            v[0], X, ffi.new("GrB_Type*"), n, ffi.new("uint64_t*"), handling, ffi.NULL
        )
        check_status(v, info)
        assert handling[0] < lib.GxB_IS_READONLY  # else the data is not GraphBLAS's to give
        return claim_buffer(ffi, X[0], n[0], int64, arena=handling[0] - lib.GrB_DEFAULT)


    def roundtrip(**kwargs):
        # NumPy -> GraphBLAS -> NumPy: without a copy (if "numpy"), with one, and from `empty`
        ones = empty(N, int64, **kwargs)
        ones[:] = 1
        for array in [np.arange(N, dtype=int64), np.arange(2 * N, dtype=int64)[::2], ones]:
            total = array.sum()
            assert unload(load(array, **kwargs)).sum() == total
        v = load(np.arange(N, dtype=int64), **kwargs)
        data = vector.serialize(v, lib.GxB_COMPRESSION_NONE)
        assert data.nbytes > N * 8
        assert vector.vector_nvals(vector.deserialize(data)) == N


    def main():
        array = np.arange(N, dtype=int64)
        assert can_unclaim_buffer(array) == (memory_manager == "numpy")
        load(array)
        assert array.flags.owndata == (memory_manager != "numpy")  # was it copied?
        roundtrip()

        # GraphBLAS has nothing to give for an empty array, so NumPy allocates it
        assert claim_buffer(ffi, ffi.NULL, 0, int64).shape == (0,)

        # The same array as two arguments: GraphBLAS must get two buffers
        array = np.arange(N, dtype=np.uint64)
        A = matrix.matrix_new(lib.GrB_UINT64, 1, N)
        with (
            give_buffer(np.array([0, N], np.uint64), "GrB_Index*") as (Ap, Ap_size, _),
            give_buffer(array, "GrB_Index*") as (Aj, Aj_size, _),
            give_buffer(array) as (Ax, Ax_size, _),
        ):
            info = lib.GxB_Matrix_pack_CSR(
                A[0], Ap, Aj, Ax, Ap_size, Aj_size, Ax_size, False, False, ffi.NULL
            )
            check_status(A, info)
        assert matrix.matrix_nvals(A) == N

        # Many threads at once (free-threaded Python runs them in parallel)
        def roundtrips():
            for _ in range(20):
                roundtrip()

        threads = [threading.Thread(target=roundtrips) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if sys.platform == "win32":
            return
        from suitesparse_graphblas.io import binary

        # `binread` hands its buffers to GraphBLAS; an iso matrix stores a single value, and
        # making it non-iso later must not write past it
        for sparsity in [lib.GxB_HYPERSPARSE, lib.GxB_SPARSE, lib.GxB_BITMAP, lib.GxB_FULL]:
            A = matrix.matrix_new(lib.GrB_INT64, 30, 30)
            info = lib.GrB_Matrix_assign_INT64(
                A[0], ffi.NULL, ffi.NULL, 7, lib.GrB_ALL, 30, lib.GrB_ALL, 30, ffi.NULL
            )
            check_status(A, info)
            matrix.matrix_set_sparsity_control(A, sparsity)
            binary.binwrite(A, tmpdir / "iso.binfile")
            B = binary.binread(tmpdir / "iso.binfile")
            matrix.set_int64(B, 99, 0, 0)
            check_status(B, lib.GrB_Matrix_wait(B[0], lib.GrB_MATERIALIZE))
            assert matrix.matrix_nvals(B) == 900

        # Another arena, using libc's allocator.  The last one, since GraphBLAS means to give
        # CUDA an arena of its own (the first 10.4.0 betas reserved arena 1).  It needs a realloc:
        # without one, GraphBLAS (10.4.0 to 10.5.1) frees the arena's memory with arena 0's
        # free when it grows a buffer.
        k = lib.GxB_NARENAS - 1
        import cffi

        std = cffi.FFI()
        std.cdef("void *malloc(size_t); void *calloc(size_t, size_t);")
        std.cdef("void *realloc(void *, size_t); void free(void *);")
        libc = std.dlopen(None)
        funcs = [
            ffi.cast(sig, int(std.cast("uintptr_t", std.addressof(libc, name))))
            for name, sig in [
                ("malloc", "void *(*)(size_t)"),
                ("calloc", "void *(*)(size_t, size_t)"),
                ("realloc", "void *(*)(void *, size_t)"),
                ("free", "void (*)(void *)"),
            ]
        ]
        assert lib.GxB_arena_init(k, *funcs) == lib.GrB_SUCCESS
        array = empty(N, int64, arena=k)
        assert can_unclaim_buffer(array, arena=k) and not can_unclaim_buffer(array, arena=0)
        assert not can_unclaim_buffer(np.arange(N), arena=k)
        roundtrip(arena=k)

        def others():
            # GraphBLAS -> NumPy and NumPy -> GraphBLAS in functions that have no `arena`
            roundtrip()
            binary.binwrite(A, tmpdir / "arena.binfile")
            assert matrix.matrix_nvals(binary.binread(tmpdir / "arena.binfile")) == 900

        # ...as the global data arena, which is then the default
        assert lib.GrB_Global_set_INT32(lib.GrB_GLOBAL, k, lib.GxB_ARENA_DATA) == lib.GrB_SUCCESS
        assert can_unclaim_buffer(empty(N)) and not can_unclaim_buffer(np.arange(N))
        others()
        assert lib.GrB_Global_set_INT32(lib.GrB_GLOBAL, 0, lib.GxB_ARENA_DATA) == lib.GrB_SUCCESS

        # ...and as the data arena of an engaged Context, which GraphBLAS doesn't reveal
        ctx = context.context_new()
        check_status(ctx, lib.GxB_Context_set_INT(ctx[0], k, lib.GxB_ARENA_DATA))
        context.context_engage(ctx)
        others()
        context.context_disengage(ctx)


    main()
    gc.collect()
    assert not unraisable, [str(u.exc_value) for u in unraisable]
    """)


@pytest.mark.skipif(sys.platform == "emscripten", reason="needs a subprocess")
@pytest.mark.parametrize("memory_manager", ["numpy", "c"])
def test_allocators_match(memory_manager, tmp_path):
    """Memory must be freed by the allocator that allocated it, whoever ends up owning it.

    A mismatch is silent wherever both allocators happen to be libc's, which used to be
    everywhere.  They differ on free-threaded Python 3.15, and Python's debug hooks make
    them differ on any Python (given NumPy >=2.5), so this test doesn't depend on which
    Python runs it.  It needs its own process: a mismatch aborts the interpreter, and
    GraphBLAS can only be initialized once.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::RuntimeWarning",
            "-c",
            SCRIPT,
            memory_manager,
            str(tmp_path),
        ],
        env={**os.environ, "PYTHONMALLOC": "debug", "NUMPY_WARN_IF_NO_MEM_POLICY": "1"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr


def load(array, **kwargs):
    v = vector.vector_new(lib.GrB_INT64)
    with give_buffer(array, **kwargs) as (X, nbytes, arena):
        info = lib.GxB_Vector_load(
            v[0], X, lib.GrB_INT64, array.size, nbytes, lib.GrB_DEFAULT + arena, ffi.NULL
        )
        check_status(v, info)
    return v


def test_claim_buffer():
    int64 = np.dtype(np.int64)
    assert claim_buffer(ffi, ffi.NULL, 0, int64).shape == (0,)
    assert claim_buffer_2d(ffi, ffi.NULL, 0, 0, 5, int64, True).shape == (0, 5)
    with pytest.raises(ValueError, match="NULL pointer"):
        claim_buffer(ffi, ffi.NULL, 3, int64)
    with pytest.raises(ValueError, match="NULL pointer"):
        claim_buffer_2d(ffi, ffi.NULL, 6, 2, 3, int64, True)


def test_can_unclaim_buffer():
    array = np.arange(10)
    assert can_unclaim_buffer(array)
    assert can_unclaim_buffer(matrix.serialize(matrix.matrix_new(lib.GrB_BOOL, 2, 2)))
    assert can_unclaim_buffer(empty(3))
    # GraphBLAS can only take ownership from an owner, and only if it may change the data
    assert not can_unclaim_buffer(array[:5])
    assert not can_unclaim_buffer(np.frombuffer(b"1234", np.uint8))
    readonly = np.arange(10)
    readonly.flags.writeable = False
    assert not can_unclaim_buffer(readonly)
    assert not can_unclaim_buffer(np.array([None, 1]))
    assert not can_unclaim_buffer([1, 2, 3])
    for arena in [8, -1, 2**40]:
        with pytest.raises(ValueError, match="range"):
            can_unclaim_buffer(array, arena=arena)
    with pytest.raises(ValueError, match="arena 5 has not been initialized"):
        can_unclaim_buffer(array, arena=5)


def test_unclaim_buffer():
    def handover(array):
        # The low-level way: hand over (after checking `can_unclaim_buffer`), then unclaim
        v = vector.vector_new(lib.GrB_INT64)
        X = ffi.new("void**", ffi.cast("void*", array.ctypes.data))
        info = lib.GxB_Vector_load(
            v[0], X, lib.GrB_INT64, array.size, array.nbytes, lib.GrB_DEFAULT, ffi.NULL
        )
        check_status(v, info)
        return v

    array = np.arange(10, dtype=np.int64)
    assert can_unclaim_buffer(array)
    v = handover(array)
    unclaim_buffer(array)
    assert not array.flags.owndata
    assert not array.flags.writeable
    assert vector.vector_nvals(v) == 10

    # GraphBLAS should not have been given this, but it now has it, so NumPy must let go
    array = np.arange(10, dtype=np.int64)
    array.flags.writeable = False
    assert not can_unclaim_buffer(array)
    v = handover(array)
    with pytest.raises(ValueError, match="it is read-only.  Check `can_unclaim_buffer`"):
        unclaim_buffer(array)
    assert not array.flags.owndata
    assert vector.vector_nvals(v) == 10

    for array in [np.arange(10)[::2], np.frombuffer(bytearray(8), np.uint8)]:
        with pytest.raises(ValueError, match="does not own its data"):
            unclaim_buffer(array)
        assert array.flags.writeable  # unchanged


def test_empty():
    a = empty((2, 3), np.int32)
    assert a.shape == (2, 3)
    assert a.dtype == np.int32
    assert a.flags.c_contiguous
    assert a.flags.owndata
    a[...] = 1
    assert a.sum() == 6
    f = empty([2, 3], order="F")
    assert f.dtype == np.float64
    assert f.flags.f_contiguous
    assert not f.flags.c_contiguous
    assert empty(4).shape == empty(np.int64(4)).shape == (4,)
    assert empty(()).shape == ()
    assert empty((0, 5)).size == 0
    with pytest.raises(ValueError, match="negative"):
        empty(-1)
    with pytest.raises(TypeError, match="sequence of ints"):
        empty(1.5)
    with pytest.raises(TypeError, match="Python objects"):
        empty(3, object)
    with pytest.raises(ValueError, match="order"):
        empty(3, order="A")


def test_give_buffer():
    # Zero-copy: the array no longer owns the data GraphBLAS took
    array = np.arange(5, dtype=np.int64)
    v = load(array)
    assert not array.flags.owndata
    assert not array.flags.writeable
    x = ffi.new("int64_t*")
    check_status(v, lib.GrB_Vector_extractElement_INT64(x, v[0], 4))
    assert x[0] == 4

    # A copy, so the array is unchanged
    for array, kwargs in [
        (np.arange(10, dtype=np.int64)[::2], {}),
        (np.arange(5, dtype=np.int64), {"copy": True}),
    ]:
        v = load(array, **kwargs)
        assert array.flags.writeable
        check_status(v, lib.GrB_Vector_extractElement_INT64(x, v[0], 4))
        assert x[0] == array[4]
    with pytest.raises(ValueError, match="without copying: it does not own its data"):
        load(np.arange(10)[::2], copy=False)

    # GraphBLAS didn't take ownership, so the array keeps it
    array = np.arange(5, dtype=np.int64)
    v = vector.vector_new(lib.GrB_INT64)
    with pytest.raises(exceptions.InvalidValue):
        with give_buffer(array) as (X, nbytes, arena):
            assert not array.flags.owndata  # so that it can't be given twice
            info = lib.GxB_Vector_load(  # claims more entries than fit
                v[0], X, lib.GrB_INT64, array.size + 1, nbytes, lib.GrB_DEFAULT, ffi.NULL
            )
            check_status(v, info)
    assert array.flags.owndata
    assert array.flags.writeable

    # The same array twice: the second is a copy, or an error if it may not be
    array = np.arange(5, dtype=np.int64)
    with give_buffer(array) as (X, _, _), give_buffer(array) as (Y, _, _):
        assert X[0] != Y[0]
    with pytest.raises(ValueError, match="already being given"):
        with give_buffer(array), give_buffer(array, copy=False):
            pass
    assert array.flags.owndata

    with pytest.raises(TypeError, match="Python objects"):
        load(np.array([None, 1]))
    with pytest.raises(TypeError, match="copy"):
        load(np.arange(5), copy="yes")
