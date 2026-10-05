import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from suitesparse_graphblas import lib, matrix
from suitesparse_graphblas.utils import can_unclaim_buffer

# Move memory between owners in every supported direction, then free all of it.
# `binary` is not tested on Windows; see test_io.py.
SCRIPT = textwrap.dedent("""
    import gc
    import sys

    import numpy as np

    import suitesparse_graphblas as ssgb
    from suitesparse_graphblas import check_status, ffi, lib, matrix, vector
    from suitesparse_graphblas.utils import can_unclaim_buffer, unclaim_buffer

    memory_manager = sys.argv[1]
    ssgb.initialize(memory_manager=memory_manager)
    A = matrix.matrix_new(lib.GrB_INT64, 3, 3)
    matrix.set_int64(A, 7, 1, 2)

    # GraphBLAS -> NumPy (`claim_buffer`): NumPy frees what GraphBLAS allocated
    data = matrix.serialize(A)
    B = matrix.deserialize(data)
    assert matrix.matrix_nvals(B) == 1
    del data

    # NumPy -> GraphBLAS (`unclaim_buffer`): GraphBLAS frees what it may take ownership of
    numpy_allocated = np.ones(4096, np.uint8)
    assert can_unclaim_buffer(numpy_allocated) == (memory_manager == "numpy")
    for array in matrix.serialize(A), numpy_allocated:
        if can_unclaim_buffer(array):
            v = vector.vector_new(lib.GrB_UINT8)
            x = ffi.new("void**", ffi.cast("void*", array.ctypes.data))
            info = lib.GxB_Vector_load(
                v[0], x, lib.GrB_UINT8, array.size, array.nbytes, lib.GrB_DEFAULT, ffi.NULL
            )
            check_status(v, info)
            unclaim_buffer(array)
            assert not can_unclaim_buffer(array)
            assert vector.vector_nvals(v) == array.size
            del v

    # Python -> GraphBLAS (`binread`): GraphBLAS frees what `graphblas_malloc` allocated
    if sys.platform != "win32":
        from suitesparse_graphblas.io import binary

        binary.binwrite(A, sys.argv[2])
        C = binary.binread(sys.argv[2])
        assert matrix.matrix_nvals(C) == 1
        del C
    del A, B
    gc.collect()
    """)


@pytest.mark.parametrize(
    "memory_manager",
    [
        "numpy",
        # The C runtime of libgraphblas need not be the one NumPy frees with on Windows
        pytest.param("c", marks=pytest.mark.skipif(sys.platform == "win32", reason="untested")),
    ],
)
def test_allocators_match(memory_manager, tmp_path):
    """Memory must be freed by the allocator that allocated it, whoever ends up owning it.

    A mismatch is silent wherever both allocators happen to be libc's, which used to be
    everywhere.  They differ on free-threaded Python 3.15, and Python's debug hooks make
    them differ on any Python (given NumPy >=2.5), so this test doesn't depend on which
    Python runs it.  It needs its own process: a mismatch aborts the interpreter, and
    GraphBLAS can only be initialized once.
    """
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT, memory_manager, str(tmp_path / "matrix.binfile")],
        env={**os.environ, "PYTHONMALLOC": "debug"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr


def test_can_unclaim_buffer():
    array = np.arange(10)
    assert can_unclaim_buffer(array)
    assert can_unclaim_buffer(matrix.serialize(matrix.matrix_new(lib.GrB_BOOL, 2, 2)))
    # GraphBLAS can only take ownership from an owner
    assert not can_unclaim_buffer(array[:5])
    assert not can_unclaim_buffer(np.frombuffer(b"1234", np.uint8))
