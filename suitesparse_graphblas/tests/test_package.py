import pytest

import suitesparse_graphblas
from suitesparse_graphblas import ffi, lib


def test_matrix_existence():
    assert hasattr(lib, "GrB_Matrix_new")


def test_version():
    # Example dev version: 9.4.5.0+2.g5590dba8.dirty
    # Example reslease version: 9.4.5.0
    version = suitesparse_graphblas.__version__
    version = [int(x) for x in version.split("+")[0].split(".")]
    assert version > [9, 4, 4, 0]


def test_openmp():
    # GraphBLAS only warns at configure time when OpenMP is missing and still
    # builds a working, but serial, library, so a serial build is invisible
    # from the outside: conda-forge's graphblas 10.5.0 shipped that way on
    # osx-arm64. Wheels build GraphBLAS from source in suitesparse.sh, where
    # macOS has to point CMake at Homebrew's libomp by hand, so the same slip
    # is possible here.
    val_ptr = ffi.new("int32_t*")
    info = lib.GrB_Global_get_INT32(lib.GrB_GLOBAL, val_ptr, lib.GxB_LIBRARY_OPENMP)
    assert info == lib.GrB_SUCCESS
    assert val_ptr[0], "libgraphblas was built without OpenMP"


def test_libgraphblas_version_matches_the_loaded_library():
    """The GraphBLAS this extension was compiled against must be the one it loads.

    ``libgraphblas_version`` reads cffi ``#define`` constants, which are
    resolved when the C extension is compiled against a particular
    GraphBLAS.h. The values below come from the shared library actually loaded
    at runtime. When those disagree, every cffi call is reading a struct laid
    out by a different build -- a mismatch that surfaces later as an
    unexplained ``GrB_OUT_OF_MEMORY`` or a crash, never as an import error.
    """
    val_ptr = ffi.new("int32_t*")
    runtime = []
    for field in (
        lib.GrB_LIBRARY_VER_MAJOR,
        lib.GrB_LIBRARY_VER_MINOR,
        lib.GrB_LIBRARY_VER_PATCH,
    ):
        assert lib.GrB_Global_get_INT32(lib.GrB_GLOBAL, val_ptr, field) == lib.GrB_SUCCESS
        runtime.append(val_ptr[0])
    assert tuple(runtime) == suitesparse_graphblas.libgraphblas_version


def test_version_tracks_libgraphblas():
    """A released version's first three parts are the SuiteSparse:GraphBLAS version.

    That convention is what lets a caller infer the library version from the
    package version, and it only holds for a release: between releases the
    version reports the previous tag, which is why ``libgraphblas_version``
    exists and why anything gating on a library feature should read that
    instead. Skipped on a development build rather than asserted, since
    disagreeing there is the expected state, not a defect.
    """
    version = suitesparse_graphblas.__version__
    if "+" in version:
        pytest.skip(f"development build between releases: {version}")
    parts = tuple(int(x) for x in version.split(".")[:3])
    assert parts == suitesparse_graphblas.libgraphblas_version
