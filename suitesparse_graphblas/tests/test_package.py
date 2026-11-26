import os
import sys

import pytest

import suitesparse_graphblas
from suitesparse_graphblas import (
    bool_types,
    check_status,
    complex_types,
    ffi,
    grb_types,
    integer_types,
    lib,
    libget,
    matrix,
    real_types,
    signed_integer_types,
    supports_complex,
    unsigned_integer_types,
    vararg,
)
from suitesparse_graphblas.api import grb_type


def test_matrix_existence():
    assert hasattr(lib, "GrB_Matrix_new")


def test_version():
    # Example dev version: 9.4.5.0+2.g5590dba8.dirty
    # Example reslease version: 9.4.5.0
    version = suitesparse_graphblas.__version__
    version = [int(x) for x in version.split("+")[0].split(".")]
    assert version > [9, 4, 4, 0]


@pytest.mark.skipif(
    sys.platform == "emscripten", reason="WebAssembly (Pyodide) has no threads, so no OpenMP"
)
def test_openmp():
    # GraphBLAS only warns at configure time when OpenMP is missing and still
    # builds a working, but serial, library, so a serial build is invisible
    # from the outside: conda-forge's graphblas 10.5.0 shipped that way on
    # osx-arm64. Wheels build GraphBLAS from source in suitesparse.sh, where
    # macOS has to point CMake at libomp by hand, so the same slip
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


def test_libget():
    assert libget("GrB_INP0") == lib.GrB_INP0
    assert libget("GrB_FP64") == lib.GrB_FP64
    # A GrB_ name that GraphBLAS has only as GxB_ falls back to it
    assert not hasattr(lib, "GrB_HYPER_SWITCH")
    assert libget("GrB_HYPER_SWITCH") == lib.GxB_HYPER_SWITCH
    assert not hasattr(lib, "GrB_FC64")
    if supports_complex():
        assert libget("GrB_FC64") == lib.GxB_FC64
    # If neither exists, the error names what was asked for
    with pytest.raises(AttributeError, match="GrB_NO_SUCH_NAME"):
        libget("GrB_NO_SUCH_NAME")


def _type_names(types):
    return {grb_type.grb_type_get_string(ffi.new("GrB_Type*", T), lib.GrB_NAME) for T in types}


def test_type_sets():
    assert _type_names(bool_types) == {"GrB_BOOL"}
    assert _type_names(signed_integer_types) == {"GrB_INT8", "GrB_INT16", "GrB_INT32", "GrB_INT64"}
    assert _type_names(unsigned_integer_types) == {
        "GrB_UINT8",
        "GrB_UINT16",
        "GrB_UINT32",
        "GrB_UINT64",
    }
    assert integer_types == signed_integer_types | unsigned_integer_types
    assert _type_names(real_types) == {"GrB_FP32", "GrB_FP64"}
    expected_complex = {"GxB_FC32", "GxB_FC64"} if supports_complex() else set()
    assert _type_names(complex_types) == expected_complex
    assert grb_types == bool_types | integer_types | real_types | complex_types
    assert len(grb_types) == 11 + len(complex_types)


# conda-forge's cffi links conda's libffi, and through it a variadic call on macOS arm64 loses
# its variadic arguments, with or without vararg: a double arrives as 0.0 and an int32 setting
# doesn't change (cffi 2.0.0, libffi 3.5.2).  PyPI's cffi links the system libffi and works.
_conda_on_osx_arm64 = (
    sys.platform == "darwin"
    and suitesparse_graphblas._is_osx_arm64
    and os.path.isdir(os.path.join(sys.prefix, "conda-meta"))
)


@pytest.mark.xfail(
    _conda_on_osx_arm64,
    reason="variadic calls lose their arguments with conda-forge's cffi on macOS arm64",
)
def test_vararg():
    # vararg makes a value safe to pass to a variadic GraphBLAS function on every
    # platform: double, int32, and pointer (for the getters)
    A = matrix.matrix_new(lib.GrB_FP64, 3, 3)
    check_status(
        A,
        lib.GxB_Matrix_Option_set(A[0], lib.GxB_HYPER_SWITCH, vararg(ffi.cast("double", 0.375))),
    )
    assert matrix.matrix_hyper_switch(A) == 0.375
    check_status(
        A,
        lib.GxB_Matrix_Option_set(
            A[0], lib.GxB_SPARSITY_CONTROL, vararg(ffi.cast("int32_t", lib.GxB_BITMAP))
        ),
    )
    assert matrix.matrix_sparsity_control(A) == lib.GxB_BITMAP
    hyper_switch = ffi.new("double*")
    check_status(A, lib.GxB_Matrix_Option_get(A[0], lib.GxB_HYPER_SWITCH, vararg(hyper_switch)))
    assert hyper_switch[0] == 0.375
    sparsity = ffi.new("int32_t*")
    check_status(A, lib.GxB_Matrix_Option_get(A[0], lib.GxB_SPARSITY_CONTROL, vararg(sparsity)))
    assert sparsity[0] == lib.GxB_BITMAP
