import pytest

from suitesparse_graphblas import ffi, lib, supports_complex  # noqa: F401


@pytest.mark.skipif("not supports_complex()")
def test_complex():
    s = ffi.new("GrB_Scalar*")
    success = lib.GrB_SUCCESS
    assert lib.GrB_Scalar_new(s, lib.GxB_FC64) == success
    assert lib.GxB_Scalar_setElement_FC64(s[0], 1j) == success
    assert lib.GrB_Scalar_free(s) == success
