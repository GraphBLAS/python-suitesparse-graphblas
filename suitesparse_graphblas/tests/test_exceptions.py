from suitesparse_graphblas import (
    ffi,
    lib,
    check_status,
    exceptions,
)

import pytest


def test_check_status():
    A = ffi.new("GrB_Matrix*")
    check_status(A, lib.GrB_Matrix_new(A, lib.GrB_BOOL, 2, 2))
    with pytest.raises(exceptions.Panic):
        check_status(A, lib.GrB_PANIC)
    with pytest.raises(exceptions.Panic):
        check_status(A[0], lib.GrB_PANIC)
