import pytest

from suitesparse_graphblas import check_status, exceptions, ffi, lib
from suitesparse_graphblas.api import grb_type


def test_check_status():
    A = ffi.new("GrB_Matrix*")
    check_status(A, lib.GrB_Matrix_new(A, lib.GrB_BOOL, 2, 2))
    with pytest.raises(exceptions.Panic):
        check_status(A, lib.GrB_PANIC)
    with pytest.raises(exceptions.Panic):
        check_status(A[0], lib.GrB_PANIC)


def test_check_status_success_and_no_value():
    A = ffi.new("GrB_Matrix*")
    check_status(A, lib.GrB_Matrix_new(A, lib.GrB_BOOL, 2, 2))
    assert check_status(A, lib.GrB_SUCCESS) is None
    # GrB_NO_VALUE is not an error: the exception class is returned, not raised
    assert check_status(A, lib.GrB_NO_VALUE) is exceptions.NoValue


_error_names = sorted(
    name
    for name, info in ffi.typeof("GrB_Info").relements.items()
    if info not in {lib.GrB_SUCCESS, lib.GrB_NO_VALUE}
)


@pytest.mark.parametrize("name", _error_names)
def test_check_status_raises_for_every_error_code(name):
    # Every GrB_Info that GraphBLAS can return, so a code added by a new
    # GraphBLAS can't escape as a KeyError
    info = getattr(lib, name)
    A = ffi.new("GrB_Matrix*")
    check_status(A, lib.GrB_Matrix_new(A, lib.GrB_BOOL, 2, 2))
    expected = StopIteration if info == lib.GxB_EXHAUSTED else exceptions.GraphBLASException
    with pytest.raises(expected):
        check_status(A, info)


def test_check_status_already_set():
    # GraphBLAS returns GrB_ALREADY_SET when the name of an object is set twice
    T = grb_type.grb_type_new(8)
    grb_type.grb_type_set_string(T, lib.GrB_NAME, "first_name")
    with pytest.raises(exceptions.GraphBLASException):
        grb_type.grb_type_set_string(T, lib.GrB_NAME, "second_name")
    assert grb_type.grb_type_get_string(T, lib.GrB_NAME) == "first_name"


def test_check_status_unknown_object():
    # The error string is looked up by the type of the object, so an object
    # that isn't a GraphBLAS handle is a TypeError, not a crash
    with pytest.raises(TypeError, match="Unknown cname int \\*"):
        check_status(ffi.new("int*"), lib.GrB_INVALID_VALUE)
