import pytest

import suitesparse_graphblas
from suitesparse_graphblas import (
    _error_code_lookup,
    _error_func_lookup,
    check_status,
    exceptions,
    ffi,
    lib,
)
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
    # Every GrB_Info that GraphBLAS can return has its own exception class.
    # This fails when a new GraphBLAS adds a code, as a reminder to add one.
    info = getattr(lib, name)
    assert info in _error_code_lookup
    expected = _error_code_lookup[info]
    assert expected is StopIteration or (
        issubclass(expected, exceptions.GraphBLASException)
        and expected is not exceptions.GraphBLASException
    )
    A = ffi.new("GrB_Matrix*")
    check_status(A, lib.GrB_Matrix_new(A, lib.GrB_BOOL, 2, 2))
    with pytest.raises(expected):
        check_status(A, info)


def test_check_status_unknown_code():
    # A code that this build doesn't know raises the base class, not KeyError
    A = ffi.new("GrB_Matrix*")
    check_status(A, lib.GrB_Matrix_new(A, lib.GrB_BOOL, 2, 2))
    unknown = -12345
    assert unknown not in _error_code_lookup
    with pytest.raises(exceptions.GraphBLASException) as excinfo:
        check_status(A, unknown)
    assert type(excinfo.value) is exceptions.GraphBLASException


@pytest.mark.thread_unsafe(reason="patches the package")
@pytest.mark.parametrize(
    "info, exception",
    [(lib.GrB_OUT_OF_MEMORY, exceptions.OutOfMemory), (-12345, exceptions.GraphBLASException)],
    ids=["out of memory", "unknown"],
)
def test_initialize_fails(monkeypatch, info, exception):
    # If GraphBLAS fails to initialize, `initialize` raises the exception for the code, as
    # check_status does: the base class for a code that this build doesn't know
    monkeypatch.setattr(suitesparse_graphblas, "is_initialized", lambda: False)
    monkeypatch.setattr(suitesparse_graphblas.utils, "call_gxb_init", lambda ffi, lib, mode: info)
    with pytest.raises(exception, match=f"info={info}") as excinfo:
        suitesparse_graphblas.initialize()
    assert type(excinfo.value) is exception


def test_check_status_already_set():
    # GraphBLAS returns GrB_ALREADY_SET when the name of an object is set twice
    T = grb_type.grb_type_new(8)
    grb_type.grb_type_set_string(T, lib.GrB_NAME, "first_name")
    with pytest.raises(exceptions.AlreadySet):
        grb_type.grb_type_set_string(T, lib.GrB_NAME, "second_name")
    assert grb_type.grb_type_get_string(T, lib.GrB_NAME) == "first_name"


def test_check_status_output_is_readonly():
    # A vector loaded with GxB_IS_READONLY borrows the buffer, so GraphBLAS
    # refuses to write to it with GxB_OUTPUT_IS_READONLY
    v = ffi.new("GrB_Vector*")
    check_status(v, lib.GrB_Vector_new(v, lib.GrB_INT64, 0))
    buf = ffi.new("int64_t[4]", [1, 2, 3, 4])
    X = ffi.new("void**", buf)
    try:
        check_status(
            v,
            lib.GxB_Vector_load(
                v[0], X, lib.GrB_INT64, 4, ffi.sizeof(buf), lib.GxB_IS_READONLY, ffi.NULL
            ),
        )
        with pytest.raises(exceptions.OutputIsReadonly):
            check_status(v, lib.GrB_Vector_setElement_INT64(v[0], 9, 0))
        assert list(buf) == [1, 2, 3, 4]
    finally:
        check_status(v, lib.GrB_Vector_free(v))


def test_error_func_lookup_has_every_object_type():
    # check_status gets the error string with the object's *_error function.
    # Every *_error function's object type must be in the table, or a new
    # GraphBLAS object type would raise TypeError instead of its error.
    error_funcs = [name for name in dir(lib) if name.endswith("_error")]
    assert error_funcs
    for name in error_funcs:
        cname = ffi.typeof(getattr(lib, name)).args[1].cname
        assert cname in _error_func_lookup, name


def test_check_status_unknown_object():
    # The error string is looked up by the type of the object, so an object
    # that isn't a GraphBLAS handle is a TypeError, not a crash
    with pytest.raises(TypeError, match="Unknown cname int \\*"):
        check_status(ffi.new("int*"), lib.GrB_INVALID_VALUE)
