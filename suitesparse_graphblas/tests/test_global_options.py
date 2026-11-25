import pytest

from suitesparse_graphblas import exceptions, lib, matrix
from suitesparse_graphblas.api import global_options as go

INVALID_FIELD = 999999


@pytest.mark.parametrize(
    "func, args",
    [
        (go.global_option_get_int32, ()),
        (go.global_option_set_int32, (1,)),
        (go.global_option_get_fp64, ()),
        (go.global_option_set_fp64, (1.0,)),
        (go.global_option_get_char, ()),
        (go.global_get_int32, ()),
        (go.global_set_int32, (1,)),
        (go.global_get_size, ()),
        (go.global_get_string, ()),
        (go.global_set_string, ("x",)),
    ],
)
def test_invalid_field(func, args):
    with pytest.raises(exceptions.InvalidValue, match="failed with info=-3"):
        func(INVALID_FIELD, *args)


@pytest.mark.parametrize("field", ["GxB_API_ABOUT", "GxB_LIBRARY_LICENSE"])
def test_global_get_string_longer_than_256(field):
    # Both are longer than 256 characters in every build
    field = getattr(lib, field)
    value = go.global_get_string(field)
    assert value == go.global_option_get_char(field)
    assert len(value) > 256
    assert go.global_get_size(field) == len(value.encode()) + 1


def test_global_get_string_and_size():
    assert go.global_get_string(lib.GrB_NAME) == "SuiteSparse:GraphBLAS"
    assert go.global_get_size(lib.GrB_NAME) == len("SuiteSparse:GraphBLAS") + 1


@pytest.mark.thread_unsafe(reason="sets a global option")
def test_global_set_string():
    # The C preface goes at the top of each JIT kernel; setting it compiles nothing
    old = go.global_get_string(lib.GxB_JIT_C_PREFACE)
    new = "/* " + "x" * 1000 + " */"
    go.global_set_string(lib.GxB_JIT_C_PREFACE, new)
    try:
        assert go.global_get_string(lib.GxB_JIT_C_PREFACE) == new
        assert go.global_option_get_char(lib.GxB_JIT_C_PREFACE) == new
        assert go.global_get_size(lib.GxB_JIT_C_PREFACE) == len(new) + 1
    finally:
        go.global_set_string(lib.GxB_JIT_C_PREFACE, old)
    assert go.global_get_string(lib.GxB_JIT_C_PREFACE) == old


@pytest.mark.thread_unsafe(reason="sets a global option")
def test_global_option_int32():
    old = go.global_option_get_int32(lib.GxB_PRINT_1BASED)
    try:
        for value in (1, 0):
            go.global_option_set_int32(lib.GxB_PRINT_1BASED, value)
            assert go.global_option_get_int32(lib.GxB_PRINT_1BASED) == value
            # The GxB_Global_Option and GrB_Global interfaces share the setting
            assert go.global_get_int32(lib.GxB_PRINT_1BASED) == value
    finally:
        go.global_option_set_int32(lib.GxB_PRINT_1BASED, old)


@pytest.mark.thread_unsafe(reason="sets a global option")
def test_global_set_int32():
    old = go.global_get_int32(lib.GxB_PRINT_1BASED)
    try:
        for value in (1, 0):
            go.global_set_int32(lib.GxB_PRINT_1BASED, value)
            assert go.global_get_int32(lib.GxB_PRINT_1BASED) == value
            assert go.global_option_get_int32(lib.GxB_PRINT_1BASED) == value
    finally:
        go.global_set_int32(lib.GxB_PRINT_1BASED, old)


@pytest.mark.thread_unsafe(reason="sets a global option")
def test_global_option_fp64():
    old = go.global_option_get_fp64(lib.GxB_HYPER_SWITCH)
    new = 0.25 if old != 0.25 else 0.125
    go.global_option_set_fp64(lib.GxB_HYPER_SWITCH, new)
    try:
        assert go.global_option_get_fp64(lib.GxB_HYPER_SWITCH) == new
        # The global setting is the default for new matrices
        A = matrix.matrix_new(lib.GrB_BOOL, 3, 3)
        assert matrix.matrix_hyper_switch(A) == new
    finally:
        go.global_option_set_fp64(lib.GxB_HYPER_SWITCH, old)
    assert go.global_option_get_fp64(lib.GxB_HYPER_SWITCH) == old
