import pytest

from suitesparse_graphblas import lib
from suitesparse_graphblas.api import global_options as go


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
