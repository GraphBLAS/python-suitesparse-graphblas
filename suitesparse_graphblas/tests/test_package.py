from suitesparse_graphblas import ffi, lib  # noqa: F401


def test_matrix_existence():
    assert hasattr(lib, "GrB_Matrix_new")
