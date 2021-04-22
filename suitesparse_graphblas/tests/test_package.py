from suitesparse_graphblas import lib, ffi


def test_matrix_existence():
    assert hasattr(lib, "GrB_Matrix_new")
