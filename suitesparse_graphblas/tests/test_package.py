import suitesparse_graphblas
from suitesparse_graphblas import ffi, lib  # noqa: F401


def test_matrix_existence():
    assert hasattr(lib, "GrB_Matrix_new")


def test_version():
    assert suitesparse_graphblas.__version__ > "7.4.2.0"
