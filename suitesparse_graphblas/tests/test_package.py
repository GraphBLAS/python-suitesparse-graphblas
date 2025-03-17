import suitesparse_graphblas
from suitesparse_graphblas import ffi, lib  # noqa: F401


def test_matrix_existence():
    assert hasattr(lib, "GrB_Matrix_new")


def test_version():
    # Example dev version: 9.4.5.0+2.g5590dba8.dirty
    # Example reslease version: 9.4.5.0
    version = suitesparse_graphblas.__version__
    version = [int(x) for x in version.split("+")[0].split(".")]
    assert version > [9, 4, 4, 0]
