import pickle

import pytest

from suitesparse_graphblas import burble, lib, matrix
from suitesparse_graphblas.api.utils import _capture_c_output


@pytest.mark.thread_unsafe(reason="burble is a global option")
def test_is_enabled():
    assert burble.is_enabled is False
    burble.enable()
    try:
        assert burble.is_enabled is True
        assert repr(burble) == "<burble is_enabled=True>"
    finally:
        burble.disable()
    assert burble.is_enabled is False
    assert repr(burble) == "<burble is_enabled=False>"


@pytest.mark.thread_unsafe(reason="burble is a global option, and this captures stdout")
def test_context_manager_prints_diagnostics():
    A = matrix.matrix_new(lib.GrB_BOOL, 3, 3)
    # A pending tuple, so that matrix_nvals has work to do and report
    matrix.set_bool(A, True, 0, 1)
    with burble as b:
        assert b is burble
        assert burble.is_enabled is True
        out = _capture_c_output(matrix.matrix_nvals, A)
    assert burble.is_enabled is False
    assert "GrB_Matrix_nvals" in out
    matrix.set_bool(A, True, 1, 2)
    assert _capture_c_output(matrix.matrix_nvals, A) == ""


@pytest.mark.thread_unsafe(reason="burble is a global option")
def test_context_manager_restores_the_previous_state():
    with burble:
        with burble:
            assert burble.is_enabled is True
        # The inner block found burble enabled, so it leaves it enabled
        assert burble.is_enabled is True
    assert burble.is_enabled is False

    burble.enable()
    try:
        with burble:
            pass
        assert burble.is_enabled is True
    finally:
        burble.disable()

    with pytest.raises(ZeroDivisionError):
        with burble:
            1 / 0
    assert burble.is_enabled is False


def test_pickle():
    # burble is a singleton, and unpickles as itself
    assert pickle.loads(pickle.dumps(burble)) is burble
