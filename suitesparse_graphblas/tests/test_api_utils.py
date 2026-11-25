import os

import pytest

from suitesparse_graphblas import exceptions, ffi, lib
from suitesparse_graphblas.api import context
from suitesparse_graphblas.api.utils import _capture_c_output


def _free_fds():
    # New file descriptors take the lowest free numbers, so these change if one leaks
    fds = [os.dup(1) for _ in range(8)]
    for fd in fds:
        os.close(fd)
    return fds


def _stdout_id():
    st = os.fstat(1)
    return st.st_dev, st.st_ino


@pytest.mark.thread_unsafe(reason="redirects file descriptor 1")
def test_capture_c_output():
    ctx = context.context_new()
    fds = _free_fds()
    stdout = _stdout_id()
    for _ in range(3):
        out = _capture_c_output(context.context_print, ctx, "my_ctx", lib.GxB_SHORT)
        assert "my_ctx" in out
    assert _stdout_id() == stdout
    assert _free_fds() == fds


@pytest.mark.thread_unsafe(reason="redirects file descriptor 1")
def test_capture_c_output_restores_stdout_on_error():
    ctx = context.context_new(free=None)
    context.context_free(ctx)
    fds = _free_fds()
    stdout = _stdout_id()
    with pytest.raises(exceptions.NullPointer):
        _capture_c_output(context.context_print, ctx)
    assert ctx[0] == ffi.NULL
    assert _stdout_id() == stdout
    assert _free_fds() == fds


@pytest.mark.thread_unsafe(reason="redirects file descriptor 1")
def test_capture_c_output_of_nothing():
    fds = _free_fds()
    assert _capture_c_output(lambda: None) == ""
    assert _free_fds() == fds
