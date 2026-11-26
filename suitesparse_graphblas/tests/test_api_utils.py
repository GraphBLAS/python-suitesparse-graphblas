import os

import numpy as np
import pytest

from suitesparse_graphblas import exceptions, ffi, lib
from suitesparse_graphblas.api import context, vector
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


@pytest.mark.thread_unsafe(reason="patches os.dup2")
def test_capture_c_output_closes_fds_when_redirecting_fails(monkeypatch):
    def failing_dup2(fd, fd2):
        raise OSError("dup2 failed")

    fds = _free_fds()
    stdout = _stdout_id()
    monkeypatch.setattr(os, "dup2", failing_dup2)
    with pytest.raises(OSError, match="dup2 failed"):
        _capture_c_output(lambda: None)
    monkeypatch.undo()
    assert _stdout_id() == stdout
    assert _free_fds() == fds


@pytest.mark.thread_unsafe(reason="redirects file descriptor 1")
def test_capture_c_output_of_more_than_a_pipe_holds():
    # The output used to go to a pipe that was read after the call returned, so printing more
    # than the pipe holds (64 KiB on Linux and macOS) waited forever for a reader
    n = 10_000
    v = vector.vector_new(lib.GrB_INT64, n)
    indices = np.arange(n, dtype=np.uint64)
    vector.vector_build_int64(v, indices, indices.astype(np.int64), n, lib.GrB_PLUS_INT64)
    fds = _free_fds()
    out = _capture_c_output(vector.vector_print, v, "v", lib.GxB_COMPLETE)
    assert len(out) > 2**17
    entries = [line for line in out.splitlines() if line.startswith("    (")]
    assert entries == [f"    ({i},0)   {i}" for i in range(n)]
    assert _free_fds() == fds
