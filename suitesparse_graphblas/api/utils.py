import os
import sys


def _capture_c_output(fn, *args):
    """Capture C-level stdout output from a function call."""
    sys.stdout.flush()
    r, w = os.pipe()
    old = os.dup(1)
    try:
        os.dup2(w, 1)
        try:
            fn(*args)
        finally:
            os.dup2(old, 1)
            os.close(w)
        try:
            return os.read(r, 100000).decode()
        except BlockingIOError:  # pragma: no cover (Emscripten: an empty pipe has no EOF)
            return ""
    finally:
        os.close(old)
        os.close(r)
