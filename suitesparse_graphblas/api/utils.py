import os
import sys


def _capture_c_output(fn, *args):
    """Capture C-level stdout output from a function call."""
    sys.stdout.flush()
    r, w = os.pipe()
    old = os.dup(1)
    os.dup2(w, 1)
    fn(*args)
    os.dup2(old, 1)
    os.close(w)
    out = os.read(r, 100000).decode()
    os.close(r)
    return out
