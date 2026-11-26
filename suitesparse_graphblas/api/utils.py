import os
import sys
import tempfile


def _capture_c_output(fn, *args):
    """Capture C-level stdout output from a function call."""
    sys.stdout.flush()
    # A file, unlike a pipe, takes any amount of output without a reader, and needs no thread
    with tempfile.TemporaryFile() as f:
        old = os.dup(1)
        try:
            os.dup2(f.fileno(), 1)
            try:
                fn(*args)
            finally:
                os.dup2(old, 1)
        finally:
            os.close(old)
        f.seek(0)
        return f.read().decode()
