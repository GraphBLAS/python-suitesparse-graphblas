import os
import sys
from cffi import FFI

is_win = sys.platform.startswith("win")
thisdir = os.path.dirname(__file__)

ffibuilder = FFI()

with open(os.path.join(thisdir, "source.c")) as f:
    source = f.read()

ffibuilder.set_source(
    "suitesparse_graphblas._graphblas",
    source,
    libraries=["graphblas"],
    include_dirs=[os.path.join(sys.prefix, "include")],
)

header = "suitesparse_graphblas.h"
if is_win:
    header = "suitesparse_graphblas_no_complex.h"
gb_cdef = open(os.path.join(thisdir, header))

ffibuilder.cdef(gb_cdef.read())

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
