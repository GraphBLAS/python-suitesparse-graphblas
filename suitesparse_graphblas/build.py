import os
import sys
import platform

from cffi import FFI

is_win = sys.platform.startswith("win")
is_arm64 = platform.machine() == "arm64"
is_ppc64le = platform.machine() == "ppc64le"  # Use same header as arm64, which *may* work
thisdir = os.path.dirname(__file__)

ffibuilder = FFI()

with open(os.path.join(thisdir, "source.c")) as f:
    source = f.read()

include_dirs = [os.path.join(sys.prefix, "include")]
library_dirs = [os.path.join(sys.prefix, "lib")]
if is_win:
    include_dirs.append(os.path.join(sys.prefix, "Library", "include"))
    library_dirs.append(os.path.join(sys.prefix, "Library", "lib"))

ffibuilder.set_source(
    "suitesparse_graphblas._graphblas",
    source,
    libraries=["graphblas"],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
)

header = "suitesparse_graphblas.h"
if is_win:
    header = "suitesparse_graphblas_no_complex.h"
if is_arm64 or is_ppc64le:
    header = "suitesparse_graphblas_arm64.h"
gb_cdef = open(os.path.join(thisdir, header))

ffibuilder.cdef(gb_cdef.read())

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
