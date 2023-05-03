import os
import sys
from pathlib import Path

from cffi import FFI
from setuptools import Extension

is_win = sys.platform.startswith("win")
ss_g = Path(__file__).parent / "suitesparse_graphblas"

ffibuilder = FFI()

# GraphBLAS_ROOT env var can point to the root directory of GraphBLAS to link against.
# Expected subdirectories: include/ (contains GraphBLAS.h), lib/, and bin/ (on Windows only)
# Otherwise fallback to default system folders.
graphblas_root = os.environ.get("GraphBLAS_ROOT", None)
if not graphblas_root:
    # Windows wheels.yml configures suitesparse.sh to install GraphBLAS to "C:\\GraphBLAS".
    graphblas_root = "C:\\GraphBLAS" if is_win else sys.prefix

include_dirs = [os.path.join(graphblas_root, "include")]
library_dirs = [os.path.join(graphblas_root, "lib")]
if is_win:
    include_dirs.append(os.path.join(sys.prefix, "Library", "include"))
    library_dirs.append(os.path.join(sys.prefix, "Library", "lib"))

    include_dirs.append(os.path.join(graphblas_root, "include"))
    library_dirs.append(os.path.join(graphblas_root, "lib"))
    library_dirs.append(os.path.join(graphblas_root, "bin"))

ffibuilder.set_source(
    "suitesparse_graphblas._graphblas",
    (ss_g / "source.c").read_text(),
    libraries=["graphblas"],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
)

ffibuilder.cdef((ss_g / "suitesparse_graphblas.h").read_text())


def get_extension(apply_msvc_patch: bool = None, extra_compile_args=()):
    """
    Get a setuptools.Extension version of this CFFI builder.

    In other words, enables `setup(ext_modules=[get_extension()])`
    instead of `setup(cffi_modules=["build_graphblas_cffi.py:ffibuilder"])`.

    The main reason for this is to allow a patch for complex values when compiling on MSVC.
    MSVC famously lacks support for standard C complex types like `double _Complex` and
    `float _Complex`. Instead, MSVC has its own `_Dcomplex` and `_Fcomplex` types.
    Cffi's machinery cannot be made to work with these types, so we instead
    emit the regular standard C code and patch it manually.

    :param apply_msvc_patch: whether to apply the MSVC patch.
    If None then auto-detect based on platform.
    :param extra_compile_args: forwarded to Extension constructor.
    """
    code_path = ss_g / "_graphblas.c"
    ffibuilder.emit_c_code(str(code_path))

    if apply_msvc_patch is None:
        apply_msvc_patch = is_win

    if apply_msvc_patch:
        msvc_code = code_path.read_text()
        msvc_code = msvc_code.replace("float _Complex", "_Fcomplex")
        msvc_code = msvc_code.replace("double _Complex", "_Dcomplex")
        code_path.write_text(msvc_code)

    return Extension(
        "suitesparse_graphblas._graphblas",
        [os.path.join("suitesparse_graphblas", "_graphblas.c")],
        libraries=["graphblas"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_compile_args=extra_compile_args,
    )


if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
