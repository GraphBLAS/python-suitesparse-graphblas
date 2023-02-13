import os
import sys
from glob import glob

import numpy as np
from setuptools import Extension, setup

# Add current directory to the Python path because it's not present when running `pip install .`
sys.path.append(os.path.dirname(__file__))
import build_graphblas_cffi  # noqa: E402 # isort:skip

try:
    from Cython.Build import cythonize
    from Cython.Compiler.Options import get_directive_defaults

    use_cython = True
except ImportError:
    use_cython = False

define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

# /d2FH4- flag needed only for early Python 3.8 builds on Windows.
# See https://cibuildwheel.readthedocs.io/en/stable/faq/
# (Search for flag on page. Full link is long and causes the linter to fail the tests.)
#
# The /std:c11 flag is because the MSVC default is C89.
extra_compile_args = ["/d2FH4-", "/std:c11"] if sys.platform == "win32" else []

if use_cython:
    suffix = ".pyx"
    directive_defaults = get_directive_defaults()
    directive_defaults["binding"] = True
    directive_defaults["language_level"] = 3
    if os.environ.get("CYTHON_COVERAGE"):
        directive_defaults["linetrace"] = True
        define_macros.append(("CYTHON_TRACE_NOGIL", "1"))
else:
    suffix = ".c"
    # Make sure all required .c files are here
    pyx_files = glob("suitesparse_graphblas/**.pyx", recursive=True)
    c_files = glob("suitesparse_graphblas/**.c", recursive=True)
    missing = {x[:-4] for x in pyx_files} - {x[:-2] for x in c_files}
    if missing:
        missing_c = sorted(x + ".c" for x in missing)
        raise RuntimeError("Cython required when missing C files: " + ", ".join(missing_c))

include_dirs = [np.get_include(), os.path.join(sys.prefix, "include")]
ext_modules = [
    Extension(
        name[: -len(suffix)].replace("/", ".").replace("\\", "."),
        [name],
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )
    for name in glob(f"suitesparse_graphblas/**/*{suffix}", recursive=True)
]
if use_cython:
    ext_modules = cythonize(ext_modules, include_path=include_dirs)

ext_modules.append(build_graphblas_cffi.get_extension(extra_compile_args=extra_compile_args))

setup(
    ext_modules=ext_modules,
)
