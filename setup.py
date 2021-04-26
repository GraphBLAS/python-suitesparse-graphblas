from setuptools import setup, find_packages, Extension
from glob import glob

try:
    from Cython.Build import cythonize
    from Cython.Compiler.Options import get_directive_defaults

    use_cython = True
except ImportError:
    use_cython = False
import numpy as np
import os
import sys
import versioneer

is_win = sys.platform.startswith("win")
define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

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
    )
    for name in glob(f"suitesparse_graphblas/**/*{suffix}", recursive=True)
]
if use_cython:
    ext_modules = cythonize(ext_modules, include_path=include_dirs)

with open("README.md") as f:
    long_description = f.read()

package_data = {"suitesparse_graphblas": ["*.pyx", "*.pxd", "*.c", "*.h"]}
if is_win:
    package_data["suitesparse_graphblas"].append("*.dll")

setup(
    name="suitesparse-graphblas",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="SuiteSparse:GraphBLAS Python bindings.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    author="Michel Pelletier, James Kitchen, Erik Welch",
    author_email="michel@graphegon.com,jim22k@gmail.com,erik.n.welch@gmail.com",
    url="https://github.com/GraphBLAS/python-suitesparse-graphblas",
    ext_modules=ext_modules,
    cffi_modules=["suitesparse_graphblas/build.py:ffibuilder"],
    python_requires=">=3.7",
    install_requires=["cffi>=1.0.0", "numpy>=1.15"],
    setup_requires=["cffi>=1.0.0", "pytest-runner"],
    tests_require=["pytest"],
    license="Apache License 2.0",
    package_data=package_data,
    include_package_data=True,
    zip_safe=False,
)
