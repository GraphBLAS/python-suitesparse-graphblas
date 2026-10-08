#!/bin/bash

set -x  # echo on

# parse SuiteSparse version from first argument, a git tag that ends in the version (no leading v)
if [[ $1 =~ refs/tags/v?([0-9]+\.[0-9]+\.[0-9]+-beta\.[0-9]+) ]]; then
    # Naming used since v10.4.0-beta.1, e.g. "10.4.0-beta.2".  Note that callers
    # append the psg patch level (".0"), which is not part of the upstream tag.
    echo "Beta version detected (X.Y.Z-beta.N)"
    VERSION=${BASH_REMATCH[1]}
elif [[ $1 =~ refs/tags/([0-9]*\.[0-9]*\.[0-9]*\.beta[0-9]*).*$ ]]; then
    # Older naming, e.g. "8.0.1.beta1"
    echo "Beta version detected"
    VERSION=${BASH_REMATCH[1]}
elif [[ $1 =~ refs/tags/([0-9]*\.[0-9]*\.[0-9]*)\..*$ ]]; then
    VERSION=${BASH_REMATCH[1]}
else
    echo "Specify a SuiteSparse version, such as: $0 refs/tags/7.4.3.0 (got: $1)"
    exit 1
fi
echo VERSION: "$VERSION"

NPROC="$(nproc)"
if [ -z "${NPROC}" ]; then
    # Default for platforms that don't have nproc. Mostly Windows.
    NPROC="2"
fi

cmake_params=()

# Fail the build if OpenMP is missing instead of silently producing a serial
# library. GraphBLAS only warns and carries on, so a serial build stays
# invisible until someone measures it: conda-forge's graphblas 10.5.0 shipped
# that way on osx-arm64 while still depending on llvm-openmp. The matching
# runtime check on the built wheel is tests/test_package.py::test_openmp.
if [ -n "${SUITESPARSE_EMSCRIPTEN}" ]; then
    # ...except for WebAssembly, which has no threads in Pyodide
    cmake_params+=(-DSUITESPARSE_USE_OPENMP=OFF)
else
    cmake_params+=(-DSUITESPARSE_USE_OPENMP=ON)
fi
cmake_params+=(-DSUITESPARSE_USE_STRICT=ON)
# STRICT makes any requested-but-missing feature fatal, and SuiteSparsePolicy
# defaults both of these to ON, so they must be turned off explicitly or the
# configure step dies on "CUDA required for SuiteSparse but not found".
cmake_params+=(-DSUITESPARSE_USE_CUDA=OFF)
cmake_params+=(-DSUITESPARSE_USE_FORTRAN=OFF)

if [ -n "${BREW_LIBOMP}" ]; then
    # macOS OpenMP flags.
    # FindOpenMP doesn't find brew's libomp, so set the necessary configs manually.
    cmake_params+=(-DOpenMP_C_FLAGS="-Xclang -fopenmp -I$(brew --prefix libomp)/include")
    cmake_params+=(-DOpenMP_C_LIB_NAMES="libomp")
    cmake_params+=(-DOpenMP_libomp_LIBRARY="omp")
    LDFLAGS="-L$(brew --prefix libomp)/lib"
    export LDFLAGS

    if [ -n "${SUITESPARSE_MACOS_ARCH}" ]; then
        export CFLAGS="-arch ${SUITESPARSE_MACOS_ARCH}"
    else
        # build both x86 and ARM
        export CFLAGS="-arch x86_64 -arch arm64"
    fi
fi

if [ -n "${CMAKE_GNUtoMS}" ]; then
    # Windows needs .lib libraries, not .a
    cmake_params+=(-DCMAKE_GNUtoMS=ON)
    # Windows expects 'graphblas.lib', not 'libgraphblas.lib'
    cmake_params+=(-DCMAKE_SHARED_LIBRARY_PREFIX=)
    cmake_params+=(-DCMAKE_STATIC_LIBRARY_PREFIX=)
fi

cmake_cmd=(cmake)
if [ -n "${SUITESPARSE_EMSCRIPTEN}" ]; then
    # WebAssembly for Pyodide (emcc and pyodide must be on PATH, as in cibuildwheel's
    # before-build).  Build a static library to link into the extension module, with Pyodide's
    # compiler flags (-fPIC and its exception handling ABI).  Its -Oz is overridden by
    # CMake's Release flags (-O3), or by EMCC_CFLAGS if that sets one (see wheels.yml).
    cmake_cmd=(emcmake cmake)
    CFLAGS="$(pyodide config get cflags)" || exit 1
    export CFLAGS
    cmake_params+=(-DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON)
    # No CPU features to detect
    cmake_params+=(-DGBNCPUFEAT=1)
fi

if [ -n "${GRAPHBLAS_PREFIX}" ]; then
    echo "GRAPHBLAS_PREFIX=${GRAPHBLAS_PREFIX}"
    cmake_params+=(-DCMAKE_INSTALL_PREFIX="${GRAPHBLAS_PREFIX}")
fi

# Start clean: this may run more than once (e.g., per Python version for Pyodide)
rm -rf "GraphBLAS-${VERSION}"
curl -L "https://github.com/DrTimothyAldenDavis/GraphBLAS/archive/refs/tags/v${VERSION}.tar.gz" | tar xzf -
cd "GraphBLAS-${VERSION}/build" || exit

# Disable optimizing some rarely-used types for significantly faster builds and significantly smaller wheel size.
# Also the build with all types enabled sometimes stalls on GitHub Actions. Probably due to exceeded resource limits.
# These can still be used, they'll just have reduced performance (AFAIK similar to UDTs).
# GraphBLAS's own GB_control.h (as of 10.5.1) already disables INT8, INT16, UINT8 and UINT16, so
# the types left without FactoryKernels here are those four plus FC32, FC64 and UINT32.
# shellcheck disable=SC2129
# echo "#define GxB_NO_BOOL      1" >> ../Source/GB_control.h #
# echo "#define GxB_NO_FP32      1" >> ../Source/GB_control.h #
# echo "#define GxB_NO_FP64      1" >> ../Source/GB_control.h #
echo "#define GxB_NO_FC32      1" >> ../Source/GB_control.h
echo "#define GxB_NO_FC64      1" >> ../Source/GB_control.h
# echo "#define GxB_NO_INT16     1" >> ../Source/GB_control.h #
# echo "#define GxB_NO_INT32     1" >> ../Source/GB_control.h #
# echo "#define GxB_NO_INT64     1" >> ../Source/GB_control.h #
# echo "#define GxB_NO_INT8      1" >> ../Source/GB_control.h #
echo "#define GxB_NO_UINT16    1" >> ../Source/GB_control.h
echo "#define GxB_NO_UINT32    1" >> ../Source/GB_control.h
# echo "#define GxB_NO_UINT64    1" >> ../Source/GB_control.h #
echo "#define GxB_NO_UINT8     1" >> ../Source/GB_control.h

if [ -n "${SUITESPARSE_FAST_BUILD}" ]; then
    echo "suitesparse.sh: Fast build requested."
    # Disable optimizing even more types. This is for builds that don't finish in runner resource limits,
    # such as emulated aarm64.
    # shellcheck disable=SC2129

#    echo "#define GxB_NO_BOOL      1" >> ../Source/GB_control.h
#    echo "#define GxB_NO_FP32      1" >> ../Source/GB_control.h
#    echo "#define GxB_NO_FP64      1" >> ../Source/GB_control.h
    echo "#define GxB_NO_FC32      1" >> ../Source/GB_control.h
    echo "#define GxB_NO_FC64      1" >> ../Source/GB_control.h
    echo "#define GxB_NO_INT16     1" >> ../Source/GB_control.h
    echo "#define GxB_NO_INT32     1" >> ../Source/GB_control.h
#    echo "#define GxB_NO_INT64     1" >> ../Source/GB_control.h
    echo "#define GxB_NO_INT8      1" >> ../Source/GB_control.h
    echo "#define GxB_NO_UINT16    1" >> ../Source/GB_control.h
    echo "#define GxB_NO_UINT32    1" >> ../Source/GB_control.h
    echo "#define GxB_NO_UINT64    1" >> ../Source/GB_control.h
    echo "#define GxB_NO_UINT8     1" >> ../Source/GB_control.h
fi

if [ -n "${SUITESPARSE_FASTEST_BUILD}" ]; then
    echo "suitesparse.sh: Fastest build requested."
    # Fastest build possible. For use in development and automated tests that do not depend on performance.
    # shellcheck disable=SC2129

    echo "#define GxB_NO_BOOL      1" >> ../Source/GB_control.h
    echo "#define GxB_NO_FP32      1" >> ../Source/GB_control.h
    echo "#define GxB_NO_FP64      1" >> ../Source/GB_control.h
    echo "#define GxB_NO_FC32      1" >> ../Source/GB_control.h
    echo "#define GxB_NO_FC64      1" >> ../Source/GB_control.h
    echo "#define GxB_NO_INT16     1" >> ../Source/GB_control.h
    echo "#define GxB_NO_INT32     1" >> ../Source/GB_control.h
    echo "#define GxB_NO_INT64     1" >> ../Source/GB_control.h
    echo "#define GxB_NO_INT8      1" >> ../Source/GB_control.h
    echo "#define GxB_NO_UINT16    1" >> ../Source/GB_control.h
    echo "#define GxB_NO_UINT32    1" >> ../Source/GB_control.h
    echo "#define GxB_NO_UINT64    1" >> ../Source/GB_control.h
    echo "#define GxB_NO_UINT8     1" >> ../Source/GB_control.h

    # No FactoryKernels at all, which probably makes the GB_control.h settings above unnecessary
    cmake_params+=(-DGRAPHBLAS_COMPACT=ON)
fi

if [ -n "${CMAKE_GNUtoMS}" ]; then
    # Windows options
    echo "Skipping JIT on Windows for now because it fails to build."
    cmake_params+=(-DGRAPHBLAS_USE_JIT=OFF)
else
    # Use `GRAPHBLAS_JITINIT=2` so that the JIT functionality is available, but disabled by default.
    # Level 2, "run", means that pre-JIT kernels may be used, which does not require a compiler at runtime.
    cmake_params+=(-DGRAPHBLAS_JITINIT=2)

    # Disable JIT here too to not segfault in tests
    cmake_params+=(-DGRAPHBLAS_USE_JIT=OFF)
fi

# some platforms require sudo for installation, some don't have sudo at all
if [ "$(uname)" == "Darwin" ] && [ -z "${SUITESPARSE_EMSCRIPTEN}" ]; then
    SUDO=sudo
else
    SUDO=""
fi

# Stop at the first failure.  Carrying on would hide it until the wheel fails to link, or let
# the wheel link against a library left by an earlier run (e.g., for another Python).
"${cmake_cmd[@]}" .. -DCMAKE_BUILD_TYPE=Release -G 'Unix Makefiles' "${cmake_params[@]}" || exit 1
make -j"$NPROC" || exit 1
$SUDO make install || exit 1

if [ -n "${CMAKE_GNUtoMS}" ]; then
    if [ -z "${GRAPHBLAS_PREFIX}" ]; then
        # Windows default
        GRAPHBLAS_PREFIX="C:/Program Files (x86)"
    fi

    # Windows:
    # CMAKE_STATIC_LIBRARY_PREFIX is sometimes ignored, possibly when the MinGW toolchain is selected.
    # Drop the 'lib' prefix manually.
    echo "manually removing lib prefix"
    mv "${GRAPHBLAS_PREFIX}/lib/libgraphblas.lib" "${GRAPHBLAS_PREFIX}/lib/graphblas.lib"
    mv "${GRAPHBLAS_PREFIX}/lib/libgraphblas.dll.a" "${GRAPHBLAS_PREFIX}/lib/graphblas.dll.a"
    # cp instead of mv because the GNU tools expect libgraphblas.dll and the MS tools expect graphblas.dll.
    cp "${GRAPHBLAS_PREFIX}/bin/libgraphblas.dll" "${GRAPHBLAS_PREFIX}/bin/graphblas.dll"
fi
