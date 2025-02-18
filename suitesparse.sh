#!/bin/bash

set -x  # echo on

# parse SuiteSparse version from first argument, a git tag that ends in the version (no leading v)
if [[ $1 =~ refs/tags/([0-9]*\.[0-9]*\.[0-9]*)\..*$ ]]; then
    VERSION=${BASH_REMATCH[1]}
else
    echo "Specify a SuiteSparse version, such as: $0 refs/tags/7.4.3.0 (got: $1)"
    exit -1
fi
echo VERSION: $VERSION

NPROC="$(nproc)"
if [ -z "${NPROC}" ]; then
    # Default for platforms that don't have nproc. Mostly Windows.
    NPROC="2"
fi

cmake_params=()
if [ -n "${BREW_LIBOMP}" ]; then
    # macOS OpenMP flags.
    # FindOpenMP doesn't find brew's libomp, so set the necessary configs manually.
    cmake_params+=(-DOpenMP_C_FLAGS="-Xclang -fopenmp -I$(brew --prefix libomp)/include")
    cmake_params+=(-DOpenMP_C_LIB_NAMES="libomp")
    cmake_params+=(-DOpenMP_libomp_LIBRARY="omp")
    export LDFLAGS="-L$(brew --prefix libomp)/lib"

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

if [ -n "${GRAPHBLAS_PREFIX}" ]; then
    echo "GRAPHBLAS_PREFIX=${GRAPHBLAS_PREFIX}"
    cmake_params+=(-DCMAKE_INSTALL_PREFIX="${GRAPHBLAS_PREFIX}")
fi

curl -L https://github.com/DrTimothyAldenDavis/GraphBLAS/archive/refs/tags/v${VERSION}.tar.gz | tar xzf -
cd GraphBLAS-${VERSION}/build

# Disable optimizing some rarely-used types for significantly faster builds and significantly smaller wheel size.
# Also the build with all types enabled sometimes stalls on GitHub Actions. Probably due to exceeded resource limits.
# These can still be used, they'll just have reduced performance (AFAIK similar to UDTs).
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

    # Setting COMPACT probably makes setting config in GB_control.h above unnecessary
    cmake_params+=(-DCOMPACT=1)
    # Also no JIT for the fastest possible build
    cmake_params+=(-DNJIT=1)
    # Disable all Source/Generated2 kernels. For workflow development only.
    cmake_params+=(-DCMAKE_CUDA_DEV=1)
fi

if [ -n "${CMAKE_GNUtoMS}" ]; then
    # Windows options
    echo "Skipping JIT on Windows for now because it fails to build."
    cmake_params+=(-DGRAPHBLAS_USE_JIT=OFF)
else
    # Use `-DJITINIT=2` so that the JIT functionality is available, but disabled by default.
    # Level 2, "run", means that pre-JIT kernels may be used, which does not require a compiler at runtime.
    cmake_params+=(-DJITINIT=2)

    # Disable JIT here too to not segfault in tests
    cmake_params+=(-DGRAPHBLAS_USE_JIT=OFF)
fi

# some platforms require sudo for installation, some don't have sudo at all
if [ "$(uname)" == "Darwin" ]; then
    SUDO=sudo
else
    SUDO=""
fi

cmake .. -DCMAKE_BUILD_TYPE=Release -G 'Unix Makefiles' "${cmake_params[@]}"
make -j$NPROC
$SUDO make install

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
