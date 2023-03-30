#!/bin/bash

# parse SuiteSparse version from first argument, a git tag that ends in the version (no leading v)
if [[ $1 =~ refs/tags/([0-9]*\.[0-9]*\.[0-9]*)\..*$ ]];
then
    VERSION=${BASH_REMATCH[1]}
else
    echo "Specify a SuiteSparse version, such as: $0 refs/tags/7.4.3.0"
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

    export CFLAGS="-arch x86_64"
#    # build both x86 and ARM
#    export CFLAGS="-arch x86_64 -arch arm64"
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
echo "#define GxB_NO_FC32      1" >> ../Source/GB_control.h #
echo "#define GxB_NO_FC64      1" >> ../Source/GB_control.h #
# echo "#define GxB_NO_INT16     1" >> ../Source/GB_control.h
# echo "#define GxB_NO_INT32     1" >> ../Source/GB_control.h
# echo "#define GxB_NO_INT64     1" >> ../Source/GB_control.h #
# echo "#define GxB_NO_INT8      1" >> ../Source/GB_control.h
echo "#define GxB_NO_UINT16    1" >> ../Source/GB_control.h
echo "#define GxB_NO_UINT32    1" >> ../Source/GB_control.h
# echo "#define GxB_NO_UINT64    1" >> ../Source/GB_control.h
echo "#define GxB_NO_UINT8     1" >> ../Source/GB_control.h

# Disable all Source/Generated2 kernels. For workflow development only.
#cmake_params+=(-DCMAKE_CUDA_DEV=1)

cmake .. -DCMAKE_BUILD_TYPE=Release -G 'Unix Makefiles' "${cmake_params[@]}"
make -j$NPROC
make install

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
