# suitesparse-graphblas

Low-level Python CFFI bindings for [SuiteSparse:GraphBLAS](https://github.com/DrTimothyAldenDavis/GraphBLAS).

## Overview

`suitesparse-graphblas` exposes the raw `ffi` and `lib` symbols from the
SuiteSparse:GraphBLAS C library and provides a **functional API** in the
`suitesparse_graphblas.api` subpackage. The functional API consists of
module-level functions operating on opaque CFFI handles for each GraphBLAS type.

## Quick Start

```python
from suitesparse_graphblas import initialize, lib
from suitesparse_graphblas.api import matrix

initialize()

# Create a 3x3 boolean matrix
A = matrix.matrix_new(lib.GrB_BOOL, 3, 3)
matrix.set_bool(A, True, 0, 1)
matrix.set_bool(A, True, 1, 2)

print(matrix.matrix_nvals(A))  # 2
print(matrix.matrix_shape(A))  # (3, 3)
```

## API Modules

The functional API covers all GraphBLAS types:

| Module | Type | Description |
|--------|------|-------------|
| [matrix](api/matrix.md) | GrB_Matrix | Sparse matrix operations |
| [vector](api/vector.md) | GrB_Vector | Sparse vector operations |
| [scalar](api/scalar.md) | GxB_Scalar | Scalar values |
| [iterator](api/iterator.md) | GxB_Iterator | Matrix/vector iteration |
| [context](api/context.md) | GxB_Context | Thread configuration |
| [grb_type](api/grb_type.md) | GrB_Type | Type definitions |
| [unaryop](api/unaryop.md) | GrB_UnaryOp | Unary operators |
| [binaryop](api/binaryop.md) | GrB_BinaryOp | Binary operators |
| [indexunaryop](api/indexunaryop.md) | GrB_IndexUnaryOp | Index unary operators |
| [indexbinaryop](api/indexbinaryop.md) | GxB_IndexBinaryOp | Index binary operators |
| [monoid](api/monoid.md) | GrB_Monoid | Monoids |
| [semiring](api/semiring.md) | GrB_Semiring | Semirings |
| [descriptor](api/descriptor.md) | GrB_Descriptor | Operation descriptors |
| [selectop](api/selectop.md) | GxB_SelectOp | Select operators (deprecated) |
| [container](api/container.md) | GxB_Container | Containers |
| [global_options](api/global_options.md) | — | Global option get/set (GxB) |
| [global_](api/global_.md) | GrB_Global | Global get/set (GrB 2.1) |
| [io](api/io.md) | — | Serialization and binary I/O |

## Installation

```bash
pip install suitesparse-graphblas
```

Or from source with a local GraphBLAS build:

```bash
export GraphBLAS_ROOT="/path/to/graphblas"
pip install -e . --no-deps
```
