"""[SuiteSparse:GraphBLAS](https://github.com/DrTimothyAldenDavis/GraphBLAS)
is a full implementation of the GraphBLAS standard, which defines a
set of sparse matrix operations on an extended algebra of semirings
using an almost unlimited variety of operators and types. When applied
to sparse adjacency matrices, these algebraic operations are
equivalent to computations on graphs.  GraphBLAS provides a powerful
and expressive framework for creating high-performance graph
algorithms based on the elegant mathematics of sparse matrix
operations on a semiring.

This module contains the "functional API" for
python-suitesparse-graphblas.  This is a higher level API than the
"raw" CFFI binding, but a lower level API than the algebraic
functionality provided by the
[python-graphblas](https://python-graphblas.readthedocs.io/en/stable/api_reference/index.html)
library.

For documentation on the SuiteSparse implementation that describes all
of these functions in detail, read the [SuiteSparse:GraphBLAS User
Guide](https://github.com/DrTimothyAldenDavis/GraphBLAS/blob/stable/Doc/GraphBLAS_UserGuide.pdf)

"""
from suitesparse_graphblas.api import (
    binaryop,
    container,
    context,
    descriptor,
    global_,
    global_options,
    grb_type,
    indexbinaryop,
    indexunaryop,
    iterator,
    matrix,
    monoid,
    scalar,
    semiring,
    selectop,
    unaryop,
    vector,
)
from suitesparse_graphblas.api import io  # noqa: F401
