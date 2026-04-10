import numpy as np

from suitesparse_graphblas import check_status, ffi, lib, supports_complex

from .utils import _capture_c_output  # noqa: F401

from .io.serialize import deserialize_matrix as deserialize  # noqa: F401
from .io.serialize import serialize_matrix as serialize  # noqa: F401


def matrix_free(A):
    """Free a matrix."""
    check_status(A, lib.GrB_Matrix_free(A))


def matrix_new(T, nrows=lib.GxB_INDEX_MAX, ncols=lib.GxB_INDEX_MAX, *, free=matrix_free):
    """Create a new `GrB_Matrix` of type `T` and initialize it.  The
    following example creates an eight bit unsigned 2x2 matrix:

    >>> A = matrix_new(lib.GrB_UINT8, 2, 2)
    >>> matrix_shape(A)
    (2, 2)

    The default value for `nrows` and `ncols` is `lib.GxB_INDEX_MAX`
    which creates a Matrix with maximal bounds:

    >>> A = matrix_new(lib.GrB_UINT8)
    >>> matrix_shape(A) == (lib.GxB_INDEX_MAX, lib.GxB_INDEX_MAX)
    True

    The `free` argument is called when the object is garbage
    collected, the default is `matrix.matrix_free()`.  If `free` is None then
    there is no automatic garbage collection and it is up to the user
    to free the matrix.

    """
    A = ffi.new("GrB_Matrix*")
    check_status(A, lib.GrB_Matrix_new(A, T, nrows, ncols))
    if free:
        return ffi.gc(A, free)
    return A


def matrix_type(A):
    """Return the GraphBLAS type of the vector.

    >>> A = matrix_new(lib.GrB_UINT8)
    >>> matrix_type(A) == lib.GrB_UINT8
    True

    """
    T = ffi.new("GrB_Type*")
    check_status(A, lib.GxB_Matrix_type(T, A[0]))
    return T[0]


def matrix_nrows(A):
    """Return the number of rows in the matrix.

    >>> A = matrix_new(lib.GrB_UINT8, 2, 3)
    >>> matrix_nrows(A)
    2

    """
    n = ffi.new("GrB_Index*")
    check_status(A, lib.GrB_Matrix_nrows(n, A[0]))
    return n[0]


def matrix_ncols(A):
    """Return the number of columns in the matrix.

    >>> A = matrix_new(lib.GrB_UINT8, 2, 3)
    >>> matrix_ncols(A)
    3

    """
    n = ffi.new("GrB_Index*")
    check_status(A, lib.GrB_Matrix_ncols(n, A[0]))
    return n[0]


def matrix_nvals(A):
    """Return the number of stored elements in the matrix.

    >>> A = matrix_new(lib.GrB_UINT8, 2, 3)
    >>> matrix_nvals(A)
    0

    """
    n = ffi.new("GrB_Index*")
    check_status(A, lib.GrB_Matrix_nvals(n, A[0]))
    return n[0]


def matrix_shape(A):
    """Return the shape of the matrix as a two tuple `(nrows, ncols)`

    >>> A = matrix_new(lib.GrB_UINT8, 2, 2)
    >>> matrix_shape(A)
    (2, 2)

    """
    return (matrix_nrows(A), matrix_ncols(A))


def matrix_format(A):
    """Return the format of the matrix.

    >>> A = matrix_new(lib.GrB_UINT8, 2, 2)
    >>> matrix_format(A) == lib.GxB_BY_ROW
    True

    """
    format = ffi.new("int32_t*")
    check_status(A, lib.GxB_Matrix_Option_get_INT32(A[0], lib.GxB_FORMAT, format))
    return format[0]


def matrix_set_format(A, format):
    """Set the format of the matrix.

    >>> A = matrix_new(lib.GrB_UINT8, 2, 2)
    >>> matrix_set_format(A, lib.GxB_BY_COL)
    >>> matrix_format(A) == lib.GxB_BY_COL
    True

    """
    format_val = ffi.cast("int32_t", format)
    check_status(A, lib.GxB_Matrix_Option_set_INT32(A[0], lib.GxB_FORMAT, format_val))


def matrix_sparsity_status(A):
    """Get the sparsity status of the matrix."""
    sparsity_status = ffi.new("int32_t*")
    check_status(A, lib.GxB_Matrix_Option_get_INT32(A[0], lib.GxB_SPARSITY_STATUS, sparsity_status))
    return sparsity_status[0]


def matrix_sparsity_control(A):
    """Get the sparsity control of the matrix."""
    sparsity_control = ffi.new("int32_t*")
    check_status(
        A, lib.GxB_Matrix_Option_get_INT32(A[0], lib.GxB_SPARSITY_CONTROL, sparsity_control)
    )
    return sparsity_control[0]


def matrix_set_sparsity_control(A, sparsity):
    """Set the sparsity control of the matrix."""
    sparsity_control = ffi.cast("int32_t", sparsity)
    check_status(
        A, lib.GxB_Matrix_Option_set_INT32(A[0], lib.GxB_SPARSITY_CONTROL, sparsity_control)
    )


def matrix_hyper_switch(A):
    """Get the hyper switch of the matrix."""
    hyper_switch = ffi.new("double*")
    check_status(A, lib.GxB_Matrix_Option_get_FP64(A[0], lib.GxB_HYPER_SWITCH, hyper_switch))
    return hyper_switch[0]


def matrix_set_hyper_switch(A, hyper_switch):
    """Set the hyper switch of the matrix."""
    hyper_switch = ffi.cast("double", hyper_switch)
    check_status(A, lib.GxB_Matrix_Option_set_FP64(A[0], lib.GxB_HYPER_SWITCH, hyper_switch))


def matrix_bitmap_switch(A):
    """Get the bitmap switch of the matrix."""
    bitmap_switch = ffi.new("double*")
    check_status(A, lib.GxB_Matrix_Option_get_FP64(A[0], lib.GxB_BITMAP_SWITCH, bitmap_switch))
    return bitmap_switch[0]


def matrix_set_bitmap_switch(A, bitmap_switch):
    """Set the bitmap switch of the matrix."""
    bitmap_switch = ffi.cast("double", bitmap_switch)
    check_status(A, lib.GxB_Matrix_Option_set_FP64(A[0], lib.GxB_BITMAP_SWITCH, bitmap_switch))


def matrix_option_get_int32(A, field):
    """Get a matrix option as an int32.

    >>> A = matrix_new(lib.GrB_INT64, 2, 2)
    >>> matrix_option_get_int32(A, lib.GxB_FORMAT) == lib.GxB_BY_ROW
    True

    """
    val = ffi.new("int32_t*")
    check_status(A, lib.GxB_Matrix_Option_get_INT32(A[0], field, val))
    return val[0]


def matrix_option_set_int32(A, field, value):
    """Set a matrix option from an int32.

    >>> A = matrix_new(lib.GrB_INT64, 2, 2)
    >>> matrix_option_set_int32(A, lib.GxB_FORMAT, lib.GxB_BY_COL)
    >>> matrix_option_get_int32(A, lib.GxB_FORMAT) == lib.GxB_BY_COL
    True

    """
    check_status(A, lib.GxB_Matrix_Option_set_INT32(
        A[0], field, ffi.cast("int32_t", value),
    ))


def matrix_option_get_fp64(A, field):
    """Get a matrix option as a float64.

    >>> A = matrix_new(lib.GrB_INT64, 2, 2)
    >>> isinstance(matrix_option_get_fp64(A, lib.GxB_HYPER_SWITCH), float)
    True

    """
    val = ffi.new("double*")
    check_status(A, lib.GxB_Matrix_Option_get_FP64(A[0], field, val))
    return val[0]


def matrix_option_set_fp64(A, field, value):
    """Set a matrix option from a float64.

    >>> A = matrix_new(lib.GrB_INT64, 2, 2)
    >>> matrix_option_set_fp64(A, lib.GxB_HYPER_SWITCH, 0.5)
    >>> matrix_option_get_fp64(A, lib.GxB_HYPER_SWITCH) == 0.5
    True

    """
    check_status(A, lib.GxB_Matrix_Option_set_FP64(
        A[0], field, ffi.cast("double", value),
    ))


# ---------------------------------------------------------------------------
# GraphBLAS operations
# ---------------------------------------------------------------------------


def matrix_dup(A, *, free=matrix_free):
    """Duplicate a matrix.

    >>> A = matrix_new(lib.GrB_INT64, 2, 2)
    >>> set_int64(A, 42, 0, 0)
    >>> B = matrix_dup(A)
    >>> get_int64(B, 0, 0) == 42
    True

    """
    C = ffi.new("GrB_Matrix*")
    check_status(C, lib.GrB_Matrix_dup(C, A[0]))
    if free:
        return ffi.gc(C, free)
    return C


def matrix_clear(A):
    """Remove all entries from a matrix.

    >>> A = matrix_new(lib.GrB_INT64, 2, 2)
    >>> set_int64(A, 42, 0, 0)
    >>> matrix_nvals(A) == 1
    True
    >>> matrix_clear(A)
    >>> matrix_nvals(A) == 0
    True

    """
    check_status(A, lib.GrB_Matrix_clear(A[0]))


def matrix_wait(A, waitmode=lib.GrB_COMPLETE):
    """Wait for a matrix to complete pending operations.

    >>> A = matrix_new(lib.GrB_INT64, 2, 2)
    >>> matrix_wait(A)

    """
    check_status(A, lib.GrB_Matrix_wait(A[0], waitmode))


def matrix_print(A, name="", level=lib.GxB_COMPLETE):
    """Print a matrix to stdout.

    ``level`` controls verbosity: ``lib.GxB_SILENT``, ``lib.GxB_SUMMARY``,
    ``lib.GxB_SHORT``, ``lib.GxB_COMPLETE``, ``lib.GxB_SHORT_VERBOSE``,
    or ``lib.GxB_COMPLETE_VERBOSE``.

    >>> A = matrix_new(lib.GrB_INT64, 2, 2)
    >>> set_int64(A, 42, 0, 0)
    >>> matrix_wait(A, lib.GrB_MATERIALIZE)
    >>> out = _capture_c_output(matrix_print, A, 'A', lib.GxB_SHORT)
    >>> '2x2 GraphBLAS int64_t matrix' in out
    True
    >>> '(0,0)   42' in out
    True

    """
    check_status(A, lib.GxB_Matrix_fprint(
        A[0], name.encode() if isinstance(name, str) else name,
        level, ffi.NULL,
    ))


def matrix_fprint(A, f, name="", level=lib.GxB_COMPLETE):
    """Print a matrix to a C FILE* stream.

    Pass ``ffi.NULL`` for ``f`` to print to stdout.
    ``level`` controls verbosity (see :func:`matrix_print`).

    >>> A = matrix_new(lib.GrB_INT64, 2, 2)
    >>> set_int64(A, 7, 1, 1)
    >>> matrix_wait(A, lib.GrB_MATERIALIZE)
    >>> out = _capture_c_output(matrix_fprint, A, ffi.NULL, 'A', lib.GxB_SHORT)
    >>> '2x2 GraphBLAS int64_t matrix' in out
    True
    >>> '(1,1)   7' in out
    True

    """
    check_status(A, lib.GxB_Matrix_fprint(
        A[0], name.encode() if isinstance(name, str) else name,
        level, f,
    ))


def matrix_mxm(C, semiring, A, B, mask=None, accum=None, desc=None):
    """Multiply two matrices using a semiring: C = A (+.*) B.

    >>> C = matrix_new(lib.GrB_INT64, 2, 2)
    >>> A = matrix_new(lib.GrB_INT64, 2, 2)
    >>> B = matrix_new(lib.GrB_INT64, 2, 2)
    >>> set_int64(A, 2, 0, 0)
    >>> set_int64(B, 3, 0, 0)
    >>> matrix_mxm(C, lib.GrB_PLUS_TIMES_SEMIRING_INT64, A, B)
    >>> get_int64(C, 0, 0) == 6
    True

    """
    check_status(C, lib.GrB_mxm(
        C[0],
        ffi.NULL if mask is None else mask[0],
        ffi.NULL if accum is None else accum,
        semiring,
        A[0],
        B[0],
        ffi.NULL if desc is None else desc[0],
    ))


def matrix_ewise_add(C, op, A, B, mask=None, accum=None, desc=None):
    """Element-wise addition of two matrices using a binary operator.

    >>> C = matrix_new(lib.GrB_INT64, 2, 2)
    >>> A = matrix_new(lib.GrB_INT64, 2, 2)
    >>> B = matrix_new(lib.GrB_INT64, 2, 2)
    >>> set_int64(A, 2, 0, 0)
    >>> set_int64(B, 3, 0, 0)
    >>> matrix_ewise_add(C, lib.GrB_PLUS_INT64, A, B)
    >>> get_int64(C, 0, 0) == 5
    True

    """
    check_status(C, lib.GrB_Matrix_eWiseAdd_BinaryOp(
        C[0],
        ffi.NULL if mask is None else mask[0],
        ffi.NULL if accum is None else accum,
        op,
        A[0],
        B[0],
        ffi.NULL if desc is None else desc[0],
    ))


def matrix_ewise_mult(C, op, A, B, mask=None, accum=None, desc=None):
    """Element-wise multiplication of two matrices using a binary operator.

    >>> C = matrix_new(lib.GrB_INT64, 2, 2)
    >>> A = matrix_new(lib.GrB_INT64, 2, 2)
    >>> B = matrix_new(lib.GrB_INT64, 2, 2)
    >>> set_int64(A, 2, 0, 0)
    >>> set_int64(B, 3, 0, 0)
    >>> matrix_ewise_mult(C, lib.GrB_TIMES_INT64, A, B)
    >>> get_int64(C, 0, 0) == 6
    True

    """
    check_status(C, lib.GrB_Matrix_eWiseMult_BinaryOp(
        C[0],
        ffi.NULL if mask is None else mask[0],
        ffi.NULL if accum is None else accum,
        op,
        A[0],
        B[0],
        ffi.NULL if desc is None else desc[0],
    ))


def matrix_ewise_union(C, op, A, alpha, B, beta, mask=None, accum=None, desc=None):
    """Element-wise union of two matrices using a binary operator.

    Unlike eWiseAdd, entries present in only one operand use the
    corresponding default scalar (``alpha`` for A, ``beta`` for B).

    >>> from suitesparse_graphblas import scalar
    >>> C = matrix_new(lib.GrB_INT64, 2, 2)
    >>> A = matrix_new(lib.GrB_INT64, 2, 2)
    >>> B = matrix_new(lib.GrB_INT64, 2, 2)
    >>> set_int64(A, 2, 0, 0)
    >>> set_int64(B, 3, 1, 1)
    >>> a = scalar.scalar_new(lib.GrB_INT64)
    >>> b = scalar.scalar_new(lib.GrB_INT64)
    >>> scalar.set_int64(a, 0)
    >>> scalar.set_int64(b, 0)
    >>> matrix_ewise_union(C, lib.GrB_PLUS_INT64, A, a, B, b)
    >>> get_int64(C, 0, 0) == 2
    True
    >>> get_int64(C, 1, 1) == 3
    True

    """
    check_status(C, lib.GxB_Matrix_eWiseUnion(
        C[0],
        ffi.NULL if mask is None else mask[0],
        ffi.NULL if accum is None else accum,
        op,
        A[0], alpha[0],
        B[0], beta[0],
        ffi.NULL if desc is None else desc[0],
    ))


def matrix_apply(C, op, A, mask=None, accum=None, desc=None):
    """Apply a unary operator to each entry of a matrix.

    >>> C = matrix_new(lib.GrB_INT64, 2, 2)
    >>> A = matrix_new(lib.GrB_INT64, 2, 2)
    >>> set_int64(A, -5, 0, 0)
    >>> matrix_apply(C, lib.GrB_ABS_INT64, A)
    >>> get_int64(C, 0, 0) == 5
    True

    """
    check_status(C, lib.GrB_Matrix_apply(
        C[0],
        ffi.NULL if mask is None else mask[0],
        ffi.NULL if accum is None else accum,
        op,
        A[0],
        ffi.NULL if desc is None else desc[0],
    ))


def matrix_apply_first(C, op, x, A, mask=None, accum=None, desc=None):
    """Apply a binary operator with a scalar bound to the first argument: C = op(x, A).

    >>> from suitesparse_graphblas import scalar
    >>> C = matrix_new(lib.GrB_INT64, 2, 2)
    >>> A = matrix_new(lib.GrB_INT64, 2, 2)
    >>> set_int64(A, 3, 0, 0)
    >>> s = scalar.scalar_new(lib.GrB_INT64)
    >>> scalar.set_int64(s, 10)
    >>> matrix_apply_first(C, lib.GrB_TIMES_INT64, s, A)
    >>> get_int64(C, 0, 0) == 30
    True

    """
    check_status(C, lib.GrB_Matrix_apply_BinaryOp1st_Scalar(
        C[0],
        ffi.NULL if mask is None else mask[0],
        ffi.NULL if accum is None else accum,
        op,
        x[0],
        A[0],
        ffi.NULL if desc is None else desc[0],
    ))


def matrix_apply_second(C, op, A, y, mask=None, accum=None, desc=None):
    """Apply a binary operator with a scalar bound to the second argument: C = op(A, y).

    >>> from suitesparse_graphblas import scalar
    >>> C = matrix_new(lib.GrB_INT64, 2, 2)
    >>> A = matrix_new(lib.GrB_INT64, 2, 2)
    >>> set_int64(A, 10, 0, 0)
    >>> s = scalar.scalar_new(lib.GrB_INT64)
    >>> scalar.set_int64(s, 3)
    >>> matrix_apply_second(C, lib.GrB_TIMES_INT64, A, s)
    >>> get_int64(C, 0, 0) == 30
    True

    """
    check_status(C, lib.GrB_Matrix_apply_BinaryOp2nd_Scalar(
        C[0],
        ffi.NULL if mask is None else mask[0],
        ffi.NULL if accum is None else accum,
        op,
        A[0],
        y[0],
        ffi.NULL if desc is None else desc[0],
    ))


def matrix_select(C, op, A, thunk, mask=None, accum=None, desc=None):
    """Select entries from a matrix using an index unary operator and scalar thunk.

    >>> from suitesparse_graphblas import scalar
    >>> C = matrix_new(lib.GrB_INT64, 2, 2)
    >>> A = matrix_new(lib.GrB_INT64, 2, 2)
    >>> set_int64(A, 5, 0, 0)
    >>> set_int64(A, 1, 1, 1)
    >>> s = scalar.scalar_new(lib.GrB_INT64)
    >>> scalar.set_int64(s, 3)
    >>> matrix_select(C, lib.GrB_VALUEGT_INT64, A, s)
    >>> matrix_nvals(C) == 1
    True
    >>> get_int64(C, 0, 0) == 5
    True

    """
    check_status(C, lib.GrB_Matrix_select_Scalar(
        C[0],
        ffi.NULL if mask is None else mask[0],
        ffi.NULL if accum is None else accum,
        op,
        A[0],
        thunk[0],
        ffi.NULL if desc is None else desc[0],
    ))


def matrix_reduce(w, monoid, A, mask=None, accum=None, desc=None):
    """Reduce a matrix to a vector using a monoid (row-wise reduction).

    >>> from suitesparse_graphblas import vector
    >>> A = matrix_new(lib.GrB_INT64, 2, 2)
    >>> set_int64(A, 2, 0, 0)
    >>> set_int64(A, 3, 0, 1)
    >>> w = vector.vector_new(lib.GrB_INT64, 2)
    >>> matrix_reduce(w, lib.GrB_PLUS_MONOID_INT64, A)
    >>> vector.get_int64(w, 0) == 5
    True

    """
    check_status(w, lib.GrB_Matrix_reduce_Monoid(
        w[0],
        ffi.NULL if mask is None else mask[0],
        ffi.NULL if accum is None else accum,
        monoid,
        A[0],
        ffi.NULL if desc is None else desc[0],
    ))


def matrix_reduce_scalar(s, monoid, A, accum=None, desc=None):
    """Reduce a matrix to a scalar using a monoid.

    >>> from suitesparse_graphblas import scalar
    >>> A = matrix_new(lib.GrB_INT64, 2, 2)
    >>> set_int64(A, 2, 0, 0)
    >>> set_int64(A, 3, 1, 1)
    >>> s = scalar.scalar_new(lib.GrB_INT64)
    >>> matrix_reduce_scalar(s, lib.GrB_PLUS_MONOID_INT64, A)
    >>> scalar.get_int64(s) == 5
    True

    """
    check_status(s, lib.GrB_Matrix_reduce_Monoid_Scalar(
        s[0],
        ffi.NULL if accum is None else accum,
        monoid,
        A[0],
        ffi.NULL if desc is None else desc[0],
    ))


def matrix_transpose(C, A, mask=None, accum=None, desc=None):
    """Transpose a matrix.

    >>> C = matrix_new(lib.GrB_INT64, 3, 2)
    >>> A = matrix_new(lib.GrB_INT64, 2, 3)
    >>> set_int64(A, 7, 0, 2)
    >>> matrix_transpose(C, A)
    >>> get_int64(C, 2, 0) == 7
    True

    """
    check_status(C, lib.GrB_transpose(
        C[0],
        ffi.NULL if mask is None else mask[0],
        ffi.NULL if accum is None else accum,
        A[0],
        ffi.NULL if desc is None else desc[0],
    ))


def matrix_kronecker(C, op, A, B, mask=None, accum=None, desc=None):
    """Kronecker product of two matrices using a binary operator.

    >>> C = matrix_new(lib.GrB_INT64, 4, 4)
    >>> A = matrix_new(lib.GrB_INT64, 2, 2)
    >>> B = matrix_new(lib.GrB_INT64, 2, 2)
    >>> set_int64(A, 2, 0, 0)
    >>> set_int64(B, 3, 0, 0)
    >>> matrix_kronecker(C, lib.GrB_TIMES_INT64, A, B)
    >>> get_int64(C, 0, 0) == 6
    True

    """
    check_status(C, lib.GrB_Matrix_kronecker_BinaryOp(
        C[0],
        ffi.NULL if mask is None else mask[0],
        ffi.NULL if accum is None else accum,
        op,
        A[0],
        B[0],
        ffi.NULL if desc is None else desc[0],
    ))


def matrix_diag(C, v, k=0, desc=None):
    """Build a diagonal matrix from a vector.

    The vector ``v`` is placed on the ``k``-th diagonal of ``C``
    (``k=0`` is the main diagonal, positive is above, negative below).

    >>> from suitesparse_graphblas import vector
    >>> C = matrix_new(lib.GrB_INT64, 3, 3)
    >>> v = vector.vector_new(lib.GrB_INT64, 3)
    >>> vector.set_int64(v, 1, 0)
    >>> vector.set_int64(v, 2, 1)
    >>> vector.set_int64(v, 3, 2)
    >>> matrix_diag(C, v, 0)
    >>> get_int64(C, 0, 0) == 1
    True
    >>> get_int64(C, 1, 1) == 2
    True
    >>> get_int64(C, 2, 2) == 3
    True

    """
    check_status(C, lib.GxB_Matrix_diag(
        C[0], v[0], k,
        ffi.NULL if desc is None else desc[0],
    ))


def matrix_concat(C, tiles, m, n, desc=None):
    """Concatenate an m-by-n grid of tile matrices into C.

    ``tiles`` is a flat list of matrix handles in row-major order,
    representing an m-by-n grid of submatrices.

    >>> A = matrix_new(lib.GrB_INT64, 1, 2)
    >>> B = matrix_new(lib.GrB_INT64, 1, 2)
    >>> set_int64(A, 1, 0, 0)
    >>> set_int64(A, 2, 0, 1)
    >>> set_int64(B, 3, 0, 0)
    >>> set_int64(B, 4, 0, 1)
    >>> C = matrix_new(lib.GrB_INT64, 2, 2)
    >>> matrix_concat(C, [A, B], 2, 1)
    >>> get_int64(C, 0, 0) == 1
    True
    >>> get_int64(C, 1, 0) == 3
    True

    """
    tile_array = ffi.new("GrB_Matrix[]", [t[0] for t in tiles])
    check_status(C, lib.GxB_Matrix_concat(
        C[0], tile_array, m, n,
        ffi.NULL if desc is None else desc[0],
    ))


def matrix_assign(C, A, rows, nrows, cols, ncols, mask=None, accum=None, desc=None):
    """Assign a submatrix: C[rows, cols] = A.

    >>> C = matrix_new(lib.GrB_INT64, 3, 3)
    >>> A = matrix_new(lib.GrB_INT64, 2, 2)
    >>> set_int64(A, 10, 0, 0)
    >>> set_int64(A, 20, 1, 1)
    >>> rows = ffi.new("GrB_Index[2]", [0, 1])
    >>> cols = ffi.new("GrB_Index[2]", [0, 1])
    >>> matrix_assign(C, A, rows, 2, cols, 2)
    >>> get_int64(C, 0, 0) == 10
    True
    >>> get_int64(C, 1, 1) == 20
    True

    """
    check_status(C, lib.GrB_Matrix_assign(
        C[0],
        ffi.NULL if mask is None else mask[0],
        ffi.NULL if accum is None else accum,
        A[0],
        rows, nrows, cols, ncols,
        ffi.NULL if desc is None else desc[0],
    ))


def matrix_assign_scalar(C, x, rows, nrows, cols, ncols, mask=None, accum=None, desc=None):
    """Assign a scalar to a submatrix: C[rows, cols] = x.

    >>> from suitesparse_graphblas import scalar
    >>> C = matrix_new(lib.GrB_INT64, 3, 3)
    >>> s = scalar.scalar_new(lib.GrB_INT64)
    >>> scalar.set_int64(s, 99)
    >>> rows = ffi.new("GrB_Index[2]", [0, 1])
    >>> cols = ffi.new("GrB_Index[2]", [0, 1])
    >>> matrix_assign_scalar(C, s, rows, 2, cols, 2)
    >>> get_int64(C, 0, 0) == 99
    True
    >>> get_int64(C, 1, 1) == 99
    True

    """
    check_status(C, lib.GrB_Matrix_assign_Scalar(
        C[0],
        ffi.NULL if mask is None else mask[0],
        ffi.NULL if accum is None else accum,
        x[0],
        rows, nrows, cols, ncols,
        ffi.NULL if desc is None else desc[0],
    ))


def matrix_extract(C, A, rows, nrows, cols, ncols, mask=None, accum=None, desc=None):
    """Extract a submatrix: C = A[rows, cols].

    >>> A = matrix_new(lib.GrB_INT64, 3, 3)
    >>> set_int64(A, 42, 1, 2)
    >>> C = matrix_new(lib.GrB_INT64, 1, 1)
    >>> rows = ffi.new("GrB_Index[1]", [1])
    >>> cols = ffi.new("GrB_Index[1]", [2])
    >>> matrix_extract(C, A, rows, 1, cols, 1)
    >>> get_int64(C, 0, 0) == 42
    True

    """
    check_status(C, lib.GrB_Matrix_extract(
        C[0],
        ffi.NULL if mask is None else mask[0],
        ffi.NULL if accum is None else accum,
        A[0],
        rows, nrows, cols, ncols,
        ffi.NULL if desc is None else desc[0],
    ))


def matrix_col_extract(w, A, rows, nrows, j, mask=None, accum=None, desc=None):
    """Extract a column from a matrix into a vector: w = A[rows, j].

    >>> from suitesparse_graphblas import vector
    >>> A = matrix_new(lib.GrB_INT64, 3, 3)
    >>> set_int64(A, 42, 1, 2)
    >>> w = vector.vector_new(lib.GrB_INT64, 3)
    >>> matrix_col_extract(w, A, lib.GrB_ALL, 3, 2)
    >>> vector.get_int64(w, 1) == 42
    True

    """
    check_status(w, lib.GrB_Col_extract(
        w[0],
        ffi.NULL if mask is None else mask[0],
        ffi.NULL if accum is None else accum,
        A[0],
        rows, nrows, j,
        ffi.NULL if desc is None else desc[0],
    ))


def matrix_col_assign(C, u, rows, nrows, j, mask=None, accum=None, desc=None):
    """Assign a vector to a column of a matrix: C[rows, j] = u.

    >>> from suitesparse_graphblas import vector
    >>> C = matrix_new(lib.GrB_INT64, 3, 3)
    >>> u = vector.vector_new(lib.GrB_INT64, 3)
    >>> vector.set_int64(u, 42, 1)
    >>> matrix_col_assign(C, u, lib.GrB_ALL, 3, 2)
    >>> get_int64(C, 1, 2) == 42
    True

    """
    check_status(C, lib.GrB_Col_assign(
        C[0],
        ffi.NULL if mask is None else mask[0],
        ffi.NULL if accum is None else accum,
        u[0],
        rows, nrows, j,
        ffi.NULL if desc is None else desc[0],
    ))


def matrix_row_assign(C, u, i, cols, ncols, mask=None, accum=None, desc=None):
    """Assign a vector to a row of a matrix: C[i, cols] = u.

    >>> from suitesparse_graphblas import vector
    >>> C = matrix_new(lib.GrB_INT64, 3, 3)
    >>> u = vector.vector_new(lib.GrB_INT64, 3)
    >>> vector.set_int64(u, 42, 1)
    >>> matrix_row_assign(C, u, 2, lib.GrB_ALL, 3)
    >>> get_int64(C, 2, 1) == 42
    True

    """
    check_status(C, lib.GrB_Row_assign(
        C[0],
        ffi.NULL if mask is None else mask[0],
        ffi.NULL if accum is None else accum,
        u[0],
        i, cols, ncols,
        ffi.NULL if desc is None else desc[0],
    ))


def matrix_build_bool(C, rows, cols, vals, nvals, dup):
    """Build a matrix from COO arrays of boolean values.

    >>> C = matrix_new(lib.GrB_BOOL, 2, 2)
    >>> rows = np.array([0], dtype=np.uint64)
    >>> cols = np.array([0], dtype=np.uint64)
    >>> vals = np.array([True], dtype=np.bool_)
    >>> matrix_build_bool(C, rows, cols, vals, 1, lib.GrB_FIRST_BOOL)
    >>> get_bool(C, 0, 0) == True
    True

    """
    check_status(C, lib.GrB_Matrix_build_BOOL(
        C[0],
        ffi.cast("GrB_Index*", ffi.from_buffer(rows)),
        ffi.cast("GrB_Index*", ffi.from_buffer(cols)),
        ffi.cast("bool*", ffi.from_buffer(vals)),
        nvals, dup,
    ))


def matrix_build_int8(C, rows, cols, vals, nvals, dup):
    """Build a matrix from COO arrays of int8 values.

    >>> C = matrix_new(lib.GrB_INT8, 2, 2)
    >>> rows = np.array([0], dtype=np.uint64)
    >>> cols = np.array([0], dtype=np.uint64)
    >>> vals = np.array([7], dtype=np.int8)
    >>> matrix_build_int8(C, rows, cols, vals, 1, lib.GrB_PLUS_INT8)
    >>> get_int8(C, 0, 0) == 7
    True

    """
    check_status(C, lib.GrB_Matrix_build_INT8(
        C[0],
        ffi.cast("GrB_Index*", ffi.from_buffer(rows)),
        ffi.cast("GrB_Index*", ffi.from_buffer(cols)),
        ffi.cast("int8_t*", ffi.from_buffer(vals)),
        nvals, dup,
    ))


def matrix_build_int16(C, rows, cols, vals, nvals, dup):
    """Build a matrix from COO arrays of int16 values.

    >>> C = matrix_new(lib.GrB_INT16, 2, 2)
    >>> rows = np.array([0], dtype=np.uint64)
    >>> cols = np.array([0], dtype=np.uint64)
    >>> vals = np.array([7], dtype=np.int16)
    >>> matrix_build_int16(C, rows, cols, vals, 1, lib.GrB_PLUS_INT16)
    >>> get_int16(C, 0, 0) == 7
    True

    """
    check_status(C, lib.GrB_Matrix_build_INT16(
        C[0],
        ffi.cast("GrB_Index*", ffi.from_buffer(rows)),
        ffi.cast("GrB_Index*", ffi.from_buffer(cols)),
        ffi.cast("int16_t*", ffi.from_buffer(vals)),
        nvals, dup,
    ))


def matrix_build_int32(C, rows, cols, vals, nvals, dup):
    """Build a matrix from COO arrays of int32 values.

    >>> C = matrix_new(lib.GrB_INT32, 2, 2)
    >>> rows = np.array([0], dtype=np.uint64)
    >>> cols = np.array([0], dtype=np.uint64)
    >>> vals = np.array([7], dtype=np.int32)
    >>> matrix_build_int32(C, rows, cols, vals, 1, lib.GrB_PLUS_INT32)
    >>> get_int32(C, 0, 0) == 7
    True

    """
    check_status(C, lib.GrB_Matrix_build_INT32(
        C[0],
        ffi.cast("GrB_Index*", ffi.from_buffer(rows)),
        ffi.cast("GrB_Index*", ffi.from_buffer(cols)),
        ffi.cast("int32_t*", ffi.from_buffer(vals)),
        nvals, dup,
    ))


def matrix_build_int64(C, rows, cols, vals, nvals, dup):
    """Build a matrix from COO arrays of int64 values.

    >>> C = matrix_new(lib.GrB_INT64, 2, 2)
    >>> rows = np.array([0], dtype=np.uint64)
    >>> cols = np.array([0], dtype=np.uint64)
    >>> vals = np.array([7], dtype=np.int64)
    >>> matrix_build_int64(C, rows, cols, vals, 1, lib.GrB_PLUS_INT64)
    >>> get_int64(C, 0, 0) == 7
    True

    """
    check_status(C, lib.GrB_Matrix_build_INT64(
        C[0],
        ffi.cast("GrB_Index*", ffi.from_buffer(rows)),
        ffi.cast("GrB_Index*", ffi.from_buffer(cols)),
        ffi.cast("int64_t*", ffi.from_buffer(vals)),
        nvals, dup,
    ))


def matrix_build_uint8(C, rows, cols, vals, nvals, dup):
    """Build a matrix from COO arrays of uint8 values.

    >>> C = matrix_new(lib.GrB_UINT8, 2, 2)
    >>> rows = np.array([0], dtype=np.uint64)
    >>> cols = np.array([0], dtype=np.uint64)
    >>> vals = np.array([7], dtype=np.uint8)
    >>> matrix_build_uint8(C, rows, cols, vals, 1, lib.GrB_PLUS_UINT8)
    >>> get_uint8(C, 0, 0) == 7
    True

    """
    check_status(C, lib.GrB_Matrix_build_UINT8(
        C[0],
        ffi.cast("GrB_Index*", ffi.from_buffer(rows)),
        ffi.cast("GrB_Index*", ffi.from_buffer(cols)),
        ffi.cast("uint8_t*", ffi.from_buffer(vals)),
        nvals, dup,
    ))


def matrix_build_uint16(C, rows, cols, vals, nvals, dup):
    """Build a matrix from COO arrays of uint16 values.

    >>> C = matrix_new(lib.GrB_UINT16, 2, 2)
    >>> rows = np.array([0], dtype=np.uint64)
    >>> cols = np.array([0], dtype=np.uint64)
    >>> vals = np.array([7], dtype=np.uint16)
    >>> matrix_build_uint16(C, rows, cols, vals, 1, lib.GrB_PLUS_UINT16)
    >>> get_uint16(C, 0, 0) == 7
    True

    """
    check_status(C, lib.GrB_Matrix_build_UINT16(
        C[0],
        ffi.cast("GrB_Index*", ffi.from_buffer(rows)),
        ffi.cast("GrB_Index*", ffi.from_buffer(cols)),
        ffi.cast("uint16_t*", ffi.from_buffer(vals)),
        nvals, dup,
    ))


def matrix_build_uint32(C, rows, cols, vals, nvals, dup):
    """Build a matrix from COO arrays of uint32 values.

    >>> C = matrix_new(lib.GrB_UINT32, 2, 2)
    >>> rows = np.array([0], dtype=np.uint64)
    >>> cols = np.array([0], dtype=np.uint64)
    >>> vals = np.array([7], dtype=np.uint32)
    >>> matrix_build_uint32(C, rows, cols, vals, 1, lib.GrB_PLUS_UINT32)
    >>> get_uint32(C, 0, 0) == 7
    True

    """
    check_status(C, lib.GrB_Matrix_build_UINT32(
        C[0],
        ffi.cast("GrB_Index*", ffi.from_buffer(rows)),
        ffi.cast("GrB_Index*", ffi.from_buffer(cols)),
        ffi.cast("uint32_t*", ffi.from_buffer(vals)),
        nvals, dup,
    ))


def matrix_build_uint64(C, rows, cols, vals, nvals, dup):
    """Build a matrix from COO arrays of uint64 values.

    >>> C = matrix_new(lib.GrB_UINT64, 2, 2)
    >>> rows = np.array([0], dtype=np.uint64)
    >>> cols = np.array([0], dtype=np.uint64)
    >>> vals = np.array([7], dtype=np.uint64)
    >>> matrix_build_uint64(C, rows, cols, vals, 1, lib.GrB_PLUS_UINT64)
    >>> get_uint64(C, 0, 0) == 7
    True

    """
    check_status(C, lib.GrB_Matrix_build_UINT64(
        C[0],
        ffi.cast("GrB_Index*", ffi.from_buffer(rows)),
        ffi.cast("GrB_Index*", ffi.from_buffer(cols)),
        ffi.cast("uint64_t*", ffi.from_buffer(vals)),
        nvals, dup,
    ))


def matrix_build_fp32(C, rows, cols, vals, nvals, dup):
    """Build a matrix from COO arrays of fp32 values.

    >>> C = matrix_new(lib.GrB_FP32, 2, 2)
    >>> rows = np.array([0], dtype=np.uint64)
    >>> cols = np.array([0], dtype=np.uint64)
    >>> vals = np.array([1.5], dtype=np.float32)
    >>> matrix_build_fp32(C, rows, cols, vals, 1, lib.GrB_PLUS_FP32)
    >>> get_fp32(C, 0, 0) == 1.5
    True

    """
    check_status(C, lib.GrB_Matrix_build_FP32(
        C[0],
        ffi.cast("GrB_Index*", ffi.from_buffer(rows)),
        ffi.cast("GrB_Index*", ffi.from_buffer(cols)),
        ffi.cast("float*", ffi.from_buffer(vals)),
        nvals, dup,
    ))


def matrix_build_fp64(C, rows, cols, vals, nvals, dup):
    """Build a matrix from COO arrays of fp64 values.

    >>> C = matrix_new(lib.GrB_FP64, 2, 2)
    >>> rows = np.array([0], dtype=np.uint64)
    >>> cols = np.array([0], dtype=np.uint64)
    >>> vals = np.array([1.5], dtype=np.float64)
    >>> matrix_build_fp64(C, rows, cols, vals, 1, lib.GrB_PLUS_FP64)
    >>> get_fp64(C, 0, 0) == 1.5
    True

    """
    check_status(C, lib.GrB_Matrix_build_FP64(
        C[0],
        ffi.cast("GrB_Index*", ffi.from_buffer(rows)),
        ffi.cast("GrB_Index*", ffi.from_buffer(cols)),
        ffi.cast("double*", ffi.from_buffer(vals)),
        nvals, dup,
    ))


def matrix_extract_tuples_int64(A):
    """Extract all tuples from a matrix as numpy arrays.

    Returns (rows, cols, vals) as numpy arrays.

    >>> A = matrix_new(lib.GrB_INT64, 2, 2)
    >>> set_int64(A, 42, 0, 0)
    >>> rows, cols, vals = matrix_extract_tuples_int64(A)
    >>> len(vals) == 1
    True
    >>> int(vals[0]) == 42
    True

    """
    n = matrix_nvals(A)
    rows = np.empty(n, dtype=np.uint64)
    cols = np.empty(n, dtype=np.uint64)
    vals = np.empty(n, dtype=np.int64)
    nvals_p = ffi.new("GrB_Index*", n)
    if n > 0:
        check_status(A, lib.GrB_Matrix_extractTuples_INT64(
            ffi.cast("GrB_Index*", ffi.from_buffer(rows)),
            ffi.cast("GrB_Index*", ffi.from_buffer(cols)),
            ffi.cast("int64_t*", ffi.from_buffer(vals)),
            nvals_p, A[0],
        ))
    return rows, cols, vals


def matrix_extract_tuples_fp64(A):
    """Extract all tuples from a matrix as numpy arrays.

    Returns (rows, cols, vals) as numpy arrays.

    >>> A = matrix_new(lib.GrB_FP64, 2, 2)
    >>> set_fp64(A, 1.5, 0, 0)
    >>> rows, cols, vals = matrix_extract_tuples_fp64(A)
    >>> len(vals) == 1
    True
    >>> float(vals[0]) == 1.5
    True

    """
    n = matrix_nvals(A)
    rows = np.empty(n, dtype=np.uint64)
    cols = np.empty(n, dtype=np.uint64)
    vals = np.empty(n, dtype=np.float64)
    nvals_p = ffi.new("GrB_Index*", n)
    if n > 0:
        check_status(A, lib.GrB_Matrix_extractTuples_FP64(
            ffi.cast("GrB_Index*", ffi.from_buffer(rows)),
            ffi.cast("GrB_Index*", ffi.from_buffer(cols)),
            ffi.cast("double*", ffi.from_buffer(vals)),
            nvals_p, A[0],
        ))
    return rows, cols, vals

def matrix_extract_tuples_bool(A):
    """Extract all tuples from a matrix as numpy arrays.

    Returns (rows, cols, vals) as numpy arrays.

    >>> A = matrix_new(lib.GrB_BOOL, 2, 2)
    >>> set_bool(A, True, 0, 0)
    >>> rows, cols, vals = matrix_extract_tuples_bool(A)
    >>> len(vals) == 1
    True
    >>> bool(vals[0]) == True
    True

    """
    n = matrix_nvals(A)
    rows = np.empty(n, dtype=np.uint64)
    cols = np.empty(n, dtype=np.uint64)
    vals = np.empty(n, dtype=np.bool_)
    nvals_p = ffi.new("GrB_Index*", n)
    if n > 0:
        check_status(A, lib.GrB_Matrix_extractTuples_BOOL(
            ffi.cast("GrB_Index*", ffi.from_buffer(rows)),
            ffi.cast("GrB_Index*", ffi.from_buffer(cols)),
            ffi.cast("bool*", ffi.from_buffer(vals)),
            nvals_p, A[0],
        ))
    return rows, cols, vals

def matrix_extract_tuples_int8(A):
    """Extract all tuples from a matrix as numpy arrays.

    Returns (rows, cols, vals) as numpy arrays.

    >>> A = matrix_new(lib.GrB_INT8, 2, 2)
    >>> set_int8(A, 7, 0, 0)
    >>> rows, cols, vals = matrix_extract_tuples_int8(A)
    >>> len(vals) == 1
    True
    >>> int(vals[0]) == 7
    True

    """
    n = matrix_nvals(A)
    rows = np.empty(n, dtype=np.uint64)
    cols = np.empty(n, dtype=np.uint64)
    vals = np.empty(n, dtype=np.int8)
    nvals_p = ffi.new("GrB_Index*", n)
    if n > 0:
        check_status(A, lib.GrB_Matrix_extractTuples_INT8(
            ffi.cast("GrB_Index*", ffi.from_buffer(rows)),
            ffi.cast("GrB_Index*", ffi.from_buffer(cols)),
            ffi.cast("int8_t*", ffi.from_buffer(vals)),
            nvals_p, A[0],
        ))
    return rows, cols, vals

def matrix_extract_tuples_int16(A):
    """Extract all tuples from a matrix as numpy arrays.

    Returns (rows, cols, vals) as numpy arrays.

    >>> A = matrix_new(lib.GrB_INT16, 2, 2)
    >>> set_int16(A, 7, 0, 0)
    >>> rows, cols, vals = matrix_extract_tuples_int16(A)
    >>> len(vals) == 1
    True
    >>> int(vals[0]) == 7
    True

    """
    n = matrix_nvals(A)
    rows = np.empty(n, dtype=np.uint64)
    cols = np.empty(n, dtype=np.uint64)
    vals = np.empty(n, dtype=np.int16)
    nvals_p = ffi.new("GrB_Index*", n)
    if n > 0:
        check_status(A, lib.GrB_Matrix_extractTuples_INT16(
            ffi.cast("GrB_Index*", ffi.from_buffer(rows)),
            ffi.cast("GrB_Index*", ffi.from_buffer(cols)),
            ffi.cast("int16_t*", ffi.from_buffer(vals)),
            nvals_p, A[0],
        ))
    return rows, cols, vals

def matrix_extract_tuples_int32(A):
    """Extract all tuples from a matrix as numpy arrays.

    Returns (rows, cols, vals) as numpy arrays.

    >>> A = matrix_new(lib.GrB_INT32, 2, 2)
    >>> set_int32(A, 7, 0, 0)
    >>> rows, cols, vals = matrix_extract_tuples_int32(A)
    >>> len(vals) == 1
    True
    >>> int(vals[0]) == 7
    True

    """
    n = matrix_nvals(A)
    rows = np.empty(n, dtype=np.uint64)
    cols = np.empty(n, dtype=np.uint64)
    vals = np.empty(n, dtype=np.int32)
    nvals_p = ffi.new("GrB_Index*", n)
    if n > 0:
        check_status(A, lib.GrB_Matrix_extractTuples_INT32(
            ffi.cast("GrB_Index*", ffi.from_buffer(rows)),
            ffi.cast("GrB_Index*", ffi.from_buffer(cols)),
            ffi.cast("int32_t*", ffi.from_buffer(vals)),
            nvals_p, A[0],
        ))
    return rows, cols, vals

def matrix_extract_tuples_uint8(A):
    """Extract all tuples from a matrix as numpy arrays.

    Returns (rows, cols, vals) as numpy arrays.

    >>> A = matrix_new(lib.GrB_UINT8, 2, 2)
    >>> set_uint8(A, 7, 0, 0)
    >>> rows, cols, vals = matrix_extract_tuples_uint8(A)
    >>> len(vals) == 1
    True
    >>> int(vals[0]) == 7
    True

    """
    n = matrix_nvals(A)
    rows = np.empty(n, dtype=np.uint64)
    cols = np.empty(n, dtype=np.uint64)
    vals = np.empty(n, dtype=np.uint8)
    nvals_p = ffi.new("GrB_Index*", n)
    if n > 0:
        check_status(A, lib.GrB_Matrix_extractTuples_UINT8(
            ffi.cast("GrB_Index*", ffi.from_buffer(rows)),
            ffi.cast("GrB_Index*", ffi.from_buffer(cols)),
            ffi.cast("uint8_t*", ffi.from_buffer(vals)),
            nvals_p, A[0],
        ))
    return rows, cols, vals

def matrix_extract_tuples_uint16(A):
    """Extract all tuples from a matrix as numpy arrays.

    Returns (rows, cols, vals) as numpy arrays.

    >>> A = matrix_new(lib.GrB_UINT16, 2, 2)
    >>> set_uint16(A, 7, 0, 0)
    >>> rows, cols, vals = matrix_extract_tuples_uint16(A)
    >>> len(vals) == 1
    True
    >>> int(vals[0]) == 7
    True

    """
    n = matrix_nvals(A)
    rows = np.empty(n, dtype=np.uint64)
    cols = np.empty(n, dtype=np.uint64)
    vals = np.empty(n, dtype=np.uint16)
    nvals_p = ffi.new("GrB_Index*", n)
    if n > 0:
        check_status(A, lib.GrB_Matrix_extractTuples_UINT16(
            ffi.cast("GrB_Index*", ffi.from_buffer(rows)),
            ffi.cast("GrB_Index*", ffi.from_buffer(cols)),
            ffi.cast("uint16_t*", ffi.from_buffer(vals)),
            nvals_p, A[0],
        ))
    return rows, cols, vals

def matrix_extract_tuples_uint32(A):
    """Extract all tuples from a matrix as numpy arrays.

    Returns (rows, cols, vals) as numpy arrays.

    >>> A = matrix_new(lib.GrB_UINT32, 2, 2)
    >>> set_uint32(A, 7, 0, 0)
    >>> rows, cols, vals = matrix_extract_tuples_uint32(A)
    >>> len(vals) == 1
    True
    >>> int(vals[0]) == 7
    True

    """
    n = matrix_nvals(A)
    rows = np.empty(n, dtype=np.uint64)
    cols = np.empty(n, dtype=np.uint64)
    vals = np.empty(n, dtype=np.uint32)
    nvals_p = ffi.new("GrB_Index*", n)
    if n > 0:
        check_status(A, lib.GrB_Matrix_extractTuples_UINT32(
            ffi.cast("GrB_Index*", ffi.from_buffer(rows)),
            ffi.cast("GrB_Index*", ffi.from_buffer(cols)),
            ffi.cast("uint32_t*", ffi.from_buffer(vals)),
            nvals_p, A[0],
        ))
    return rows, cols, vals

def matrix_extract_tuples_uint64(A):
    """Extract all tuples from a matrix as numpy arrays.

    Returns (rows, cols, vals) as numpy arrays.

    >>> A = matrix_new(lib.GrB_UINT64, 2, 2)
    >>> set_uint64(A, 7, 0, 0)
    >>> rows, cols, vals = matrix_extract_tuples_uint64(A)
    >>> len(vals) == 1
    True
    >>> int(vals[0]) == 7
    True

    """
    n = matrix_nvals(A)
    rows = np.empty(n, dtype=np.uint64)
    cols = np.empty(n, dtype=np.uint64)
    vals = np.empty(n, dtype=np.uint64)
    nvals_p = ffi.new("GrB_Index*", n)
    if n > 0:
        check_status(A, lib.GrB_Matrix_extractTuples_UINT64(
            ffi.cast("GrB_Index*", ffi.from_buffer(rows)),
            ffi.cast("GrB_Index*", ffi.from_buffer(cols)),
            ffi.cast("uint64_t*", ffi.from_buffer(vals)),
            nvals_p, A[0],
        ))
    return rows, cols, vals

def matrix_extract_tuples_fp32(A):
    """Extract all tuples from a matrix as numpy arrays.

    Returns (rows, cols, vals) as numpy arrays.

    >>> A = matrix_new(lib.GrB_FP32, 2, 2)
    >>> set_fp32(A, 1.5, 0, 0)
    >>> rows, cols, vals = matrix_extract_tuples_fp32(A)
    >>> len(vals) == 1
    True
    >>> float(vals[0]) == 1.5
    True

    """
    n = matrix_nvals(A)
    rows = np.empty(n, dtype=np.uint64)
    cols = np.empty(n, dtype=np.uint64)
    vals = np.empty(n, dtype=np.float32)
    nvals_p = ffi.new("GrB_Index*", n)
    if n > 0:
        check_status(A, lib.GrB_Matrix_extractTuples_FP32(
            ffi.cast("GrB_Index*", ffi.from_buffer(rows)),
            ffi.cast("GrB_Index*", ffi.from_buffer(cols)),
            ffi.cast("float*", ffi.from_buffer(vals)),
            nvals_p, A[0],
        ))
    return rows, cols, vals


def set_bool(A, value, i, j):
    """Set a boolean value to the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_BOOL, 3, 3)
    >>> set_bool(A, True, 2, 2)
    >>> get_bool(A, 2, 2) == True
    True

    """
    check_status(A, lib.GrB_Matrix_setElement_BOOL(A[0], value, i, j))


def get_bool(A, i, j):
    """Get a boolean value from the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_BOOL, 3, 3)
    >>> set_bool(A, True, 2, 2)
    >>> get_bool(A, 2, 2) == True
    True

    """
    value = ffi.new("bool*")
    check_status(A, lib.GrB_Matrix_extractElement_BOOL(value, A[0], i, j))
    return value[0]


def set_int8(A, value, i, j):
    """Set an int8 value to the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_INT8, 3, 3)
    >>> set_int8(A, 7, 2, 2)
    >>> get_int8(A, 2, 2) == 7
    True

    """
    check_status(A, lib.GrB_Matrix_setElement_INT8(A[0], value, i, j))


def get_int8(A, i, j):
    """Get an int8 value from the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_INT8, 3, 3)
    >>> set_int8(A, 7, 2, 2)
    >>> get_int8(A, 2, 2) == 7
    True

    """
    value = ffi.new("int8_t*")
    check_status(A, lib.GrB_Matrix_extractElement_INT8(value, A[0], i, j))
    return value[0]


def set_int16(A, value, i, j):
    """Set an int16 value to the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_INT16, 3, 3)
    >>> set_int16(A, 7, 2, 2)
    >>> get_int16(A, 2, 2) == 7
    True

    """
    check_status(A, lib.GrB_Matrix_setElement_INT16(A[0], value, i, j))


def get_int16(A, i, j):
    """Get an int16 value from the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_INT16, 3, 3)
    >>> set_int16(A, 7, 2, 2)
    >>> get_int16(A, 2, 2) == 7
    True

    """
    value = ffi.new("int16_t*")
    check_status(A, lib.GrB_Matrix_extractElement_INT16(value, A[0], i, j))
    return value[0]


def set_int32(A, value, i, j):
    """Set an int32 value to the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_INT32, 3, 3)
    >>> set_int32(A, 7, 2, 2)
    >>> get_int32(A, 2, 2) == 7
    True

    """
    check_status(A, lib.GrB_Matrix_setElement_INT32(A[0], value, i, j))


def get_int32(A, i, j):
    """Get an int32 value from the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_INT32, 3, 3)
    >>> set_int32(A, 7, 2, 2)
    >>> get_int32(A, 2, 2) == 7
    True

    """
    value = ffi.new("int32_t*")
    check_status(A, lib.GrB_Matrix_extractElement_INT32(value, A[0], i, j))
    return value[0]


def set_int64(A, value, i, j):
    """Set an int64 value to the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_INT64, 3, 3)
    >>> set_int64(A, 7, 2, 2)
    >>> get_int64(A, 2, 2) == 7
    True

    """
    check_status(A, lib.GrB_Matrix_setElement_INT64(A[0], value, i, j))


def get_int64(A, i, j):
    """Get an int64 value from the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_INT64, 3, 3)
    >>> set_int64(A, 7, 2, 2)
    >>> get_int64(A, 2, 2) == 7
    True

    """
    value = ffi.new("int64_t*")
    check_status(A, lib.GrB_Matrix_extractElement_INT64(value, A[0], i, j))
    return value[0]


def set_uint8(A, value, i, j):
    """Set a uint8 value to the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_UINT8, 3, 3)
    >>> set_uint8(A, 7, 2, 2)
    >>> get_uint8(A, 2, 2) == 7
    True

    """
    check_status(A, lib.GrB_Matrix_setElement_UINT8(A[0], value, i, j))


def get_uint8(A, i, j):
    """Get a uint8 value from the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_UINT8, 3, 3)
    >>> set_uint8(A, 7, 2, 2)
    >>> get_uint8(A, 2, 2) == 7
    True

    """
    value = ffi.new("uint8_t*")
    check_status(A, lib.GrB_Matrix_extractElement_UINT8(value, A[0], i, j))
    return value[0]


def set_uint16(A, value, i, j):
    """Set a uint16 value to the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_UINT16, 3, 3)
    >>> set_uint16(A, 7, 2, 2)
    >>> get_uint16(A, 2, 2) == 7
    True

    """
    check_status(A, lib.GrB_Matrix_setElement_UINT16(A[0], value, i, j))


def get_uint16(A, i, j):
    """Get a uint16 value from the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_UINT16, 3, 3)
    >>> set_uint16(A, 7, 2, 2)
    >>> get_uint16(A, 2, 2) == 7
    True

    """
    value = ffi.new("uint16_t*")
    check_status(A, lib.GrB_Matrix_extractElement_UINT16(value, A[0], i, j))
    return value[0]


def set_uint32(A, value, i, j):
    """Set a uint32 value to the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_UINT32, 3, 3)
    >>> set_uint32(A, 7, 2, 2)
    >>> get_uint32(A, 2, 2) == 7
    True

    """
    check_status(A, lib.GrB_Matrix_setElement_UINT32(A[0], value, i, j))


def get_uint32(A, i, j):
    """Get a uint32 value from the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_UINT32, 3, 3)
    >>> set_uint32(A, 7, 2, 2)
    >>> get_uint32(A, 2, 2) == 7
    True

    """
    value = ffi.new("uint32_t*")
    check_status(A, lib.GrB_Matrix_extractElement_UINT32(value, A[0], i, j))
    return value[0]


def set_uint64(A, value, i, j):
    """Set a uint64 value to the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_UINT64, 3, 3)
    >>> set_uint64(A, 7, 2, 2)
    >>> get_uint64(A, 2, 2) == 7
    True

    """
    check_status(A, lib.GrB_Matrix_setElement_UINT64(A[0], value, i, j))


def get_uint64(A, i, j):
    """Get a uint64 value from the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_UINT64, 3, 3)
    >>> set_uint64(A, 7, 2, 2)
    >>> get_uint64(A, 2, 2) == 7
    True

    """
    value = ffi.new("uint64_t*")
    check_status(A, lib.GrB_Matrix_extractElement_UINT64(value, A[0], i, j))
    return value[0]


def set_fp32(A, value, i, j):
    """Set an fp32 value to the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_FP32, 3, 3)
    >>> set_fp32(A, 1.5, 2, 2)
    >>> get_fp32(A, 2, 2) == 1.5
    True

    """
    check_status(A, lib.GrB_Matrix_setElement_FP32(A[0], value, i, j))


def get_fp32(A, i, j):
    """Get an fp32 value from the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_FP32, 3, 3)
    >>> set_fp32(A, 1.5, 2, 2)
    >>> get_fp32(A, 2, 2) == 1.5
    True

    """
    value = ffi.new("float*")
    check_status(A, lib.GrB_Matrix_extractElement_FP32(value, A[0], i, j))
    return value[0]


def set_fp64(A, value, i, j):
    """Set an fp64 value to the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_FP64, 3, 3)
    >>> set_fp64(A, 1.5, 2, 2)
    >>> get_fp64(A, 2, 2) == 1.5
    True

    """
    check_status(A, lib.GrB_Matrix_setElement_FP64(A[0], value, i, j))


def get_fp64(A, i, j):
    """Get an fp64 value from the matrix at row `i` column `j`.

    >>> A = matrix_new(lib.GrB_FP64, 3, 3)
    >>> set_fp64(A, 1.5, 2, 2)
    >>> get_fp64(A, 2, 2) == 1.5
    True

    """
    value = ffi.new("double*")
    check_status(A, lib.GrB_Matrix_extractElement_FP64(value, A[0], i, j))
    return value[0]


if supports_complex():

    def set_fc32(A, value, i, j):
        """Set an fc32 value to the matrix at row `i` column `j`.

        >>> A = matrix_new(lib.GxB_FC32, 3, 3)
        >>> set_fc32(A, 2+3j, 2, 2)
        >>> get_fc32(A, 2, 2) == 2+3j
        True

        """
        check_status(A, lib.GxB_Matrix_setElement_FC32(A[0], value, i, j))

    def get_fc32(A, i, j):
        """Get an fc32 value from the matrix at row `i` column `j`.

        >>> A = matrix_new(lib.GxB_FC32, 3, 3)
        >>> set_fc32(A, 2+3j, 2, 2)
        >>> get_fc32(A, 2, 2) == 2+3j
        True

        """
        value = ffi.new("GxB_FC32_t*")
        check_status(A, lib.GxB_Matrix_extractElement_FC32(value, A[0], i, j))
        return value[0]

    def set_fc64(A, value, i, j):
        """Set an fc64 value to the matrix at row `i` column `j`.

        >>> A = matrix_new(lib.GxB_FC64, 3, 3)
        >>> set_fc64(A, 2+3j, 2, 2)
        >>> get_fc64(A, 2, 2) == 2+3j
        True

        """
        check_status(A, lib.GxB_Matrix_setElement_FC64(A[0], value, i, j))

    def get_fc64(A, i, j):
        """Get an fc64 value from the matrix at row `i` column `j`.

        >>> A = matrix_new(lib.GxB_FC64, 3, 3)
        >>> set_fc64(A, 2+3j, 2, 2)
        >>> get_fc64(A, 2, 2) == 2+3j
        True

        """
        value = ffi.new("GxB_FC64_t*")
        check_status(A, lib.GxB_Matrix_extractElement_FC64(value, A[0], i, j))
        return value[0]

    def matrix_build_fc32(C, rows, cols, vals, nvals, dup):
        """Build a matrix from COO arrays of fc32 values.

        >>> C = matrix_new(lib.GxB_FC32, 2, 2)
        >>> rows = np.array([0], dtype=np.uint64)
        >>> cols = np.array([0], dtype=np.uint64)
        >>> vals = np.array([2+3j], dtype=np.complex64)
        >>> matrix_build_fc32(C, rows, cols, vals, 1, lib.GxB_PLUS_FC32)
        >>> get_fc32(C, 0, 0) == 2+3j
        True

        """
        check_status(C, lib.GxB_Matrix_build_FC32(
            C[0],
            ffi.cast("GrB_Index*", ffi.from_buffer(rows)),
            ffi.cast("GrB_Index*", ffi.from_buffer(cols)),
            ffi.cast("GxB_FC32_t*", ffi.from_buffer(vals)),
            nvals, dup,
        ))

    def matrix_extract_tuples_fc32(A):
        """Extract all tuples from a matrix as numpy arrays.

        Returns (rows, cols, vals) as numpy arrays.

        >>> A = matrix_new(lib.GxB_FC32, 2, 2)
        >>> set_fc32(A, 2+3j, 0, 0)
        >>> rows, cols, vals = matrix_extract_tuples_fc32(A)
        >>> len(vals) == 1
        True
        >>> complex(vals[0]) == 2+3j
        True

        """
        n = matrix_nvals(A)
        rows = np.empty(n, dtype=np.uint64)
        cols = np.empty(n, dtype=np.uint64)
        vals = np.empty(n, dtype=np.complex64)
        nvals_p = ffi.new("GrB_Index*", n)
        if n > 0:
            check_status(A, lib.GxB_Matrix_extractTuples_FC32(
                ffi.cast("GrB_Index*", ffi.from_buffer(rows)),
                ffi.cast("GrB_Index*", ffi.from_buffer(cols)),
                ffi.cast("GxB_FC32_t*", ffi.from_buffer(vals)),
                nvals_p, A[0],
            ))
        return rows, cols, vals

    def matrix_build_fc64(C, rows, cols, vals, nvals, dup):
        """Build a matrix from COO arrays of fc64 values.

        >>> C = matrix_new(lib.GxB_FC64, 2, 2)
        >>> rows = np.array([0], dtype=np.uint64)
        >>> cols = np.array([0], dtype=np.uint64)
        >>> vals = np.array([2+3j], dtype=np.complex128)
        >>> matrix_build_fc64(C, rows, cols, vals, 1, lib.GxB_PLUS_FC64)
        >>> get_fc64(C, 0, 0) == 2+3j
        True

        """
        check_status(C, lib.GxB_Matrix_build_FC64(
            C[0],
            ffi.cast("GrB_Index*", ffi.from_buffer(rows)),
            ffi.cast("GrB_Index*", ffi.from_buffer(cols)),
            ffi.cast("GxB_FC64_t*", ffi.from_buffer(vals)),
            nvals, dup,
        ))

    def matrix_extract_tuples_fc64(A):
        """Extract all tuples from a matrix as numpy arrays.

        Returns (rows, cols, vals) as numpy arrays.

        >>> A = matrix_new(lib.GxB_FC64, 2, 2)
        >>> set_fc64(A, 2+3j, 0, 0)
        >>> rows, cols, vals = matrix_extract_tuples_fc64(A)
        >>> len(vals) == 1
        True
        >>> complex(vals[0]) == 2+3j
        True

        """
        n = matrix_nvals(A)
        rows = np.empty(n, dtype=np.uint64)
        cols = np.empty(n, dtype=np.uint64)
        vals = np.empty(n, dtype=np.complex128)
        nvals_p = ffi.new("GrB_Index*", n)
        if n > 0:
            check_status(A, lib.GxB_Matrix_extractTuples_FC64(
                ffi.cast("GrB_Index*", ffi.from_buffer(rows)),
                ffi.cast("GrB_Index*", ffi.from_buffer(cols)),
                ffi.cast("GxB_FC64_t*", ffi.from_buffer(vals)),
                nvals_p, A[0],
            ))
        return rows, cols, vals
