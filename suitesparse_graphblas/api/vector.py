"""Create, manipulate, and query GrB_Vector objects."""

import numpy as np

from suitesparse_graphblas import check_status, ffi, lib, supports_complex

from .utils import _capture_c_output  # noqa: F401

from .io.serialize import deserialize_vector as deserialize  # noqa: F401
from .io.serialize import serialize_vector as serialize  # noqa: F401


def vector_free(v):
    """Free a vector."""
    check_status(v, lib.GrB_Vector_free(v))


def vector_new(T, size=lib.GxB_INDEX_MAX, *, free=vector_free):
    """Create a new `GrB_Vector` of type `T` and initialize it.

    >>> A = vector_new(lib.GrB_UINT8, 2)
    >>> vector_size(A)
    2

    The default `size` is `lib.GxB_INDEX_MAX`.

    >>> A = vector_new(lib.GrB_UINT8)
    >>> vector_size(A) == lib.GxB_INDEX_MAX
    True

    The `free` argument is called when the object is garbage
    collected, the default is `vector.vector_free()`.  If `free` is None then
    there is no automatic garbage collection and it is up to the user
    to free the vector.
    """
    v = ffi.new("GrB_Vector*")
    check_status(v, lib.GrB_Vector_new(v, T, size))
    if free:
        return ffi.gc(v, free)
    return v


def vector_type(v):
    """Return the GraphBLAS type of the vector.

    >>> v = vector_new(lib.GrB_UINT8, 2)
    >>> vector_type(v) == lib.GrB_UINT8
    True


    """
    T = ffi.new("GrB_Type*")
    check_status(v, lib.GxB_Vector_type(T, v[0]))
    return T[0]


def vector_size(v):
    """Return the size of the vector.

    >>> v = vector_new(lib.GrB_UINT8, 2)
    >>> vector_size(v) == 2
    True

    """
    n = ffi.new("GrB_Index*")
    check_status(v, lib.GrB_Vector_size(n, v[0]))
    return n[0]


def vector_nvals(v):
    """Return the number of stored elements in the vector.

    >>> v = vector_new(lib.GrB_BOOL, 2)
    >>> vector_nvals(v)
    0
    >>> set_bool(v, True, 1)
    >>> vector_nvals(v)
    1

    """
    n = ffi.new("GrB_Index*")
    check_status(v, lib.GrB_Vector_nvals(n, v[0]))
    return n[0]


def vector_option_get_int32(v, field):
    """Get a vector option as an int32.

    >>> v = vector_new(lib.GrB_INT64, 3)
    >>> vector_option_get_int32(v, lib.GxB_SPARSITY_STATUS) in (1, 2, 4, 8)
    True

    """
    val = ffi.new("int32_t*")
    check_status(v, lib.GxB_Vector_Option_get_INT32(v[0], field, val))
    return val[0]


def vector_option_set_int32(v, field, value):
    """Set a vector option from an int32.

    >>> v = vector_new(lib.GrB_INT64, 3)
    >>> vector_option_set_int32(v, lib.GxB_SPARSITY_CONTROL, lib.GxB_SPARSE)
    >>> vector_option_get_int32(v, lib.GxB_SPARSITY_CONTROL) == lib.GxB_SPARSE
    True

    """
    check_status(v, lib.GxB_Vector_Option_set_INT32(
        v[0], field, ffi.cast("int32_t", value),
    ))


def vector_option_get_fp64(v, field):
    """Get a vector option as a float64.

    >>> v = vector_new(lib.GrB_INT64, 3)
    >>> isinstance(vector_option_get_fp64(v, lib.GxB_BITMAP_SWITCH), float)
    True

    """
    val = ffi.new("double*")
    check_status(v, lib.GxB_Vector_Option_get_FP64(v[0], field, val))
    return val[0]


def vector_option_set_fp64(v, field, value):
    """Set a vector option from a float64.

    >>> v = vector_new(lib.GrB_INT64, 3)
    >>> vector_option_set_fp64(v, lib.GxB_BITMAP_SWITCH, 0.25)
    >>> vector_option_get_fp64(v, lib.GxB_BITMAP_SWITCH) == 0.25
    True

    """
    check_status(v, lib.GxB_Vector_Option_set_FP64(
        v[0], field, ffi.cast("double", value),
    ))


# ---------------------------------------------------------------------------
# GraphBLAS operations
# ---------------------------------------------------------------------------


def vector_dup(v, *, free=vector_free):
    """Duplicate a vector.

    >>> v = vector_new(lib.GrB_INT64, 3)
    >>> set_int64(v, 42, 0)
    >>> w = vector_dup(v)
    >>> get_int64(w, 0) == 42
    True

    """
    w = ffi.new("GrB_Vector*")
    check_status(w, lib.GrB_Vector_dup(w, v[0]))
    if free:
        return ffi.gc(w, free)
    return w


def vector_clear(v):
    """Remove all entries from a vector.

    >>> v = vector_new(lib.GrB_INT64, 3)
    >>> set_int64(v, 42, 0)
    >>> vector_nvals(v) == 1
    True
    >>> vector_clear(v)
    >>> vector_nvals(v) == 0
    True

    """
    check_status(v, lib.GrB_Vector_clear(v[0]))


def vector_wait(v, waitmode=lib.GrB_COMPLETE):
    """Wait for a vector to complete pending operations.

    >>> v = vector_new(lib.GrB_INT64, 3)
    >>> vector_wait(v)

    """
    check_status(v, lib.GrB_Vector_wait(v[0], waitmode))


def vector_print(v, name="", level=lib.GxB_COMPLETE):
    """Print a vector to stdout.

    ``level`` controls verbosity: ``lib.GxB_SILENT``, ``lib.GxB_SUMMARY``,
    ``lib.GxB_SHORT``, ``lib.GxB_COMPLETE``, ``lib.GxB_SHORT_VERBOSE``,
    or ``lib.GxB_COMPLETE_VERBOSE``.

    >>> v = vector_new(lib.GrB_INT64, 3)
    >>> set_int64(v, 42, 0)
    >>> vector_wait(v, lib.GrB_MATERIALIZE)
    >>> out = _capture_c_output(vector_print, v, 'v', lib.GxB_SHORT)
    >>> 'int64_t vector' in out
    True
    >>> '(0,0)   42' in out
    True

    """
    check_status(v, lib.GxB_Vector_fprint(
        v[0], name.encode() if isinstance(name, str) else name,
        level, ffi.NULL,
    ))


def vector_fprint(v, f, name="", level=lib.GxB_COMPLETE):
    """Print a vector to a C FILE* stream.

    Pass ``ffi.NULL`` for ``f`` to print to stdout.
    ``level`` controls verbosity (see :func:`vector_print`).

    >>> v = vector_new(lib.GrB_INT64, 3)
    >>> set_int64(v, 7, 2)
    >>> vector_wait(v, lib.GrB_MATERIALIZE)
    >>> out = _capture_c_output(vector_fprint, v, ffi.NULL, 'v', lib.GxB_SHORT)
    >>> 'int64_t vector' in out
    True
    >>> '(2,0)   7' in out
    True

    """
    check_status(v, lib.GxB_Vector_fprint(
        v[0], name.encode() if isinstance(name, str) else name,
        level, f,
    ))


def vector_mxv(w, semiring, A, u, mask=None, accum=None, desc=None):
    """Matrix-vector multiply: w = A (+.*) u.

    >>> from suitesparse_graphblas import matrix
    >>> w = vector_new(lib.GrB_INT64, 2)
    >>> A = matrix.matrix_new(lib.GrB_INT64, 2, 2)
    >>> u = vector_new(lib.GrB_INT64, 2)
    >>> matrix.set_int64(A, 3, 0, 0)
    >>> set_int64(u, 2, 0)
    >>> vector_mxv(w, lib.GrB_PLUS_TIMES_SEMIRING_INT64, A, u)
    >>> get_int64(w, 0) == 6
    True

    """
    check_status(w, lib.GrB_mxv(
        w[0],
        ffi.NULL if mask is None else mask[0],
        ffi.NULL if accum is None else accum,
        semiring,
        A[0],
        u[0],
        ffi.NULL if desc is None else desc,
    ))


def vector_vxm(w, semiring, u, A, mask=None, accum=None, desc=None):
    """Vector-matrix multiply: w = u (+.*) A.

    >>> from suitesparse_graphblas import matrix
    >>> w = vector_new(lib.GrB_INT64, 2)
    >>> u = vector_new(lib.GrB_INT64, 2)
    >>> A = matrix.matrix_new(lib.GrB_INT64, 2, 2)
    >>> set_int64(u, 2, 0)
    >>> matrix.set_int64(A, 3, 0, 0)
    >>> vector_vxm(w, lib.GrB_PLUS_TIMES_SEMIRING_INT64, u, A)
    >>> get_int64(w, 0) == 6
    True

    """
    check_status(w, lib.GrB_vxm(
        w[0],
        ffi.NULL if mask is None else mask[0],
        ffi.NULL if accum is None else accum,
        semiring,
        u[0],
        A[0],
        ffi.NULL if desc is None else desc,
    ))


def vector_ewise_add(w, op, u, v, mask=None, accum=None, desc=None):
    """Element-wise addition of two vectors using a binary operator.

    >>> w = vector_new(lib.GrB_INT64, 3)
    >>> u = vector_new(lib.GrB_INT64, 3)
    >>> v = vector_new(lib.GrB_INT64, 3)
    >>> set_int64(u, 2, 0)
    >>> set_int64(v, 3, 0)
    >>> vector_ewise_add(w, lib.GrB_PLUS_INT64, u, v)
    >>> get_int64(w, 0) == 5
    True

    """
    check_status(w, lib.GrB_Vector_eWiseAdd_BinaryOp(
        w[0],
        ffi.NULL if mask is None else mask[0],
        ffi.NULL if accum is None else accum,
        op,
        u[0],
        v[0],
        ffi.NULL if desc is None else desc,
    ))


def vector_ewise_mult(w, op, u, v, mask=None, accum=None, desc=None):
    """Element-wise multiplication of two vectors using a binary operator.

    >>> w = vector_new(lib.GrB_INT64, 3)
    >>> u = vector_new(lib.GrB_INT64, 3)
    >>> v = vector_new(lib.GrB_INT64, 3)
    >>> set_int64(u, 2, 0)
    >>> set_int64(v, 3, 0)
    >>> vector_ewise_mult(w, lib.GrB_TIMES_INT64, u, v)
    >>> get_int64(w, 0) == 6
    True

    """
    check_status(w, lib.GrB_Vector_eWiseMult_BinaryOp(
        w[0],
        ffi.NULL if mask is None else mask[0],
        ffi.NULL if accum is None else accum,
        op,
        u[0],
        v[0],
        ffi.NULL if desc is None else desc,
    ))


def vector_ewise_union(w, op, u, alpha, v, beta, mask=None, accum=None, desc=None):
    """Element-wise union of two vectors using a binary operator.

    Unlike eWiseAdd, entries present in only one operand use the
    corresponding default scalar (``alpha`` for u, ``beta`` for v).

    >>> from suitesparse_graphblas import scalar
    >>> w = vector_new(lib.GrB_INT64, 3)
    >>> u = vector_new(lib.GrB_INT64, 3)
    >>> v = vector_new(lib.GrB_INT64, 3)
    >>> set_int64(u, 2, 0)
    >>> set_int64(v, 3, 2)
    >>> a = scalar.scalar_new(lib.GrB_INT64)
    >>> b = scalar.scalar_new(lib.GrB_INT64)
    >>> scalar.set_int64(a, 0)
    >>> scalar.set_int64(b, 0)
    >>> vector_ewise_union(w, lib.GrB_PLUS_INT64, u, a, v, b)
    >>> get_int64(w, 0) == 2
    True
    >>> get_int64(w, 2) == 3
    True

    """
    check_status(w, lib.GxB_Vector_eWiseUnion(
        w[0],
        ffi.NULL if mask is None else mask[0],
        ffi.NULL if accum is None else accum,
        op,
        u[0], alpha[0],
        v[0], beta[0],
        ffi.NULL if desc is None else desc,
    ))


def vector_apply(w, op, u, mask=None, accum=None, desc=None):
    """Apply a unary operator to each entry of a vector.

    >>> w = vector_new(lib.GrB_INT64, 3)
    >>> u = vector_new(lib.GrB_INT64, 3)
    >>> set_int64(u, -5, 0)
    >>> vector_apply(w, lib.GrB_ABS_INT64, u)
    >>> get_int64(w, 0) == 5
    True

    """
    check_status(w, lib.GrB_Vector_apply(
        w[0],
        ffi.NULL if mask is None else mask[0],
        ffi.NULL if accum is None else accum,
        op,
        u[0],
        ffi.NULL if desc is None else desc,
    ))


def vector_apply_first(w, op, x, u, mask=None, accum=None, desc=None):
    """Apply a binary operator with a scalar bound to the first argument: w = op(x, u).

    >>> from suitesparse_graphblas import scalar
    >>> w = vector_new(lib.GrB_INT64, 3)
    >>> u = vector_new(lib.GrB_INT64, 3)
    >>> set_int64(u, 3, 0)
    >>> s = scalar.scalar_new(lib.GrB_INT64)
    >>> scalar.set_int64(s, 10)
    >>> vector_apply_first(w, lib.GrB_TIMES_INT64, s, u)
    >>> get_int64(w, 0) == 30
    True

    """
    check_status(w, lib.GrB_Vector_apply_BinaryOp1st_Scalar(
        w[0],
        ffi.NULL if mask is None else mask[0],
        ffi.NULL if accum is None else accum,
        op,
        x[0],
        u[0],
        ffi.NULL if desc is None else desc,
    ))


def vector_apply_second(w, op, u, y, mask=None, accum=None, desc=None):
    """Apply a binary operator with a scalar bound to the second argument: w = op(u, y).

    >>> from suitesparse_graphblas import scalar
    >>> w = vector_new(lib.GrB_INT64, 3)
    >>> u = vector_new(lib.GrB_INT64, 3)
    >>> set_int64(u, 10, 0)
    >>> s = scalar.scalar_new(lib.GrB_INT64)
    >>> scalar.set_int64(s, 3)
    >>> vector_apply_second(w, lib.GrB_TIMES_INT64, u, s)
    >>> get_int64(w, 0) == 30
    True

    """
    check_status(w, lib.GrB_Vector_apply_BinaryOp2nd_Scalar(
        w[0],
        ffi.NULL if mask is None else mask[0],
        ffi.NULL if accum is None else accum,
        op,
        u[0],
        y[0],
        ffi.NULL if desc is None else desc,
    ))


def vector_select(w, op, u, thunk, mask=None, accum=None, desc=None):
    """Select entries from a vector using an index unary operator and scalar thunk.

    >>> from suitesparse_graphblas import scalar
    >>> w = vector_new(lib.GrB_INT64, 3)
    >>> u = vector_new(lib.GrB_INT64, 3)
    >>> set_int64(u, 5, 0)
    >>> set_int64(u, 1, 1)
    >>> s = scalar.scalar_new(lib.GrB_INT64)
    >>> scalar.set_int64(s, 3)
    >>> vector_select(w, lib.GrB_VALUEGT_INT64, u, s)
    >>> vector_nvals(w) == 1
    True
    >>> get_int64(w, 0) == 5
    True

    """
    check_status(w, lib.GrB_Vector_select_Scalar(
        w[0],
        ffi.NULL if mask is None else mask[0],
        ffi.NULL if accum is None else accum,
        op,
        u[0],
        thunk[0],
        ffi.NULL if desc is None else desc,
    ))


def vector_reduce(s, monoid, u, accum=None, desc=None):
    """Reduce a vector to a scalar using a monoid.

    >>> from suitesparse_graphblas import scalar
    >>> u = vector_new(lib.GrB_INT64, 3)
    >>> set_int64(u, 2, 0)
    >>> set_int64(u, 3, 1)
    >>> s = scalar.scalar_new(lib.GrB_INT64)
    >>> vector_reduce(s, lib.GrB_PLUS_MONOID_INT64, u)
    >>> scalar.get_int64(s) == 5
    True

    """
    check_status(s, lib.GrB_Vector_reduce_Monoid_Scalar(
        s[0],
        ffi.NULL if accum is None else accum,
        monoid,
        u[0],
        ffi.NULL if desc is None else desc,
    ))


def vector_diag(v, A, k=0, desc=None):
    """Extract the k-th diagonal of a matrix into a vector.

    ``k=0`` is the main diagonal, positive is above, negative below.

    >>> from suitesparse_graphblas import matrix
    >>> A = matrix.matrix_new(lib.GrB_INT64, 3, 3)
    >>> matrix.set_int64(A, 10, 0, 0)
    >>> matrix.set_int64(A, 20, 1, 1)
    >>> matrix.set_int64(A, 30, 2, 2)
    >>> v = vector_new(lib.GrB_INT64, 3)
    >>> vector_diag(v, A, 0)
    >>> get_int64(v, 0) == 10
    True
    >>> get_int64(v, 1) == 20
    True
    >>> get_int64(v, 2) == 30
    True

    """
    check_status(v, lib.GxB_Vector_diag(
        v[0], A[0], k,
        ffi.NULL if desc is None else desc,
    ))


def vector_assign(w, u, indices, ni, mask=None, accum=None, desc=None):
    """Assign a subvector: w[indices] = u.

    >>> w = vector_new(lib.GrB_INT64, 3)
    >>> u = vector_new(lib.GrB_INT64, 1)
    >>> set_int64(u, 42, 0)
    >>> idx = ffi.new("GrB_Index[1]", [1])
    >>> vector_assign(w, u, idx, 1)
    >>> get_int64(w, 1) == 42
    True

    """
    check_status(w, lib.GrB_Vector_assign(
        w[0],
        ffi.NULL if mask is None else mask[0],
        ffi.NULL if accum is None else accum,
        u[0],
        indices, ni,
        ffi.NULL if desc is None else desc,
    ))


def vector_assign_scalar(w, x, indices, ni, mask=None, accum=None, desc=None):
    """Assign a scalar to subvector positions: w[indices] = x.

    >>> from suitesparse_graphblas import scalar
    >>> w = vector_new(lib.GrB_INT64, 3)
    >>> s = scalar.scalar_new(lib.GrB_INT64)
    >>> scalar.set_int64(s, 99)
    >>> idx = ffi.new("GrB_Index[2]", [0, 2])
    >>> vector_assign_scalar(w, s, idx, 2)
    >>> get_int64(w, 0) == 99
    True
    >>> get_int64(w, 2) == 99
    True

    """
    check_status(w, lib.GrB_Vector_assign_Scalar(
        w[0],
        ffi.NULL if mask is None else mask[0],
        ffi.NULL if accum is None else accum,
        x[0],
        indices, ni,
        ffi.NULL if desc is None else desc,
    ))


def vector_extract(w, u, indices, ni, mask=None, accum=None, desc=None):
    """Extract a subvector: w = u[indices].

    >>> u = vector_new(lib.GrB_INT64, 3)
    >>> set_int64(u, 42, 2)
    >>> w = vector_new(lib.GrB_INT64, 1)
    >>> idx = ffi.new("GrB_Index[1]", [2])
    >>> vector_extract(w, u, idx, 1)
    >>> get_int64(w, 0) == 42
    True

    """
    check_status(w, lib.GrB_Vector_extract(
        w[0],
        ffi.NULL if mask is None else mask[0],
        ffi.NULL if accum is None else accum,
        u[0],
        indices, ni,
        ffi.NULL if desc is None else desc,
    ))


def vector_build_int64(w, indices, vals, nvals, dup):
    """Build a vector from index and int64 value arrays.

    >>> w = vector_new(lib.GrB_INT64, 3)
    >>> idx = np.array([0, 2], dtype=np.uint64)
    >>> vals = np.array([10, 20], dtype=np.int64)
    >>> vector_build_int64(w, idx, vals, 2, lib.GrB_PLUS_INT64)
    >>> get_int64(w, 0) == 10
    True
    >>> get_int64(w, 2) == 20
    True

    """
    check_status(w, lib.GrB_Vector_build_INT64(
        w[0],
        ffi.cast("GrB_Index*", ffi.from_buffer(indices)),
        ffi.cast("int64_t*", ffi.from_buffer(vals)),
        nvals, dup,
    ))


def vector_build_fp64(w, indices, vals, nvals, dup):
    """Build a vector from index and fp64 value arrays.

    >>> w = vector_new(lib.GrB_FP64, 3)
    >>> idx = np.array([0], dtype=np.uint64)
    >>> vals = np.array([1.5], dtype=np.float64)
    >>> vector_build_fp64(w, idx, vals, 1, lib.GrB_PLUS_FP64)
    >>> get_fp64(w, 0) == 1.5
    True

    """
    check_status(w, lib.GrB_Vector_build_FP64(
        w[0],
        ffi.cast("GrB_Index*", ffi.from_buffer(indices)),
        ffi.cast("double*", ffi.from_buffer(vals)),
        nvals, dup,
    ))


def vector_extract_tuples_int64(v):
    """Extract all tuples from a vector as numpy arrays.

    Returns (indices, vals) as numpy arrays.

    >>> v = vector_new(lib.GrB_INT64, 3)
    >>> set_int64(v, 42, 1)
    >>> idx, vals = vector_extract_tuples_int64(v)
    >>> len(vals) == 1
    True
    >>> int(idx[0]) == 1
    True
    >>> int(vals[0]) == 42
    True

    """
    n = vector_nvals(v)
    indices = np.empty(n, dtype=np.uint64)
    vals = np.empty(n, dtype=np.int64)
    nvals_p = ffi.new("GrB_Index*", n)
    if n > 0:
        check_status(v, lib.GrB_Vector_extractTuples_INT64(
            ffi.cast("GrB_Index*", ffi.from_buffer(indices)),
            ffi.cast("int64_t*", ffi.from_buffer(vals)),
            nvals_p, v[0],
        ))
    return indices, vals


def vector_extract_tuples_fp64(v):
    """Extract all tuples from a vector as numpy arrays.

    Returns (indices, vals) as numpy arrays.

    >>> v = vector_new(lib.GrB_FP64, 3)
    >>> set_fp64(v, 1.5, 0)
    >>> idx, vals = vector_extract_tuples_fp64(v)
    >>> len(vals) == 1
    True
    >>> float(vals[0]) == 1.5
    True

    """
    n = vector_nvals(v)
    indices = np.empty(n, dtype=np.uint64)
    vals = np.empty(n, dtype=np.float64)
    nvals_p = ffi.new("GrB_Index*", n)
    if n > 0:
        check_status(v, lib.GrB_Vector_extractTuples_FP64(
            ffi.cast("GrB_Index*", ffi.from_buffer(indices)),
            ffi.cast("double*", ffi.from_buffer(vals)),
            nvals_p, v[0],
        ))
    return indices, vals

def vector_build_bool(w, indices, vals, nvals, dup):
    """Build a vector from index and bool value arrays.

    >>> w = vector_new(lib.GrB_BOOL, 3)
    >>> idx = np.array([0], dtype=np.uint64)
    >>> vals = np.array([True], dtype=np.bool_)
    >>> vector_build_bool(w, idx, vals, 1, lib.GrB_FIRST_BOOL)
    >>> get_bool(w, 0) == True
    True

    """
    check_status(w, lib.GrB_Vector_build_BOOL(
        w[0],
        ffi.cast("GrB_Index*", ffi.from_buffer(indices)),
        ffi.cast("bool*", ffi.from_buffer(vals)),
        nvals, dup,
    ))

def vector_build_int8(w, indices, vals, nvals, dup):
    """Build a vector from index and int8 value arrays.

    >>> w = vector_new(lib.GrB_INT8, 3)
    >>> idx = np.array([0], dtype=np.uint64)
    >>> vals = np.array([7], dtype=np.int8)
    >>> vector_build_int8(w, idx, vals, 1, lib.GrB_PLUS_INT8)
    >>> get_int8(w, 0) == 7
    True

    """
    check_status(w, lib.GrB_Vector_build_INT8(
        w[0],
        ffi.cast("GrB_Index*", ffi.from_buffer(indices)),
        ffi.cast("int8_t*", ffi.from_buffer(vals)),
        nvals, dup,
    ))

def vector_build_int16(w, indices, vals, nvals, dup):
    """Build a vector from index and int16 value arrays.

    >>> w = vector_new(lib.GrB_INT16, 3)
    >>> idx = np.array([0], dtype=np.uint64)
    >>> vals = np.array([7], dtype=np.int16)
    >>> vector_build_int16(w, idx, vals, 1, lib.GrB_PLUS_INT16)
    >>> get_int16(w, 0) == 7
    True

    """
    check_status(w, lib.GrB_Vector_build_INT16(
        w[0],
        ffi.cast("GrB_Index*", ffi.from_buffer(indices)),
        ffi.cast("int16_t*", ffi.from_buffer(vals)),
        nvals, dup,
    ))

def vector_build_int32(w, indices, vals, nvals, dup):
    """Build a vector from index and int32 value arrays.

    >>> w = vector_new(lib.GrB_INT32, 3)
    >>> idx = np.array([0], dtype=np.uint64)
    >>> vals = np.array([7], dtype=np.int32)
    >>> vector_build_int32(w, idx, vals, 1, lib.GrB_PLUS_INT32)
    >>> get_int32(w, 0) == 7
    True

    """
    check_status(w, lib.GrB_Vector_build_INT32(
        w[0],
        ffi.cast("GrB_Index*", ffi.from_buffer(indices)),
        ffi.cast("int32_t*", ffi.from_buffer(vals)),
        nvals, dup,
    ))

def vector_build_uint8(w, indices, vals, nvals, dup):
    """Build a vector from index and uint8 value arrays.

    >>> w = vector_new(lib.GrB_UINT8, 3)
    >>> idx = np.array([0], dtype=np.uint64)
    >>> vals = np.array([7], dtype=np.uint8)
    >>> vector_build_uint8(w, idx, vals, 1, lib.GrB_PLUS_UINT8)
    >>> get_uint8(w, 0) == 7
    True

    """
    check_status(w, lib.GrB_Vector_build_UINT8(
        w[0],
        ffi.cast("GrB_Index*", ffi.from_buffer(indices)),
        ffi.cast("uint8_t*", ffi.from_buffer(vals)),
        nvals, dup,
    ))

def vector_build_uint16(w, indices, vals, nvals, dup):
    """Build a vector from index and uint16 value arrays.

    >>> w = vector_new(lib.GrB_UINT16, 3)
    >>> idx = np.array([0], dtype=np.uint64)
    >>> vals = np.array([7], dtype=np.uint16)
    >>> vector_build_uint16(w, idx, vals, 1, lib.GrB_PLUS_UINT16)
    >>> get_uint16(w, 0) == 7
    True

    """
    check_status(w, lib.GrB_Vector_build_UINT16(
        w[0],
        ffi.cast("GrB_Index*", ffi.from_buffer(indices)),
        ffi.cast("uint16_t*", ffi.from_buffer(vals)),
        nvals, dup,
    ))

def vector_build_uint32(w, indices, vals, nvals, dup):
    """Build a vector from index and uint32 value arrays.

    >>> w = vector_new(lib.GrB_UINT32, 3)
    >>> idx = np.array([0], dtype=np.uint64)
    >>> vals = np.array([7], dtype=np.uint32)
    >>> vector_build_uint32(w, idx, vals, 1, lib.GrB_PLUS_UINT32)
    >>> get_uint32(w, 0) == 7
    True

    """
    check_status(w, lib.GrB_Vector_build_UINT32(
        w[0],
        ffi.cast("GrB_Index*", ffi.from_buffer(indices)),
        ffi.cast("uint32_t*", ffi.from_buffer(vals)),
        nvals, dup,
    ))

def vector_build_uint64(w, indices, vals, nvals, dup):
    """Build a vector from index and uint64 value arrays.

    >>> w = vector_new(lib.GrB_UINT64, 3)
    >>> idx = np.array([0], dtype=np.uint64)
    >>> vals = np.array([7], dtype=np.uint64)
    >>> vector_build_uint64(w, idx, vals, 1, lib.GrB_PLUS_UINT64)
    >>> get_uint64(w, 0) == 7
    True

    """
    check_status(w, lib.GrB_Vector_build_UINT64(
        w[0],
        ffi.cast("GrB_Index*", ffi.from_buffer(indices)),
        ffi.cast("uint64_t*", ffi.from_buffer(vals)),
        nvals, dup,
    ))

def vector_build_fp32(w, indices, vals, nvals, dup):
    """Build a vector from index and fp32 value arrays.

    >>> w = vector_new(lib.GrB_FP32, 3)
    >>> idx = np.array([0], dtype=np.uint64)
    >>> vals = np.array([1.5], dtype=np.float32)
    >>> vector_build_fp32(w, idx, vals, 1, lib.GrB_PLUS_FP32)
    >>> get_fp32(w, 0) == 1.5
    True

    """
    check_status(w, lib.GrB_Vector_build_FP32(
        w[0],
        ffi.cast("GrB_Index*", ffi.from_buffer(indices)),
        ffi.cast("float*", ffi.from_buffer(vals)),
        nvals, dup,
    ))

def vector_extract_tuples_bool(v):
    """Extract all tuples from a vector as numpy arrays.

    Returns (indices, vals) as numpy arrays.

    >>> v = vector_new(lib.GrB_BOOL, 3)
    >>> set_bool(v, True, 0)
    >>> idx, vals = vector_extract_tuples_bool(v)
    >>> len(vals) == 1
    True
    >>> bool(vals[0]) == True
    True

    """
    n = vector_nvals(v)
    indices = np.empty(n, dtype=np.uint64)
    vals = np.empty(n, dtype=np.bool_)
    nvals_p = ffi.new("GrB_Index*", n)
    if n > 0:
        check_status(v, lib.GrB_Vector_extractTuples_BOOL(
            ffi.cast("GrB_Index*", ffi.from_buffer(indices)),
            ffi.cast("bool*", ffi.from_buffer(vals)),
            nvals_p, v[0],
        ))
    return indices, vals

def vector_extract_tuples_int8(v):
    """Extract all tuples from a vector as numpy arrays.

    Returns (indices, vals) as numpy arrays.

    >>> v = vector_new(lib.GrB_INT8, 3)
    >>> set_int8(v, 7, 0)
    >>> idx, vals = vector_extract_tuples_int8(v)
    >>> len(vals) == 1
    True
    >>> int(vals[0]) == 7
    True

    """
    n = vector_nvals(v)
    indices = np.empty(n, dtype=np.uint64)
    vals = np.empty(n, dtype=np.int8)
    nvals_p = ffi.new("GrB_Index*", n)
    if n > 0:
        check_status(v, lib.GrB_Vector_extractTuples_INT8(
            ffi.cast("GrB_Index*", ffi.from_buffer(indices)),
            ffi.cast("int8_t*", ffi.from_buffer(vals)),
            nvals_p, v[0],
        ))
    return indices, vals

def vector_extract_tuples_int16(v):
    """Extract all tuples from a vector as numpy arrays.

    Returns (indices, vals) as numpy arrays.

    >>> v = vector_new(lib.GrB_INT16, 3)
    >>> set_int16(v, 7, 0)
    >>> idx, vals = vector_extract_tuples_int16(v)
    >>> len(vals) == 1
    True
    >>> int(vals[0]) == 7
    True

    """
    n = vector_nvals(v)
    indices = np.empty(n, dtype=np.uint64)
    vals = np.empty(n, dtype=np.int16)
    nvals_p = ffi.new("GrB_Index*", n)
    if n > 0:
        check_status(v, lib.GrB_Vector_extractTuples_INT16(
            ffi.cast("GrB_Index*", ffi.from_buffer(indices)),
            ffi.cast("int16_t*", ffi.from_buffer(vals)),
            nvals_p, v[0],
        ))
    return indices, vals

def vector_extract_tuples_int32(v):
    """Extract all tuples from a vector as numpy arrays.

    Returns (indices, vals) as numpy arrays.

    >>> v = vector_new(lib.GrB_INT32, 3)
    >>> set_int32(v, 7, 0)
    >>> idx, vals = vector_extract_tuples_int32(v)
    >>> len(vals) == 1
    True
    >>> int(vals[0]) == 7
    True

    """
    n = vector_nvals(v)
    indices = np.empty(n, dtype=np.uint64)
    vals = np.empty(n, dtype=np.int32)
    nvals_p = ffi.new("GrB_Index*", n)
    if n > 0:
        check_status(v, lib.GrB_Vector_extractTuples_INT32(
            ffi.cast("GrB_Index*", ffi.from_buffer(indices)),
            ffi.cast("int32_t*", ffi.from_buffer(vals)),
            nvals_p, v[0],
        ))
    return indices, vals

def vector_extract_tuples_uint8(v):
    """Extract all tuples from a vector as numpy arrays.

    Returns (indices, vals) as numpy arrays.

    >>> v = vector_new(lib.GrB_UINT8, 3)
    >>> set_uint8(v, 7, 0)
    >>> idx, vals = vector_extract_tuples_uint8(v)
    >>> len(vals) == 1
    True
    >>> int(vals[0]) == 7
    True

    """
    n = vector_nvals(v)
    indices = np.empty(n, dtype=np.uint64)
    vals = np.empty(n, dtype=np.uint8)
    nvals_p = ffi.new("GrB_Index*", n)
    if n > 0:
        check_status(v, lib.GrB_Vector_extractTuples_UINT8(
            ffi.cast("GrB_Index*", ffi.from_buffer(indices)),
            ffi.cast("uint8_t*", ffi.from_buffer(vals)),
            nvals_p, v[0],
        ))
    return indices, vals

def vector_extract_tuples_uint16(v):
    """Extract all tuples from a vector as numpy arrays.

    Returns (indices, vals) as numpy arrays.

    >>> v = vector_new(lib.GrB_UINT16, 3)
    >>> set_uint16(v, 7, 0)
    >>> idx, vals = vector_extract_tuples_uint16(v)
    >>> len(vals) == 1
    True
    >>> int(vals[0]) == 7
    True

    """
    n = vector_nvals(v)
    indices = np.empty(n, dtype=np.uint64)
    vals = np.empty(n, dtype=np.uint16)
    nvals_p = ffi.new("GrB_Index*", n)
    if n > 0:
        check_status(v, lib.GrB_Vector_extractTuples_UINT16(
            ffi.cast("GrB_Index*", ffi.from_buffer(indices)),
            ffi.cast("uint16_t*", ffi.from_buffer(vals)),
            nvals_p, v[0],
        ))
    return indices, vals

def vector_extract_tuples_uint32(v):
    """Extract all tuples from a vector as numpy arrays.

    Returns (indices, vals) as numpy arrays.

    >>> v = vector_new(lib.GrB_UINT32, 3)
    >>> set_uint32(v, 7, 0)
    >>> idx, vals = vector_extract_tuples_uint32(v)
    >>> len(vals) == 1
    True
    >>> int(vals[0]) == 7
    True

    """
    n = vector_nvals(v)
    indices = np.empty(n, dtype=np.uint64)
    vals = np.empty(n, dtype=np.uint32)
    nvals_p = ffi.new("GrB_Index*", n)
    if n > 0:
        check_status(v, lib.GrB_Vector_extractTuples_UINT32(
            ffi.cast("GrB_Index*", ffi.from_buffer(indices)),
            ffi.cast("uint32_t*", ffi.from_buffer(vals)),
            nvals_p, v[0],
        ))
    return indices, vals

def vector_extract_tuples_uint64(v):
    """Extract all tuples from a vector as numpy arrays.

    Returns (indices, vals) as numpy arrays.

    >>> v = vector_new(lib.GrB_UINT64, 3)
    >>> set_uint64(v, 7, 0)
    >>> idx, vals = vector_extract_tuples_uint64(v)
    >>> len(vals) == 1
    True
    >>> int(vals[0]) == 7
    True

    """
    n = vector_nvals(v)
    indices = np.empty(n, dtype=np.uint64)
    vals = np.empty(n, dtype=np.uint64)
    nvals_p = ffi.new("GrB_Index*", n)
    if n > 0:
        check_status(v, lib.GrB_Vector_extractTuples_UINT64(
            ffi.cast("GrB_Index*", ffi.from_buffer(indices)),
            ffi.cast("uint64_t*", ffi.from_buffer(vals)),
            nvals_p, v[0],
        ))
    return indices, vals

def vector_extract_tuples_fp32(v):
    """Extract all tuples from a vector as numpy arrays.

    Returns (indices, vals) as numpy arrays.

    >>> v = vector_new(lib.GrB_FP32, 3)
    >>> set_fp32(v, 1.5, 0)
    >>> idx, vals = vector_extract_tuples_fp32(v)
    >>> len(vals) == 1
    True
    >>> float(vals[0]) == 1.5
    True

    """
    n = vector_nvals(v)
    indices = np.empty(n, dtype=np.uint64)
    vals = np.empty(n, dtype=np.float32)
    nvals_p = ffi.new("GrB_Index*", n)
    if n > 0:
        check_status(v, lib.GrB_Vector_extractTuples_FP32(
            ffi.cast("GrB_Index*", ffi.from_buffer(indices)),
            ffi.cast("float*", ffi.from_buffer(vals)),
            nvals_p, v[0],
        ))
    return indices, vals


def set_bool(v, value, i):
    """Set a boolean value to the vector at position `i`.

    >>> v = vector_new(lib.GrB_BOOL, 3)
    >>> set_bool(v, True, 2)
    >>> get_bool(v, 2) == True
    True

    """
    check_status(v, lib.GrB_Vector_setElement_BOOL(v[0], value, i))


def get_bool(v, i):
    """Get a boolean value from the vector at position `i`.

    >>> v = vector_new(lib.GrB_BOOL, 3)
    >>> set_bool(v, True, 2)
    >>> get_bool(v, 2) == True
    True

    """
    value = ffi.new("bool*")
    check_status(v, lib.GrB_Vector_extractElement_BOOL(value, v[0], i))
    return value[0]


def set_int8(v, value, i):
    """Set an int8 value to the vector at position `i`.

    >>> v = vector_new(lib.GrB_INT8, 3)
    >>> set_int8(v, 7, 2)
    >>> get_int8(v, 2) == 7
    True

    """
    check_status(v, lib.GrB_Vector_setElement_INT8(v[0], value, i))


def get_int8(v, i):
    """Get an int8 value from the vector at position `i`.

    >>> v = vector_new(lib.GrB_INT8, 3)
    >>> set_int8(v, 7, 2)
    >>> get_int8(v, 2) == 7
    True

    """
    value = ffi.new("int8_t*")
    check_status(v, lib.GrB_Vector_extractElement_INT8(value, v[0], i))
    return value[0]


def set_int16(v, value, i):
    """Set an int16 value to the vector at position `i`.

    >>> v = vector_new(lib.GrB_INT16, 3)
    >>> set_int16(v, 7, 2)
    >>> get_int16(v, 2) == 7
    True

    """
    check_status(v, lib.GrB_Vector_setElement_INT16(v[0], value, i))


def get_int16(v, i):
    """Get an int16 value from the vector at position `i`.

    >>> v = vector_new(lib.GrB_INT16, 3)
    >>> set_int16(v, 7, 2)
    >>> get_int16(v, 2) == 7
    True

    """
    value = ffi.new("int16_t*")
    check_status(v, lib.GrB_Vector_extractElement_INT16(value, v[0], i))
    return value[0]


def set_int32(v, value, i):
    """Set an int32 value to the vector at position `i`.

    >>> v = vector_new(lib.GrB_INT32, 3)
    >>> set_int32(v, 7, 2)
    >>> get_int32(v, 2) == 7
    True

    """
    check_status(v, lib.GrB_Vector_setElement_INT32(v[0], value, i))


def get_int32(v, i):
    """Get an int32 value from the vector at position `i`.

    >>> v = vector_new(lib.GrB_INT32, 3)
    >>> set_int32(v, 7, 2)
    >>> get_int32(v, 2) == 7
    True

    """
    value = ffi.new("int32_t*")
    check_status(v, lib.GrB_Vector_extractElement_INT32(value, v[0], i))
    return value[0]


def set_int64(v, value, i):
    """Set an int64 value to the vector at position `i`.

    >>> v = vector_new(lib.GrB_INT64, 3)
    >>> set_int64(v, 7, 2)
    >>> get_int64(v, 2) == 7
    True

    """
    check_status(v, lib.GrB_Vector_setElement_INT64(v[0], value, i))


def get_int64(v, i):
    """Get an int64 value from the vector at position `i`.

    >>> v = vector_new(lib.GrB_INT64, 3)
    >>> set_int64(v, 7, 2)
    >>> get_int64(v, 2) == 7
    True

    """
    value = ffi.new("int64_t*")
    check_status(v, lib.GrB_Vector_extractElement_INT64(value, v[0], i))
    return value[0]


def set_uint8(v, value, i):
    """Set a uint8 value to the vector at position `i`.

    >>> v = vector_new(lib.GrB_UINT8, 3)
    >>> set_uint8(v, 7, 2)
    >>> get_uint8(v, 2) == 7
    True

    """
    check_status(v, lib.GrB_Vector_setElement_UINT8(v[0], value, i))


def get_uint8(v, i):
    """Get a uint8 value from the vector at position `i`.

    >>> v = vector_new(lib.GrB_UINT8, 3)
    >>> set_uint8(v, 7, 2)
    >>> get_uint8(v, 2) == 7
    True

    """
    value = ffi.new("uint8_t*")
    check_status(v, lib.GrB_Vector_extractElement_UINT8(value, v[0], i))
    return value[0]


def set_uint16(v, value, i):
    """Set a uint16 value to the vector at position `i`.

    >>> v = vector_new(lib.GrB_UINT16, 3)
    >>> set_uint16(v, 7, 2)
    >>> get_uint16(v, 2) == 7
    True

    """
    check_status(v, lib.GrB_Vector_setElement_UINT16(v[0], value, i))


def get_uint16(v, i):
    """Get a uint16 value from the vector at position `i`.

    >>> v = vector_new(lib.GrB_UINT16, 3)
    >>> set_uint16(v, 7, 2)
    >>> get_uint16(v, 2) == 7
    True

    """
    value = ffi.new("uint16_t*")
    check_status(v, lib.GrB_Vector_extractElement_UINT16(value, v[0], i))
    return value[0]


def set_uint32(v, value, i):
    """Set a uint32 value to the vector at position `i`.

    >>> v = vector_new(lib.GrB_UINT32, 3)
    >>> set_uint32(v, 7, 2)
    >>> get_uint32(v, 2) == 7
    True

    """
    check_status(v, lib.GrB_Vector_setElement_UINT32(v[0], value, i))


def get_uint32(v, i):
    """Get a uint32 value from the vector at position `i`.

    >>> v = vector_new(lib.GrB_UINT32, 3)
    >>> set_uint32(v, 7, 2)
    >>> get_uint32(v, 2) == 7
    True

    """
    value = ffi.new("uint32_t*")
    check_status(v, lib.GrB_Vector_extractElement_UINT32(value, v[0], i))
    return value[0]


def set_uint64(v, value, i):
    """Set a uint64 value to the vector at position `i`.

    >>> v = vector_new(lib.GrB_UINT64, 3)
    >>> set_uint64(v, 7, 2)
    >>> get_uint64(v, 2) == 7
    True

    """
    check_status(v, lib.GrB_Vector_setElement_UINT64(v[0], value, i))


def get_uint64(v, i):
    """Get a uint64 value from the vector at position `i`.

    >>> v = vector_new(lib.GrB_UINT64, 3)
    >>> set_uint64(v, 7, 2)
    >>> get_uint64(v, 2) == 7
    True

    """
    value = ffi.new("uint64_t*")
    check_status(v, lib.GrB_Vector_extractElement_UINT64(value, v[0], i))
    return value[0]


def set_fp32(v, value, i):
    """Set an fp32 value to the vector at position `i`.

    >>> v = vector_new(lib.GrB_FP32, 3)
    >>> set_fp32(v, 1.5, 2)
    >>> get_fp32(v, 2) == 1.5
    True

    """
    check_status(v, lib.GrB_Vector_setElement_FP32(v[0], value, i))


def get_fp32(v, i):
    """Get an fp32 value from the vector at position `i`.

    >>> v = vector_new(lib.GrB_FP32, 3)
    >>> set_fp32(v, 1.5, 2)
    >>> get_fp32(v, 2) == 1.5
    True

    """
    value = ffi.new("float*")
    check_status(v, lib.GrB_Vector_extractElement_FP32(value, v[0], i))
    return value[0]


def set_fp64(v, value, i):
    """Set an fp64 value to the vector at position `i`.

    >>> v = vector_new(lib.GrB_FP64, 3)
    >>> set_fp64(v, 1.5, 2)
    >>> get_fp64(v, 2) == 1.5
    True

    """
    check_status(v, lib.GrB_Vector_setElement_FP64(v[0], value, i))


def get_fp64(v, i):
    """Get an fp64 value from the vector at position `i`.

    >>> v = vector_new(lib.GrB_FP64, 3)
    >>> set_fp64(v, 1.5, 2)
    >>> get_fp64(v, 2) == 1.5
    True

    """
    value = ffi.new("double*")
    check_status(v, lib.GrB_Vector_extractElement_FP64(value, v[0], i))
    return value[0]


if supports_complex():

    def set_fc32(v, value, i):
        """Set an fc32 value to the vector at position `i`.

        >>> v = vector_new(lib.GxB_FC32, 3)
        >>> set_fc32(v, 2+3j, 2)
        >>> get_fc32(v, 2) == 2+3j
        True

        """
        check_status(v, lib.GxB_Vector_setElement_FC32(v[0], value, i))

    def get_fc32(v, i):
        """Get an fc32 value from the vector at position `i`.

        >>> v = vector_new(lib.GxB_FC32, 3)
        >>> set_fc32(v, 2+3j, 2)
        >>> get_fc32(v, 2) == 2+3j
        True

        """
        value = ffi.new("GxB_FC32_t*")
        check_status(v, lib.GxB_Vector_extractElement_FC32(value, v[0], i))
        return value[0]

    def set_fc64(v, value, i):
        """Set an fc64 value to the vector at position `i`.

        >>> v = vector_new(lib.GxB_FC64, 3)
        >>> set_fc64(v, 2+3j, 2)
        >>> get_fc64(v, 2) == 2+3j
        True

        """
        check_status(v, lib.GxB_Vector_setElement_FC64(v[0], value, i))

    def get_fc64(v, i):
        """Get an fc64 value from the vector at position `i`.

        >>> v = vector_new(lib.GxB_FC64, 3)
        >>> set_fc64(v, 2+3j, 2)
        >>> get_fc64(v, 2) == 2+3j
        True

        """
        value = ffi.new("GxB_FC64_t*")
        check_status(v, lib.GxB_Vector_extractElement_FC64(value, v[0], i))
        return value[0]

    def vector_build_fc32(w, indices, vals, nvals, dup):
        """Build a vector from index and fc32 value arrays.

        >>> w = vector_new(lib.GxB_FC32, 3)
        >>> idx = np.array([0], dtype=np.uint64)
        >>> vals = np.array([2+3j], dtype=np.complex64)
        >>> vector_build_fc32(w, idx, vals, 1, lib.GxB_PLUS_FC32)
        >>> get_fc32(w, 0) == 2+3j
        True

        """
        check_status(w, lib.GxB_Vector_build_FC32(
            w[0],
            ffi.cast("GrB_Index*", ffi.from_buffer(indices)),
            ffi.cast("GxB_FC32_t*", ffi.from_buffer(vals)),
            nvals, dup,
        ))

    def vector_extract_tuples_fc32(v):
        """Extract all tuples from a vector as numpy arrays.

        Returns (indices, vals) as numpy arrays.

        >>> v = vector_new(lib.GxB_FC32, 3)
        >>> set_fc32(v, 2+3j, 0)
        >>> idx, vals = vector_extract_tuples_fc32(v)
        >>> len(vals) == 1
        True
        >>> complex(vals[0]) == 2+3j
        True

        """
        n = vector_nvals(v)
        indices = np.empty(n, dtype=np.uint64)
        vals = np.empty(n, dtype=np.complex64)
        nvals_p = ffi.new("GrB_Index*", n)
        if n > 0:
            check_status(v, lib.GxB_Vector_extractTuples_FC32(
                ffi.cast("GrB_Index*", ffi.from_buffer(indices)),
                ffi.cast("GxB_FC32_t*", ffi.from_buffer(vals)),
                nvals_p, v[0],
            ))
        return indices, vals

    def vector_build_fc64(w, indices, vals, nvals, dup):
        """Build a vector from index and fc64 value arrays.

        >>> w = vector_new(lib.GxB_FC64, 3)
        >>> idx = np.array([0], dtype=np.uint64)
        >>> vals = np.array([2+3j], dtype=np.complex128)
        >>> vector_build_fc64(w, idx, vals, 1, lib.GxB_PLUS_FC64)
        >>> get_fc64(w, 0) == 2+3j
        True

        """
        check_status(w, lib.GxB_Vector_build_FC64(
            w[0],
            ffi.cast("GrB_Index*", ffi.from_buffer(indices)),
            ffi.cast("GxB_FC64_t*", ffi.from_buffer(vals)),
            nvals, dup,
        ))

    def vector_extract_tuples_fc64(v):
        """Extract all tuples from a vector as numpy arrays.

        Returns (indices, vals) as numpy arrays.

        >>> v = vector_new(lib.GxB_FC64, 3)
        >>> set_fc64(v, 2+3j, 0)
        >>> idx, vals = vector_extract_tuples_fc64(v)
        >>> len(vals) == 1
        True
        >>> complex(vals[0]) == 2+3j
        True

        """
        n = vector_nvals(v)
        indices = np.empty(n, dtype=np.uint64)
        vals = np.empty(n, dtype=np.complex128)
        nvals_p = ffi.new("GrB_Index*", n)
        if n > 0:
            check_status(v, lib.GxB_Vector_extractTuples_FC64(
                ffi.cast("GrB_Index*", ffi.from_buffer(indices)),
                ffi.cast("GxB_FC64_t*", ffi.from_buffer(vals)),
                nvals_p, v[0],
            ))
        return indices, vals
