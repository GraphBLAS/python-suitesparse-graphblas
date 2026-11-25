from suitesparse_graphblas import _error_code_lookup, check_status, ffi, lib, supports_complex


def _check_info(info):
    """Check a GrB_Info code, returning it on SUCCESS/EXHAUSTED/NO_VALUE, raising otherwise."""
    if info == lib.GrB_SUCCESS or info == lib.GxB_EXHAUSTED or info == lib.GrB_NO_VALUE:
        return info
    exc_class = _error_code_lookup.get(info, RuntimeError)
    raise exc_class(f"GraphBLAS error: info={info}")


# ---------------------------------------------------------------------------
# Iterator lifecycle
# ---------------------------------------------------------------------------


def iterator_free(it):
    """Free an iterator."""
    check_status(it, lib.GxB_Iterator_free(it))


def iterator_new(*, free=iterator_free):
    """Create a new GxB_Iterator.

    >>> it = iterator_new()

    """
    it = ffi.new("GxB_Iterator*")
    check_status(it, lib.GxB_Iterator_new(it))
    if free:
        return ffi.gc(it, free)
    return it


# ---------------------------------------------------------------------------
# Attach
# ---------------------------------------------------------------------------


def matrix_iterator_attach(it, A, desc=None):
    """Attach an iterator to a matrix for entry-wise iteration.

    Returns ``lib.GrB_SUCCESS``, and raises on errors.  The iterator is not at
    any entry yet: move it to one with :func:`matrix_iterator_seek`.

    >>> from suitesparse_graphblas import matrix
    >>> A = matrix.matrix_new(lib.GrB_INT64, 2, 2)
    >>> matrix.matrix_option_set_int32(A, lib.GxB_SPARSITY_CONTROL, lib.GxB_SPARSE)
    >>> matrix.set_int64(A, 10, 0, 0)
    >>> matrix.set_int64(A, 20, 1, 1)
    >>> matrix.matrix_wait(A, lib.GrB_MATERIALIZE)
    >>> it = iterator_new()
    >>> info = matrix_iterator_attach(it, A)
    >>> info == lib.GrB_SUCCESS
    True

    """
    info = lib.GxB_Matrix_Iterator_attach(
        it[0],
        A[0],
        ffi.NULL if desc is None else desc,
    )
    return _check_info(info)


def row_iterator_attach(it, A, desc=None):
    """Attach an iterator to a matrix for row-wise iteration.

    Returns ``lib.GrB_SUCCESS``, and raises on errors: ``NotImplementedException``
    if the matrix is held by column.  The iterator is not at any row yet: move it
    to one with :func:`row_iterator_seek_row` or :func:`row_iterator_kseek`.

    >>> from suitesparse_graphblas import matrix
    >>> A = matrix.matrix_new(lib.GrB_INT64, 2, 2)
    >>> matrix.matrix_option_set_int32(A, lib.GxB_SPARSITY_CONTROL, lib.GxB_SPARSE)
    >>> matrix.set_int64(A, 10, 0, 0)
    >>> matrix.set_int64(A, 20, 1, 1)
    >>> matrix.matrix_wait(A, lib.GrB_MATERIALIZE)
    >>> it = iterator_new()
    >>> info = row_iterator_attach(it, A)
    >>> info == lib.GrB_SUCCESS
    True

    """
    info = lib.GxB_rowIterator_attach(
        it[0],
        A[0],
        ffi.NULL if desc is None else desc,
    )
    return _check_info(info)


def col_iterator_attach(it, A, desc=None):
    """Attach an iterator to a matrix for column-wise iteration.

    Returns ``lib.GrB_SUCCESS``, and raises on errors: ``NotImplementedException``
    if the matrix is held by row.  The iterator is not at any column yet: move it
    to one with :func:`col_iterator_seek_col` or :func:`col_iterator_kseek`.

    >>> from suitesparse_graphblas import matrix
    >>> A = matrix.matrix_new(lib.GrB_INT64, 2, 2)
    >>> matrix.matrix_set_format(A, lib.GxB_BY_COL)
    >>> matrix.matrix_option_set_int32(A, lib.GxB_SPARSITY_CONTROL, lib.GxB_SPARSE)
    >>> matrix.set_int64(A, 10, 0, 0)
    >>> matrix.matrix_wait(A, lib.GrB_MATERIALIZE)
    >>> it = iterator_new()
    >>> info = col_iterator_attach(it, A)
    >>> info == lib.GrB_SUCCESS
    True

    """
    info = lib.GxB_colIterator_attach(
        it[0],
        A[0],
        ffi.NULL if desc is None else desc,
    )
    return _check_info(info)


def vector_iterator_attach(it, v, desc=None):
    """Attach an iterator to a vector.

    Returns ``lib.GrB_SUCCESS``, and raises on errors.  The iterator is not at
    any entry yet: move it to one with :func:`vector_iterator_seek`.

    >>> from suitesparse_graphblas import vector
    >>> v = vector.vector_new(lib.GrB_INT64, 3)
    >>> vector.set_int64(v, 42, 0)
    >>> vector.set_int64(v, 7, 2)
    >>> vector.vector_wait(v, lib.GrB_MATERIALIZE)
    >>> it = iterator_new()
    >>> info = vector_iterator_attach(it, v)
    >>> info == lib.GrB_SUCCESS
    True

    """
    info = lib.GxB_Vector_Iterator_attach(
        it[0],
        v[0],
        ffi.NULL if desc is None else desc,
    )
    return _check_info(info)


# ---------------------------------------------------------------------------
# Matrix entry iterator
# ---------------------------------------------------------------------------


def matrix_iterator_seek(it, p):
    """Seek the matrix entry iterator to position p.

    Returns ``lib.GrB_SUCCESS``, or ``lib.GxB_EXHAUSTED`` if ``p >= pmax``.  For a
    bitmap matrix, the iterator moves to the first entry at or after ``p``, and is
    exhausted if there is none.

    >>> from suitesparse_graphblas import matrix
    >>> A = matrix.matrix_new(lib.GrB_INT64, 2, 2)
    >>> matrix.matrix_option_set_int32(A, lib.GxB_SPARSITY_CONTROL, lib.GxB_SPARSE)
    >>> matrix.set_int64(A, 10, 0, 0)
    >>> matrix.matrix_wait(A, lib.GrB_MATERIALIZE)
    >>> it = iterator_new()
    >>> matrix_iterator_attach(it, A) == lib.GrB_SUCCESS
    True
    >>> matrix_iterator_seek(it, 0) == lib.GrB_SUCCESS
    True

    """
    info = lib.GxB_Matrix_Iterator_seek(it[0], p)
    return _check_info(info)


def matrix_iterator_next(it):
    """Advance the matrix entry iterator to the next entry.

    Returns ``lib.GrB_SUCCESS`` or ``lib.GxB_EXHAUSTED``.

    >>> from suitesparse_graphblas import matrix
    >>> A = matrix.matrix_new(lib.GrB_INT64, 2, 2)
    >>> matrix.matrix_option_set_int32(A, lib.GxB_SPARSITY_CONTROL, lib.GxB_SPARSE)
    >>> matrix.set_int64(A, 10, 0, 0)
    >>> matrix.set_int64(A, 20, 1, 1)
    >>> matrix.matrix_wait(A, lib.GrB_MATERIALIZE)
    >>> it = iterator_new()
    >>> matrix_iterator_attach(it, A) == lib.GrB_SUCCESS
    True
    >>> matrix_iterator_seek(it, 0) == lib.GrB_SUCCESS
    True
    >>> iterator_get_int64(it)
    10
    >>> matrix_iterator_next(it) == lib.GrB_SUCCESS
    True
    >>> iterator_get_int64(it)
    20
    >>> matrix_iterator_next(it) == lib.GxB_EXHAUSTED
    True

    """
    info = lib.GxB_Matrix_Iterator_next(it[0])
    return _check_info(info)


def matrix_iterator_get_index(it):
    """Get the row and column index of the current matrix entry.

    Returns a ``(row, col)`` tuple.

    >>> from suitesparse_graphblas import matrix
    >>> A = matrix.matrix_new(lib.GrB_INT64, 2, 2)
    >>> matrix.matrix_option_set_int32(A, lib.GxB_SPARSITY_CONTROL, lib.GxB_SPARSE)
    >>> matrix.set_int64(A, 10, 0, 1)
    >>> matrix.matrix_wait(A, lib.GrB_MATERIALIZE)
    >>> it = iterator_new()
    >>> matrix_iterator_attach(it, A) == lib.GrB_SUCCESS
    True
    >>> matrix_iterator_seek(it, 0) == lib.GrB_SUCCESS
    True
    >>> matrix_iterator_get_index(it)
    (0, 1)

    """
    row = ffi.new("GrB_Index*")
    col = ffi.new("GrB_Index*")
    lib.GxB_Matrix_Iterator_getIndex(it[0], row, col)
    return (row[0], col[0])


def matrix_iterator_getp(it):
    """Get the current position of the matrix entry iterator.

    See :func:`matrix_iterator_getpmax`.
    """
    return lib.GxB_Matrix_Iterator_getp(it[0])


def matrix_iterator_getpmax(it):
    """Get the end of the matrix entry iterator's positions.

    Positions run from 0 to ``pmax - 1``.  ``pmax`` is the number of entries,
    except for a bitmap matrix, where it is ``nrows * ncols`` (or 0 if it has no
    entries) and a position may have no entry.
    """
    return lib.GxB_Matrix_Iterator_getpmax(it[0])


# ---------------------------------------------------------------------------
# Row iterator
# ---------------------------------------------------------------------------


def row_iterator_seek_row(it, row):
    """Seek the row iterator to a specific row, or the next row it holds.

    Returns ``lib.GrB_SUCCESS`` at the row's first entry, ``lib.GrB_NO_VALUE``
    if the row has no entries, or ``lib.GxB_EXHAUSTED`` if ``row >= nrows``.  For
    a hypersparse matrix, a row it doesn't hold is skipped for the next one it
    does (see :func:`row_iterator_get_row_index`).

    >>> from suitesparse_graphblas import matrix
    >>> A = matrix.matrix_new(lib.GrB_INT64, 2, 2)
    >>> matrix.matrix_option_set_int32(A, lib.GxB_SPARSITY_CONTROL, lib.GxB_SPARSE)
    >>> matrix.set_int64(A, 10, 0, 0)
    >>> matrix.set_int64(A, 20, 1, 1)
    >>> matrix.matrix_wait(A, lib.GrB_MATERIALIZE)
    >>> it = iterator_new()
    >>> row_iterator_attach(it, A) == lib.GrB_SUCCESS
    True
    >>> row_iterator_seek_row(it, 1) == lib.GrB_SUCCESS
    True
    >>> iterator_get_int64(it)
    20

    """
    info = lib.GxB_rowIterator_seekRow(it[0], row)
    return _check_info(info)


def row_iterator_next_row(it):
    """Advance the row iterator to the next row (which may skip empty rows).

    Returns as :func:`row_iterator_seek_row` does.

    >>> from suitesparse_graphblas import matrix
    >>> A = matrix.matrix_new(lib.GrB_INT64, 2, 2)
    >>> matrix.matrix_option_set_int32(A, lib.GxB_SPARSITY_CONTROL, lib.GxB_SPARSE)
    >>> matrix.set_int64(A, 10, 0, 0)
    >>> matrix.set_int64(A, 20, 1, 1)
    >>> matrix.matrix_wait(A, lib.GrB_MATERIALIZE)
    >>> it = iterator_new()
    >>> row_iterator_attach(it, A) == lib.GrB_SUCCESS
    True
    >>> row_iterator_seek_row(it, 0) == lib.GrB_SUCCESS
    True
    >>> row_iterator_get_row_index(it)
    0
    >>> row_iterator_next_row(it) == lib.GrB_SUCCESS
    True
    >>> row_iterator_get_row_index(it)
    1
    >>> row_iterator_next_row(it) == lib.GxB_EXHAUSTED
    True

    """
    info = lib.GxB_rowIterator_nextRow(it[0])
    return _check_info(info)


def row_iterator_next_col(it):
    """Advance the row iterator to the next entry in the current row.

    Returns ``lib.GrB_SUCCESS``, or ``lib.GrB_NO_VALUE`` if the row has no more
    entries.

    >>> from suitesparse_graphblas import matrix
    >>> A = matrix.matrix_new(lib.GrB_INT64, 2, 2)
    >>> matrix.matrix_option_set_int32(A, lib.GxB_SPARSITY_CONTROL, lib.GxB_SPARSE)
    >>> matrix.set_int64(A, 10, 0, 0)
    >>> matrix.set_int64(A, 20, 0, 1)
    >>> matrix.matrix_wait(A, lib.GrB_MATERIALIZE)
    >>> it = iterator_new()
    >>> row_iterator_attach(it, A) == lib.GrB_SUCCESS
    True
    >>> row_iterator_seek_row(it, 0) == lib.GrB_SUCCESS
    True
    >>> iterator_get_int64(it)
    10
    >>> row_iterator_next_col(it) == lib.GrB_SUCCESS
    True
    >>> iterator_get_int64(it)
    20
    >>> row_iterator_get_col_index(it)
    1
    >>> row_iterator_next_col(it) == lib.GrB_NO_VALUE
    True

    """
    info = lib.GxB_rowIterator_nextCol(it[0])
    return _check_info(info)


def row_iterator_kseek(it, k):
    """Seek the row iterator to the k-th row it holds, for ``0 <= k < kount``.

    That is row ``k``, unless the matrix is hypersparse (see
    :func:`row_iterator_kount`).  Returns as :func:`row_iterator_seek_row` does.
    """
    info = lib.GxB_rowIterator_kseek(it[0], k)
    return _check_info(info)


def row_iterator_get_row_index(it):
    """Get the current row index of the row iterator (``nrows`` once exhausted)."""
    return lib.GxB_rowIterator_getRowIndex(it[0])


def row_iterator_get_col_index(it):
    """Get the column index of the row iterator's current entry."""
    return lib.GxB_rowIterator_getColIndex(it[0])


def row_iterator_kount(it):
    """Get the number of rows the row iterator holds, for :func:`row_iterator_kseek`.

    This is ``nrows``, unless the matrix is hypersparse; then it is the number of
    rows in its hyperlist, which bounds the number of non-empty rows.
    """
    return lib.GxB_rowIterator_kount(it[0])


# ---------------------------------------------------------------------------
# Column iterator
# ---------------------------------------------------------------------------


def col_iterator_seek_col(it, col):
    """Seek the column iterator to a specific column, or the next column it holds.

    Returns ``lib.GrB_SUCCESS`` at the column's first entry, ``lib.GrB_NO_VALUE``
    if the column has no entries, or ``lib.GxB_EXHAUSTED`` if ``col >= ncols``.
    For a hypersparse matrix, a column it doesn't hold is skipped for the next
    one it does (see :func:`col_iterator_get_col_index`).

    >>> from suitesparse_graphblas import matrix
    >>> A = matrix.matrix_new(lib.GrB_INT64, 2, 2)
    >>> matrix.matrix_set_format(A, lib.GxB_BY_COL)
    >>> matrix.matrix_option_set_int32(A, lib.GxB_SPARSITY_CONTROL, lib.GxB_SPARSE)
    >>> matrix.set_int64(A, 10, 0, 0)
    >>> matrix.set_int64(A, 20, 1, 1)
    >>> matrix.matrix_wait(A, lib.GrB_MATERIALIZE)
    >>> it = iterator_new()
    >>> col_iterator_attach(it, A) == lib.GrB_SUCCESS
    True
    >>> col_iterator_seek_col(it, 1) == lib.GrB_SUCCESS
    True
    >>> iterator_get_int64(it)
    20

    """
    info = lib.GxB_colIterator_seekCol(it[0], col)
    return _check_info(info)


def col_iterator_next_col(it):
    """Advance the column iterator to the next column (which may skip empty columns).

    Returns as :func:`col_iterator_seek_col` does.

    >>> from suitesparse_graphblas import matrix
    >>> A = matrix.matrix_new(lib.GrB_INT64, 2, 2)
    >>> matrix.matrix_set_format(A, lib.GxB_BY_COL)
    >>> matrix.matrix_option_set_int32(A, lib.GxB_SPARSITY_CONTROL, lib.GxB_SPARSE)
    >>> matrix.set_int64(A, 10, 0, 0)
    >>> matrix.set_int64(A, 20, 1, 1)
    >>> matrix.matrix_wait(A, lib.GrB_MATERIALIZE)
    >>> it = iterator_new()
    >>> col_iterator_attach(it, A) == lib.GrB_SUCCESS
    True
    >>> col_iterator_seek_col(it, 0) == lib.GrB_SUCCESS
    True
    >>> col_iterator_get_col_index(it)
    0
    >>> col_iterator_next_col(it) == lib.GrB_SUCCESS
    True
    >>> col_iterator_get_col_index(it)
    1
    >>> col_iterator_next_col(it) == lib.GxB_EXHAUSTED
    True

    """
    info = lib.GxB_colIterator_nextCol(it[0])
    return _check_info(info)


def col_iterator_next_row(it):
    """Advance the column iterator to the next entry in the current column.

    Returns ``lib.GrB_SUCCESS``, or ``lib.GrB_NO_VALUE`` if the column has no
    more entries.
    """
    info = lib.GxB_colIterator_nextRow(it[0])
    return _check_info(info)


def col_iterator_kseek(it, k):
    """Seek the column iterator to the k-th column it holds, for ``0 <= k < kount``.

    That is column ``k``, unless the matrix is hypersparse (see
    :func:`col_iterator_kount`).  Returns as :func:`col_iterator_seek_col` does.
    """
    info = lib.GxB_colIterator_kseek(it[0], k)
    return _check_info(info)


def col_iterator_get_col_index(it):
    """Get the current column index of the column iterator (``ncols`` once exhausted)."""
    return lib.GxB_colIterator_getColIndex(it[0])


def col_iterator_get_row_index(it):
    """Get the row index of the column iterator's current entry."""
    return lib.GxB_colIterator_getRowIndex(it[0])


def col_iterator_kount(it):
    """Get the number of columns the column iterator holds, for :func:`col_iterator_kseek`.

    This is ``ncols``, unless the matrix is hypersparse; then it is the number of
    columns in its hyperlist, which bounds the number of non-empty columns.
    """
    return lib.GxB_colIterator_kount(it[0])


# ---------------------------------------------------------------------------
# Vector iterator
# ---------------------------------------------------------------------------


def vector_iterator_seek(it, p):
    """Seek the vector iterator to position p.

    Returns ``lib.GrB_SUCCESS``, or ``lib.GxB_EXHAUSTED`` if ``p >= pmax``.  For a
    bitmap vector, the iterator moves to the first entry at or after ``p``, and is
    exhausted if there is none.

    >>> from suitesparse_graphblas import vector
    >>> v = vector.vector_new(lib.GrB_INT64, 3)
    >>> vector.set_int64(v, 42, 0)
    >>> vector.set_int64(v, 7, 2)
    >>> vector.vector_wait(v, lib.GrB_MATERIALIZE)
    >>> it = iterator_new()
    >>> vector_iterator_attach(it, v) == lib.GrB_SUCCESS
    True
    >>> vector_iterator_seek(it, 0) == lib.GrB_SUCCESS
    True
    >>> iterator_get_int64(it)
    42

    """
    info = lib.GxB_Vector_Iterator_seek(it[0], p)
    return _check_info(info)


def vector_iterator_next(it):
    """Advance the vector iterator to the next entry.

    Returns ``lib.GrB_SUCCESS`` or ``lib.GxB_EXHAUSTED``.

    >>> from suitesparse_graphblas import vector
    >>> v = vector.vector_new(lib.GrB_INT64, 3)
    >>> vector.set_int64(v, 42, 0)
    >>> vector.set_int64(v, 7, 2)
    >>> vector.vector_wait(v, lib.GrB_MATERIALIZE)
    >>> it = iterator_new()
    >>> vector_iterator_attach(it, v) == lib.GrB_SUCCESS
    True
    >>> vector_iterator_seek(it, 0) == lib.GrB_SUCCESS
    True
    >>> vector_iterator_get_index(it)
    0
    >>> iterator_get_int64(it)
    42
    >>> vector_iterator_next(it) == lib.GrB_SUCCESS
    True
    >>> vector_iterator_get_index(it)
    2
    >>> iterator_get_int64(it)
    7
    >>> vector_iterator_next(it) == lib.GxB_EXHAUSTED
    True

    """
    info = lib.GxB_Vector_Iterator_next(it[0])
    return _check_info(info)


def vector_iterator_get_index(it):
    """Get the current index of the vector iterator."""
    return lib.GxB_Vector_Iterator_getIndex(it[0])


def vector_iterator_getp(it):
    """Get the current position of the vector iterator.

    See :func:`vector_iterator_getpmax`.
    """
    return lib.GxB_Vector_Iterator_getp(it[0])


def vector_iterator_getpmax(it):
    """Get the end of the vector iterator's positions.

    Positions run from 0 to ``pmax - 1``.  ``pmax`` is the number of entries,
    except for a bitmap vector, where it is its size (or 0 if it has no entries)
    and a position may have no entry.
    """
    return lib.GxB_Vector_Iterator_getpmax(it[0])


# ---------------------------------------------------------------------------
# Value getters (shared across all iteration modes)
# ---------------------------------------------------------------------------


def iterator_get_bool(it):
    """Get the current boolean value from the iterator."""
    return lib.GxB_Iterator_get_BOOL(it[0])


def iterator_get_int8(it):
    """Get the current int8 value from the iterator."""
    return lib.GxB_Iterator_get_INT8(it[0])


def iterator_get_int16(it):
    """Get the current int16 value from the iterator."""
    return lib.GxB_Iterator_get_INT16(it[0])


def iterator_get_int32(it):
    """Get the current int32 value from the iterator."""
    return lib.GxB_Iterator_get_INT32(it[0])


def iterator_get_int64(it):
    """Get the current int64 value from the iterator."""
    return lib.GxB_Iterator_get_INT64(it[0])


def iterator_get_uint8(it):
    """Get the current uint8 value from the iterator."""
    return lib.GxB_Iterator_get_UINT8(it[0])


def iterator_get_uint16(it):
    """Get the current uint16 value from the iterator."""
    return lib.GxB_Iterator_get_UINT16(it[0])


def iterator_get_uint32(it):
    """Get the current uint32 value from the iterator."""
    return lib.GxB_Iterator_get_UINT32(it[0])


def iterator_get_uint64(it):
    """Get the current uint64 value from the iterator."""
    return lib.GxB_Iterator_get_UINT64(it[0])


def iterator_get_fp32(it):
    """Get the current fp32 value from the iterator."""
    return lib.GxB_Iterator_get_FP32(it[0])


def iterator_get_fp64(it):
    """Get the current fp64 value from the iterator."""
    return lib.GxB_Iterator_get_FP64(it[0])


if supports_complex():  # pragma: no branch (the build always declares complex types)

    def iterator_get_fc32(it):
        """Get the current fc32 value from the iterator."""
        return lib.GxB_Iterator_get_FC32(it[0])

    def iterator_get_fc64(it):
        """Get the current fc64 value from the iterator."""
        return lib.GxB_Iterator_get_FC64(it[0])
