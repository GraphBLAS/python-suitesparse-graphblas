import pytest

from suitesparse_graphblas import (
    bool_types,
    complex_types,
    exceptions,
    ffi,
    grb_types,
    lib,
    matrix,
    real_types,
    signed_integer_types,
    supports_complex,
    vector,
)
from suitesparse_graphblas.api import iterator

SUCCESS = lib.GrB_SUCCESS
NO_VALUE = lib.GrB_NO_VALUE
EXHAUSTED = lib.GxB_EXHAUSTED

_sparsities = {
    lib.GxB_HYPERSPARSE: "hypersparse",
    lib.GxB_SPARSE: "sparse",
    lib.GxB_BITMAP: "bitmap",
    lib.GxB_FULL: "full",
}

# 5x4, with rows 1 and 3 and column 1 empty
_shape = (5, 4)
_entries = {(0, 2): 10, (2, 3): 20, (2, 0): 30, (4, 2): 40, (0, 0): 50}


def _matrix(format, sparsity):
    """An INT64 matrix with `_entries`, or with every entry (i * 10 + j) if `sparsity` is full."""
    A = matrix.matrix_new(lib.GrB_INT64, *_shape)
    if sparsity == lib.GxB_FULL:
        entries = {(i, j): i * 10 + j for i in range(_shape[0]) for j in range(_shape[1])}
    else:
        entries = _entries
    for (i, j), value in entries.items():
        matrix.set_int64(A, value, i, j)
    matrix.matrix_set_format(A, format)
    matrix.matrix_set_sparsity_control(A, sparsity)
    matrix.matrix_wait(A, lib.GrB_MATERIALIZE)
    assert matrix.matrix_sparsity_status(A) == sparsity
    return A, sorted((i, j, value) for (i, j), value in entries.items())


def test_check_info():
    for info in [SUCCESS, NO_VALUE, EXHAUSTED]:
        assert iterator._check_info(info) == info
    with pytest.raises(exceptions.InvalidIndex):
        iterator._check_info(lib.GrB_INVALID_INDEX)
    with pytest.raises(RuntimeError, match="info=-12345"):
        iterator._check_info(-12345)


def test_iterator_new_free():
    it = iterator.iterator_new(free=None)
    assert it[0] != ffi.NULL
    iterator.iterator_free(it)
    assert it[0] == ffi.NULL

    freed = []

    def free(it):
        freed.append(it)
        iterator.iterator_free(it)

    it = iterator.iterator_new(free=free)
    assert freed == []
    ffi.release(it)
    assert len(freed) == 1
    assert freed[0][0] == ffi.NULL


@pytest.mark.parametrize("sparsity", _sparsities, ids=_sparsities.get)
@pytest.mark.parametrize("format", [lib.GxB_BY_ROW, lib.GxB_BY_COL], ids=["by_row", "by_col"])
def test_matrix_iterator(format, sparsity):
    A, expected = _matrix(format, sparsity)
    it = iterator.iterator_new()
    assert iterator.matrix_iterator_attach(it, A) == SUCCESS
    pmax = iterator.matrix_iterator_getpmax(it)
    if sparsity == lib.GxB_BITMAP:
        assert pmax == _shape[0] * _shape[1]
    else:
        assert pmax == len(expected)
    entries = []
    info = iterator.matrix_iterator_seek(it, 0)
    while info != EXHAUSTED:
        assert info == SUCCESS
        p = iterator.matrix_iterator_getp(it)
        assert 0 <= p < pmax
        i, j = iterator.matrix_iterator_get_index(it)
        entries.append((i, j, iterator.iterator_get_int64(it)))
        info = iterator.matrix_iterator_next(it)
    assert sorted(entries) == expected
    # Held by row, the entries come by row; held by column, by column
    if format == lib.GxB_BY_ROW:
        assert entries == expected
    else:
        assert entries == sorted(expected, key=lambda e: (e[1], e[0]))

    # Seeking a position goes to the entry there, or (for bitmap) the next one
    for p in range(pmax):
        assert iterator.matrix_iterator_seek(it, p) == SUCCESS or sparsity == lib.GxB_BITMAP
        if sparsity != lib.GxB_BITMAP:
            assert iterator.matrix_iterator_getp(it) == p
    assert iterator.matrix_iterator_seek(it, pmax) == EXHAUSTED


def _row_iterate(it, seek, next_vector, next_entry, get_vector_index, get_entry_index):
    """Iterate over the entries of A as GraphBLAS's documentation does."""
    entries = []
    info = seek(it, 0)
    while info != EXHAUSTED:
        k = get_vector_index(it)
        while info == SUCCESS:
            entries.append((k, get_entry_index(it), iterator.iterator_get_int64(it)))
            info = next_entry(it)
        assert info == NO_VALUE
        info = next_vector(it)
    return entries


@pytest.mark.parametrize("sparsity", _sparsities, ids=_sparsities.get)
def test_row_iterator(sparsity):
    A, expected = _matrix(lib.GxB_BY_ROW, sparsity)
    nrows = _shape[0]
    it = iterator.iterator_new()
    assert iterator.row_iterator_attach(it, A) == SUCCESS
    entries = _row_iterate(
        it,
        iterator.row_iterator_seek_row,
        iterator.row_iterator_next_row,
        iterator.row_iterator_next_col,
        iterator.row_iterator_get_row_index,
        iterator.row_iterator_get_col_index,
    )
    assert entries == expected
    assert iterator.row_iterator_get_row_index(it) == nrows  # exhausted

    rows = sorted({i for i, _, _ in expected})
    kount = iterator.row_iterator_kount(it)
    if sparsity == lib.GxB_HYPERSPARSE:
        assert len(rows) <= kount <= nrows
    else:
        assert kount == nrows
    visited = []
    for k in range(kount):
        info = iterator.row_iterator_kseek(it, k)
        i = iterator.row_iterator_get_row_index(it)
        if sparsity != lib.GxB_HYPERSPARSE:
            assert i == k
        assert info == (SUCCESS if i in rows else NO_VALUE)
        visited.append(i)
    assert set(rows) <= set(visited)
    assert iterator.row_iterator_kseek(it, kount) == EXHAUSTED

    # Seeking an empty row: it has no entries, or (if hypersparse) the next row is found
    info = iterator.row_iterator_seek_row(it, 1)
    if sparsity == lib.GxB_HYPERSPARSE:
        assert info == SUCCESS
        assert iterator.row_iterator_get_row_index(it) == 2
    else:
        assert info == (SUCCESS if sparsity == lib.GxB_FULL else NO_VALUE)
        assert iterator.row_iterator_get_row_index(it) == 1
    assert iterator.row_iterator_seek_row(it, nrows) == EXHAUSTED


@pytest.mark.parametrize("sparsity", _sparsities, ids=_sparsities.get)
def test_col_iterator(sparsity):
    A, expected = _matrix(lib.GxB_BY_COL, sparsity)
    expected = sorted((j, i, value) for i, j, value in expected)
    ncols = _shape[1]
    it = iterator.iterator_new()
    assert iterator.col_iterator_attach(it, A) == SUCCESS
    entries = _row_iterate(
        it,
        iterator.col_iterator_seek_col,
        iterator.col_iterator_next_col,
        iterator.col_iterator_next_row,
        iterator.col_iterator_get_col_index,
        iterator.col_iterator_get_row_index,
    )
    assert entries == expected
    assert iterator.col_iterator_get_col_index(it) == ncols  # exhausted

    cols = sorted({j for j, _, _ in expected})
    kount = iterator.col_iterator_kount(it)
    if sparsity == lib.GxB_HYPERSPARSE:
        assert len(cols) <= kount <= ncols
    else:
        assert kount == ncols
    visited = []
    for k in range(kount):
        info = iterator.col_iterator_kseek(it, k)
        j = iterator.col_iterator_get_col_index(it)
        if sparsity != lib.GxB_HYPERSPARSE:
            assert j == k
        assert info == (SUCCESS if j in cols else NO_VALUE)
        visited.append(j)
    assert set(cols) <= set(visited)
    assert iterator.col_iterator_kseek(it, kount) == EXHAUSTED

    # Seeking an empty column: it has no entries, or (if hypersparse) the next column is found
    info = iterator.col_iterator_seek_col(it, 1)
    if sparsity == lib.GxB_HYPERSPARSE:
        assert info == SUCCESS
        assert iterator.col_iterator_get_col_index(it) == 2
    else:
        assert info == (SUCCESS if sparsity == lib.GxB_FULL else NO_VALUE)
        assert iterator.col_iterator_get_col_index(it) == 1
    assert iterator.col_iterator_seek_col(it, ncols) == EXHAUSTED


def test_row_and_col_iterators_need_their_format():
    it = iterator.iterator_new()
    by_col, _ = _matrix(lib.GxB_BY_COL, lib.GxB_SPARSE)
    with pytest.raises(exceptions.NotImplementedException):
        iterator.row_iterator_attach(it, by_col)
    by_row, _ = _matrix(lib.GxB_BY_ROW, lib.GxB_SPARSE)
    with pytest.raises(exceptions.NotImplementedException):
        iterator.col_iterator_attach(it, by_row)
    # The entry iterator takes either
    assert iterator.matrix_iterator_attach(it, by_col) == SUCCESS
    assert iterator.matrix_iterator_attach(it, by_row) == SUCCESS


@pytest.mark.parametrize("sparsity", _sparsities, ids=_sparsities.get)
def test_vector_iterator(sparsity):
    size = 6
    v = vector.vector_new(lib.GrB_FP64, size)
    if sparsity == lib.GxB_FULL:
        expected = [(i, i + 0.5) for i in range(size)]
    else:
        expected = [(1, 1.5), (4, -2.5), (5, 0.25)]
    for i, value in expected:
        vector.set_fp64(v, value, i)
    check = lib.GxB_Vector_Option_set_INT32(v[0], lib.GxB_SPARSITY_CONTROL, sparsity)
    assert check == SUCCESS
    vector.vector_wait(v, lib.GrB_MATERIALIZE)

    it = iterator.iterator_new()
    assert iterator.vector_iterator_attach(it, v) == SUCCESS
    pmax = iterator.vector_iterator_getpmax(it)
    assert pmax == (size if sparsity == lib.GxB_BITMAP else len(expected))
    entries = []
    info = iterator.vector_iterator_seek(it, 0)
    while info != EXHAUSTED:
        assert info == SUCCESS
        assert 0 <= iterator.vector_iterator_getp(it) < pmax
        entries.append((iterator.vector_iterator_get_index(it), iterator.iterator_get_fp64(it)))
        info = iterator.vector_iterator_next(it)
    assert entries == expected
    if sparsity == lib.GxB_BITMAP:
        # Position 2 has no entry, so seeking it finds the one at 4
        assert iterator.vector_iterator_seek(it, 2) == SUCCESS
        assert iterator.vector_iterator_getp(it) == 4
        assert iterator.vector_iterator_get_index(it) == 4
    else:
        assert iterator.vector_iterator_seek(it, 1) == SUCCESS
        assert iterator.vector_iterator_getp(it) == 1
    assert iterator.vector_iterator_seek(it, pmax) == EXHAUSTED


def test_vector_iterator_empty():
    v = vector.vector_new(lib.GrB_INT64, 3)
    it = iterator.iterator_new()
    assert iterator.vector_iterator_attach(it, v) == SUCCESS
    assert iterator.vector_iterator_getpmax(it) == 0
    assert iterator.vector_iterator_seek(it, 0) == EXHAUSTED


_type_names = {
    lib.GrB_BOOL: "bool",
    lib.GrB_INT8: "int8",
    lib.GrB_INT16: "int16",
    lib.GrB_INT32: "int32",
    lib.GrB_INT64: "int64",
    lib.GrB_UINT8: "uint8",
    lib.GrB_UINT16: "uint16",
    lib.GrB_UINT32: "uint32",
    lib.GrB_UINT64: "uint64",
    lib.GrB_FP32: "fp32",
    lib.GrB_FP64: "fp64",
}
if supports_complex():
    _type_names[lib.GxB_FC32] = "fc32"
    _type_names[lib.GxB_FC64] = "fc64"


def _extreme_values(T):
    """Values of type T, including its limits, that its C type holds exactly."""
    name = _type_names[T]
    if T in bool_types:
        return [True, False]
    if T in signed_integer_types:
        bits = int(name[3:])
        return [-(2 ** (bits - 1)), -1, 2 ** (bits - 1) - 1]
    if T in real_types:
        return [-0.5, 2.0**100, 2.0**-20]
    if T in complex_types:
        return [complex(-0.5, 2.0), complex(2.0**100, -1.0), 2.0j]
    bits = int(name[4:])
    return [0, 1, 2**bits - 1]


@pytest.mark.parametrize("T", sorted(grb_types, key=_type_names.get), ids=_type_names.get)
def test_iterator_get(T):
    name = _type_names[T]
    values = _extreme_values(T)
    v = vector.vector_new(T, len(values))
    setter = getattr(vector, f"set_{name}")
    for i, value in enumerate(values):
        setter(v, value, i)
    vector.vector_wait(v, lib.GrB_MATERIALIZE)
    get = getattr(iterator, f"iterator_get_{name}")
    it = iterator.iterator_new()
    iterator.vector_iterator_attach(it, v)
    got = []
    info = iterator.vector_iterator_seek(it, 0)
    while info == SUCCESS:
        got.append(get(it))
        info = iterator.vector_iterator_next(it)
    assert got == values
    assert all(type(x) is type(y) for x, y in zip(got, values, strict=True))
