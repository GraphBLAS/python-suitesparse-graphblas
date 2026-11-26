import bz2
import functools
import gzip
import platform
import struct
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from suitesparse_graphblas import (
    bool_types,
    check_status,
    complex_types,
    ffi,
    grb_types,
    lib,
    matrix,
    real_types,
    signed_integer_types,
    supports_complex,
    unsigned_integer_types,
    vector,
)

try:
    import lzma
except ImportError:  # Pyodide before 314 has it as a separate package
    lzma = None

if platform.system() == "Windows":
    pytest.skip("skipping windows-only tests", allow_module_level=True)

from suitesparse_graphblas.api import context  # isort:skip
from suitesparse_graphblas.api.global_options import global_get_int32  # isort:skip
from suitesparse_graphblas.exceptions import InvalidObject, InvalidValue  # isort:skip
from suitesparse_graphblas.io import binary, serialize  # isort:skip

NULL = ffi.NULL


def _test_elements(T):
    if T in bool_types:
        return [True, False], [0, 0], [1, 1]
    elif T in signed_integer_types:
        return [1, -42], [0, 0], [1, 1]
    elif T in unsigned_integer_types:
        return [1, 42], [0, 0], [1, 1]
    elif T in real_types:
        return [1.0, -42.42], [0, 0], [1, 1]
    elif T in complex_types:
        return [complex(1.0, 1.0), complex(42.0, -42.0)], [0, 0], [1, 1]


_element_setters = {
    lib.GrB_BOOL: lib.GrB_Matrix_setElement_BOOL,
    lib.GrB_INT8: lib.GrB_Matrix_setElement_INT8,
    lib.GrB_INT16: lib.GrB_Matrix_setElement_INT16,
    lib.GrB_INT32: lib.GrB_Matrix_setElement_INT32,
    lib.GrB_INT64: lib.GrB_Matrix_setElement_INT64,
    lib.GrB_UINT8: lib.GrB_Matrix_setElement_UINT8,
    lib.GrB_UINT16: lib.GrB_Matrix_setElement_UINT16,
    lib.GrB_UINT32: lib.GrB_Matrix_setElement_UINT32,
    lib.GrB_UINT64: lib.GrB_Matrix_setElement_UINT64,
    lib.GrB_FP32: lib.GrB_Matrix_setElement_FP32,
    lib.GrB_FP64: lib.GrB_Matrix_setElement_FP64,
}

if supports_complex():
    _element_setters.update(
        {
            lib.GxB_FC32: lib.GxB_Matrix_setElement_FC32,
            lib.GxB_FC64: lib.GxB_Matrix_setElement_FC64,
        }
    )


_eq_ops = {
    lib.GrB_BOOL: lib.GrB_EQ_BOOL,
    lib.GrB_INT8: lib.GrB_EQ_INT8,
    lib.GrB_INT16: lib.GrB_EQ_INT16,
    lib.GrB_INT32: lib.GrB_EQ_INT32,
    lib.GrB_INT64: lib.GrB_EQ_INT64,
    lib.GrB_UINT8: lib.GrB_EQ_UINT8,
    lib.GrB_UINT16: lib.GrB_EQ_UINT16,
    lib.GrB_UINT32: lib.GrB_EQ_UINT32,
    lib.GrB_UINT64: lib.GrB_EQ_UINT64,
    lib.GrB_FP32: lib.GrB_EQ_FP32,
    lib.GrB_FP64: lib.GrB_EQ_FP64,
}

if supports_complex():
    _eq_ops.update(
        {
            lib.GxB_FC32: lib.GxB_EQ_FC32,
            lib.GxB_FC64: lib.GxB_EQ_FC64,
        }
    )


def test_serialize_matrix():
    T = lib.GrB_INT64
    A = matrix.matrix_new(T, 2, 2)
    for args in zip(*_test_elements(T)):
        f = _element_setters[T]
        check_status(A, f(A[0], *args))
    data = matrix.serialize(A)
    B = matrix.deserialize(data)

    # Test equal
    C = matrix.matrix_new(lib.GrB_BOOL, 2, 2)
    check_status(
        C,
        lib.GrB_Matrix_eWiseAdd_BinaryOp(C[0], NULL, NULL, _eq_ops[T], A[0], B[0], NULL),
    )
    assert matrix.matrix_nvals(A) == matrix.matrix_nvals(B) == matrix.matrix_nvals(C)
    is_eq = ffi.new("bool*")
    check_status(
        C,
        lib.GrB_Matrix_reduce_BOOL(is_eq, NULL, lib.GrB_LAND_MONOID_BOOL, C[0], NULL),
    )
    assert is_eq[0]


def test_serialize_vector():
    T = lib.GrB_INT64
    v = vector.vector_new(T, 3)
    check_status(v, lib.GrB_Vector_setElement_INT64(v[0], 2, 0))
    check_status(v, lib.GrB_Vector_setElement_INT64(v[0], 10, 1))
    data = vector.serialize(v, lib.GxB_COMPRESSION_LZ4HC, level=7)
    w = vector.deserialize(data)

    # Test equal
    x = vector.vector_new(lib.GrB_BOOL, 3)
    check_status(
        x,
        lib.GrB_Vector_eWiseAdd_BinaryOp(x[0], NULL, NULL, _eq_ops[T], v[0], w[0], NULL),
    )
    assert vector.vector_nvals(v) == vector.vector_nvals(w) == vector.vector_nvals(x)
    is_eq = ffi.new("bool*")
    check_status(
        x,
        lib.GrB_Vector_reduce_BOOL(is_eq, NULL, lib.GrB_LAND_MONOID_BOOL, x[0], NULL),
    )
    assert is_eq[0]


def test_matrix_binfile_read_write(tmp_path):
    openers = [Path.open, gzip.open, bz2.open]
    if lzma is not None:
        openers.append(lzma.open)
    for opener in openers:
        for format in (lib.GxB_BY_ROW, lib.GxB_BY_COL):
            for T in grb_types:
                for sparsity in (lib.GxB_HYPERSPARSE, lib.GxB_SPARSE, lib.GxB_BITMAP, lib.GxB_FULL):
                    A = matrix.matrix_new(T, 2, 2)

                    if sparsity == lib.GxB_FULL:
                        # A matrix can only be full if it has a value everywhere
                        values = _test_elements(T)[0]
                        for k, (i, j) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                            check_status(A, _element_setters[T](A[0], values[k % 2], i, j))
                    else:
                        for args in zip(*_test_elements(T)):
                            f = _element_setters[T]
                            check_status(A, f(A[0], *args))
                    matrix.matrix_set_sparsity_control(A, sparsity)
                    matrix.matrix_set_format(A, format)
                    assert matrix.matrix_sparsity_status(A) == sparsity

                    binfilef = tmp_path / "binfilewrite_test.binfile"
                    binary.binwrite(A, binfilef, opener=opener)
                    B = binary.binread(binfilef, opener=opener)

                    assert matrix.matrix_type(A) == matrix.matrix_type(B)
                    assert matrix.matrix_nrows(A) == matrix.matrix_nrows(B)
                    assert matrix.matrix_ncols(A) == matrix.matrix_ncols(B)
                    assert matrix.matrix_hyper_switch(A) == matrix.matrix_hyper_switch(B)
                    assert matrix.matrix_bitmap_switch(A) == matrix.matrix_bitmap_switch(B)
                    assert matrix.matrix_sparsity_control(A) == matrix.matrix_sparsity_control(B)
                    assert matrix.matrix_sparsity_status(B) == sparsity
                    assert matrix.matrix_format(B) == format

                    C = matrix.matrix_new(lib.GrB_BOOL, 2, 2)

                    check_status(
                        C,
                        lib.GrB_Matrix_eWiseAdd_BinaryOp(
                            C[0], NULL, NULL, _eq_ops[T], A[0], B[0], NULL
                        ),
                    )

                    assert (
                        matrix.matrix_nvals(A) == matrix.matrix_nvals(B) == matrix.matrix_nvals(C)
                    )

                    is_eq = ffi.new("bool*")
                    check_status(
                        C,
                        lib.GrB_Matrix_reduce_BOOL(
                            is_eq, NULL, lib.GrB_LAND_MONOID_BOOL, C[0], NULL
                        ),
                    )

                    assert is_eq[0]


def test_binwrite_failure_keeps_matrix(tmp_path):
    # `binwrite` unpacks the matrix to write its arrays, and must put them back regardless
    A = matrix.matrix_new(lib.GrB_INT64, 2, 2)
    check_status(A, lib.GrB_Matrix_setElement_INT64(A[0], 1, 0, 0))
    with pytest.raises(OSError):
        binary.binwrite(A, tmp_path / "missing_directory" / "matrix.binfile")
    assert matrix.matrix_nvals(A) == 1


_extract_tuples = {
    lib.GrB_BOOL: matrix.matrix_extract_tuples_bool,
    lib.GrB_INT8: matrix.matrix_extract_tuples_int8,
    lib.GrB_INT16: matrix.matrix_extract_tuples_int16,
    lib.GrB_INT32: matrix.matrix_extract_tuples_int32,
    lib.GrB_INT64: matrix.matrix_extract_tuples_int64,
    lib.GrB_UINT8: matrix.matrix_extract_tuples_uint8,
    lib.GrB_UINT16: matrix.matrix_extract_tuples_uint16,
    lib.GrB_UINT32: matrix.matrix_extract_tuples_uint32,
    lib.GrB_UINT64: matrix.matrix_extract_tuples_uint64,
    lib.GrB_FP32: matrix.matrix_extract_tuples_fp32,
    lib.GrB_FP64: matrix.matrix_extract_tuples_fp64,
}

if supports_complex():
    _extract_tuples.update(
        {
            lib.GxB_FC32: matrix.matrix_extract_tuples_fc32,
            lib.GxB_FC64: matrix.matrix_extract_tuples_fc64,
        }
    )

_sparsities = {
    lib.GxB_HYPERSPARSE: "hypersparse",
    lib.GxB_SPARSE: "sparse",
    lib.GxB_BITMAP: "bitmap",
    lib.GxB_FULL: "full",
}

# 3x4, so a matrix held by row and by column differ; "sparse" leaves row 1 and column 2 empty
_shape = (3, 4)
_sparse_indices = [(0, 1), (2, 3), (2, 0)]
_full_indices = [(i, j) for i in range(_shape[0]) for j in range(_shape[1])]


def _values(T):
    """Four distinct values (two for BOOL) of type T, for entries to cycle through."""
    if T in bool_types:
        return [True, False]
    if T in signed_integer_types:
        return [1, -42, 7, -100]
    if T in unsigned_integer_types:
        return [1, 42, 7, 200]
    if T in real_types:
        return [1.0, -42.5, 0.25, 3.0e5]
    return [complex(1.0, 1.0), complex(42.0, -42.0), -0.5j, complex(3.0, 0.0)]


def _make_matrix(T, kind):
    """A matrix of type T: "empty", "sparse", "full", or iso ("iso_sparse", "iso_full")."""
    A = matrix.matrix_new(T, *_shape)
    indices = {"empty": [], "sparse": _sparse_indices, "iso_sparse": _sparse_indices}.get(
        kind, _full_indices
    )
    values = _values(T)
    setter = _element_setters[T]
    for k, (i, j) in enumerate(indices):
        value = values[0] if kind.startswith("iso") else values[k % len(values)]
        check_status(A, setter(A[0], value, i, j))
    return A


def _is_iso(A):
    iso = ffi.new("bool*")
    check_status(A, lib.GxB_Matrix_iso(iso, A[0]))
    return iso[0]


def _settings(A):
    return {
        "format": matrix.matrix_format(A),
        "sparsity_status": matrix.matrix_sparsity_status(A),
        "sparsity_control": matrix.matrix_sparsity_control(A),
        "hyper_switch": matrix.matrix_hyper_switch(A),
        "bitmap_switch": matrix.matrix_bitmap_switch(A),
        "iso": _is_iso(A),
    }


def _assert_matrices_equal(A, B):
    T = matrix.matrix_type(A)
    assert matrix.matrix_type(B) == T
    assert matrix.matrix_shape(B) == matrix.matrix_shape(A)
    for x, y in zip(_extract_tuples[T](A), _extract_tuples[T](B), strict=True):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("sparsity", _sparsities, ids=_sparsities.get)
@pytest.mark.parametrize(
    "T", sorted(grb_types, key=binary._ss_typecodes.get), ids=binary._ss_typenames.get
)
def test_binwrite_binread_round_trip(tmp_path, T, sparsity):
    path = tmp_path / "matrix.binfile"
    for format in (lib.GxB_BY_ROW, lib.GxB_BY_COL):
        for kind in ("empty", "sparse", "iso_sparse", "iso_full"):
            A = _make_matrix(T, kind)
            matrix.matrix_set_format(A, format)
            matrix.matrix_set_sparsity_control(A, sparsity)
            matrix.matrix_set_hyper_switch(A, 0.25)
            matrix.matrix_set_bitmap_switch(A, 0.75)
            matrix.matrix_wait(A, lib.GrB_MATERIALIZE)
            expected = _settings(A)
            # The cases cover what they say: GraphBLAS makes a matrix iso when its values are,
            # and holds a matrix with every entry as asked (bitmap stands in for full otherwise)
            assert expected["iso"] == kind.startswith("iso")
            if kind.endswith("full"):
                assert expected["sparsity_status"] == sparsity

            binary.binwrite(A, path)
            B = binary.binread(path)

            _assert_matrices_equal(A, B)
            assert _settings(B) == expected, (format, kind)


_openers = {"open": Path.open, "gzip": gzip.open, "bz2": bz2.open}
if lzma is not None:
    _openers["lzma"] = lzma.open


@pytest.mark.parametrize("opener", _openers.values(), ids=_openers)
def test_binwrite_binread_opener(tmp_path, opener):
    A = _make_matrix(lib.GrB_FP64, "sparse")
    matrix.matrix_set_sparsity_control(A, lib.GxB_HYPERSPARSE)
    path = str(tmp_path / "matrix.binfile")  # a str works as well as a Path
    binary.binwrite(A, path, opener=opener, comments="three entries")
    B = binary.binread(path, opener=opener)
    _assert_matrices_equal(A, B)

    # The file was written through `opener`, and starts with a readable header
    with opener(path, "rb") as f:
        header = f.read(binary.GRB_HEADER_LEN)
    assert header.startswith(b"SuiteSparse:GraphBLAS matrix\n")
    lines = header.decode("ascii").split("\n")
    for line in ["nrows:   3", "ncols:   4", "nvals:   3", "format:  HCSR", "type:    GrB_FP64"]:
        assert line in lines
    assert "three entries" in lines
    with Path(path).open("rb") as f:
        assert f.read(len(header)).startswith(header) == (opener is Path.open)


@pytest.mark.parametrize("fails", [False, True], ids=["written", "failed"])
@pytest.mark.parametrize("sparsity", _sparsities, ids=_sparsities.get)
def test_binwrite_keeps_matrix_settings(tmp_path, sparsity, fails):
    # Packing the arrays back into the matrix resets its sparsity control and switches to the
    # defaults, and `binwrite` must leave the matrix as it was, whether or not writing worked
    A = _make_matrix(lib.GrB_INT64, "full")
    matrix.matrix_set_sparsity_control(A, sparsity)
    matrix.matrix_set_hyper_switch(A, 0.25)
    matrix.matrix_set_bitmap_switch(A, 0.75)
    matrix.matrix_wait(A, lib.GrB_MATERIALIZE)
    expected = _settings(A)
    path = tmp_path / ("missing_directory" if fails else "") / "matrix.binfile"
    if fails:
        with pytest.raises(OSError):
            binary.binwrite(A, path)
    else:
        binary.binwrite(A, path)
    assert _settings(A) == expected
    _assert_matrices_equal(A, _make_matrix(lib.GrB_INT64, "full"))


def test_binwrite_header_error_keeps_matrix(tmp_path):
    # The header (which must be ASCII) is made after the matrix is unpacked, so a failure there
    # must also put the arrays back
    A = _make_matrix(lib.GrB_INT64, "sparse")
    path = tmp_path / "matrix.binfile"
    with pytest.raises(UnicodeEncodeError):
        binary.binwrite(A, path, comments="café")
    assert not path.exists()
    _assert_matrices_equal(A, _make_matrix(lib.GrB_INT64, "sparse"))


@pytest.mark.parametrize("sparsity", _sparsities, ids=_sparsities.get)
def test_binread_truncated(tmp_path, sparsity):
    A = _make_matrix(lib.GrB_INT64, "sparse")
    matrix.matrix_set_sparsity_control(A, sparsity)
    path = tmp_path / "matrix.binfile"
    binary.binwrite(A, path)
    data = path.read_bytes()
    _assert_matrices_equal(A, binary.binread(path))
    for size in range(len(data)):
        path.write_bytes(data[:size])
        with pytest.raises(EOFError, match="truncated"):
            binary.binread(path)


# Where binwrite puts the matrix's format and sparsity status (int32 each), after the text
# header and the GraphBLAS version
_format_offset = binary.GRB_HEADER_LEN + ffi.sizeof("uint64_t")
_sparsity_offset = _format_offset + ffi.sizeof("GxB_Format_Value")


@pytest.mark.parametrize(
    "offset, value, message",
    [
        (_format_offset, 99, "Unknown format 99"),
        (_sparsity_offset, 3, "Unknown sparsity status 3"),
    ],
    ids=["format", "sparsity"],
)
def test_binread_unknown_field(tmp_path, offset, value, message):
    path = tmp_path / "matrix.binfile"
    binary.binwrite(_make_matrix(lib.GrB_INT64, "sparse"), path)
    data = bytearray(path.read_bytes())
    struct.pack_into("=i", data, offset, value)
    path.write_bytes(data)
    with pytest.raises(TypeError, match=message):
        binary.binread(path)


def test_binwrite_long_comments(tmp_path):
    # binread skips a header of exactly GRB_HEADER_LEN bytes, so binwrite refuses
    # comments that would make it longer, before it opens the file
    A = _make_matrix(lib.GrB_INT64, "sparse")
    path = tmp_path / "matrix.binfile"
    binary.binwrite(A, path, comments="")
    used = len(path.read_bytes()[: binary.GRB_HEADER_LEN].rstrip(b" "))
    room = binary.GRB_HEADER_LEN - used
    # The longest comment that fits round-trips
    binary.binwrite(A, path, comments="x" * room)
    _assert_matrices_equal(A, binary.binread(path))
    # One more character is refused, and nothing is written
    path.unlink()
    with pytest.raises(ValueError, match="comments are too long"):
        binary.binwrite(A, path, comments="x" * (room + 1))
    assert not path.exists()
    with pytest.raises(ValueError, match="comments are too long"):
        binary.binwrite(A, path, comments="x" * binary.GRB_HEADER_LEN)
    assert not path.exists()
    _assert_matrices_equal(A, _make_matrix(lib.GrB_INT64, "sparse"))


def _compression(desc):
    value = ffi.new("int32_t*")
    check_status(desc, lib.GrB_Descriptor_get_INT32(desc, value, lib.GxB_COMPRESSION))
    return value[0]


def test_get_serialize_desc():
    # The default needs no descriptor, whatever the level
    assert serialize.get_serialize_desc() is None
    assert serialize.get_serialize_desc(None) is None
    assert serialize.get_serialize_desc(lib.GxB_COMPRESSION_DEFAULT, 5) is None
    # A level is added to the methods that have levels, and ignored by the others
    get = serialize.get_serialize_desc
    ZSTD, LZ4HC = lib.GxB_COMPRESSION_ZSTD, lib.GxB_COMPRESSION_LZ4HC
    assert _compression(get(ZSTD)) == ZSTD
    assert _compression(get(ZSTD, 19)) == ZSTD + 19
    assert _compression(get(LZ4HC, 9)) == LZ4HC + 9
    assert _compression(get(lib.GxB_COMPRESSION_LZ4, 9)) == lib.GxB_COMPRESSION_LZ4
    assert _compression(get(lib.GxB_COMPRESSION_NONE, 9)) == lib.GxB_COMPRESSION_NONE


@pytest.mark.parametrize(
    "compression", [lib.GxB_COMPRESSION_DEFAULT, None], ids=["default", "None"]
)
def test_get_serialize_desc_nthreads_deprecated(compression):
    # The number of threads is a Context setting in GraphBLAS 10, not a descriptor setting.
    # The warning names the replacement and the caller, and GraphBLAS still rejects it.
    with pytest.warns(DeprecationWarning, match="serialize_matrix") as record:
        with pytest.raises(InvalidValue):
            serialize.get_serialize_desc(compression, nthreads=1)
    assert [w.filename for w in record] == [__file__]


_serialize_options = {
    "none": (lib.GxB_COMPRESSION_NONE, None),
    "default": (lib.GxB_COMPRESSION_DEFAULT, None),
    "lz4": (lib.GxB_COMPRESSION_LZ4, None),
    "lz4hc": (lib.GxB_COMPRESSION_LZ4HC, None),
    "lz4hc-9": (lib.GxB_COMPRESSION_LZ4HC, 9),
    "zstd": (lib.GxB_COMPRESSION_ZSTD, None),
    "zstd-1": (lib.GxB_COMPRESSION_ZSTD, 1),
    "zstd-19": (lib.GxB_COMPRESSION_ZSTD, 19),
    "unset": (None, None),
}


def _numbered_matrix(n):
    """An n-by-n INT64 matrix whose entries are 0, 1, ..., n*n - 1, by row."""
    A = matrix.matrix_new(lib.GrB_INT64, n, n)
    rows = np.repeat(np.arange(n, dtype=np.uint64), n)
    cols = np.tile(np.arange(n, dtype=np.uint64), n)
    values = np.arange(n * n, dtype=np.int64)
    matrix.matrix_build_int64(A, rows, cols, values, n * n, lib.GrB_PLUS_INT64)
    return A


@pytest.mark.parametrize("option", _serialize_options)
def test_serialize_matrix_compression(option):
    compression, level = _serialize_options[option]
    for kind in ("empty", "sparse", "full", "iso_full"):
        A = _make_matrix(lib.GrB_FP64, kind)
        data = serialize.serialize_matrix(A, compression, level)
        assert data.dtype == np.uint8
        _assert_matrices_equal(A, serialize.deserialize_matrix(data))
        # Anything with the buffer protocol will do
        _assert_matrices_equal(A, serialize.deserialize_matrix(data.tobytes()))


@pytest.mark.parametrize("option", _serialize_options)
def test_serialize_vector_compression(option):
    compression, level = _serialize_options[option]
    v = vector.vector_new(lib.GrB_FP64, 5)
    vector.set_fp64(v, 1.5, 1)
    vector.set_fp64(v, -2.5, 4)
    w = serialize.deserialize_vector(serialize.serialize_vector(v, compression, level))
    assert vector.vector_type(w) == lib.GrB_FP64
    assert vector.vector_size(w) == 5
    for x, y in zip(
        vector.vector_extract_tuples_fp64(v), vector.vector_extract_tuples_fp64(w), strict=True
    ):
        np.testing.assert_array_equal(x, y)


def test_serialize_compression_reaches_graphblas():
    # The compression and level change the result, so they get to GraphBLAS
    A = _numbered_matrix(100)
    size = {
        option: serialize.serialize_matrix(A, compression, level).nbytes
        for option, compression, level in [
            ("none", lib.GxB_COMPRESSION_NONE, None),
            ("lz4", lib.GxB_COMPRESSION_LZ4, None),
            ("lz4hc-1", lib.GxB_COMPRESSION_LZ4HC, 1),
            ("lz4hc-9", lib.GxB_COMPRESSION_LZ4HC, 9),
            ("zstd-1", lib.GxB_COMPRESSION_ZSTD, 1),
            ("zstd-19", lib.GxB_COMPRESSION_ZSTD, 19),
        ]
    }
    assert size["none"] > 100 * 100 * 8
    assert size["none"] > size["lz4"] > size["zstd-1"] > size["zstd-19"]
    assert size["lz4hc-1"] > size["lz4hc-9"]
    v = vector.vector_new(lib.GrB_INT64, 1000)
    for i in range(0, 1000, 3):
        vector.set_int64(v, i, i)
    none = serialize.serialize_vector(v, lib.GxB_COMPRESSION_NONE).nbytes
    assert none > serialize.serialize_vector(v, lib.GxB_COMPRESSION_ZSTD).nbytes


@pytest.mark.parametrize("free", [None, False], ids=["None", "False"])
def test_deserialize_without_free(free):
    # The caller frees the result
    A = _make_matrix(lib.GrB_INT64, "sparse")
    B = serialize.deserialize_matrix(serialize.serialize_matrix(A), free=free)
    _assert_matrices_equal(A, B)
    matrix.matrix_free(B)
    assert B[0] == NULL
    v = vector.vector_new(lib.GrB_INT64, 3)
    vector.set_int64(v, 7, 2)
    w = serialize.deserialize_vector(serialize.serialize_vector(v), free=free)
    assert vector.vector_nvals(w) == 1
    vector.vector_free(w)
    assert w[0] == NULL


def test_deserialize_with_free():
    # `free` is called with the result once it is released
    freed = []

    def free_matrix(A):
        freed.append(matrix.matrix_nvals(A))
        matrix.matrix_free(A)

    def free_vector(v):
        freed.append(vector.vector_nvals(v))
        vector.vector_free(v)

    A = _make_matrix(lib.GrB_INT64, "sparse")
    B = serialize.deserialize_matrix(serialize.serialize_matrix(A), free=free_matrix)
    _assert_matrices_equal(A, B)
    v = vector.vector_new(lib.GrB_INT64, 3)
    vector.set_int64(v, 7, 2)
    w = serialize.deserialize_vector(serialize.serialize_vector(v), free=free_vector)
    assert freed == []
    ffi.release(B)
    ffi.release(w)
    assert freed == [3, 1]


def test_deserialize_invalid():
    for deserialize in (serialize.deserialize_matrix, serialize.deserialize_vector):
        with pytest.raises(InvalidObject):
            deserialize(bytes(64))


def _context_engaged(probe):
    """Whether a GxB_Context is engaged on this thread.

    GraphBLAS can't say which one, but disengaging a Context that isn't engaged, `probe`,
    fails with GrB_INVALID_VALUE if another one is, and changes nothing.  Make `probe` before
    the Contexts it looks for: one freed while engaged (a bug) stays the engaged Context, and
    a probe made later could be given its address.
    """
    info = lib.GxB_Context_disengage(probe[0])
    assert info in {lib.GrB_SUCCESS, lib.GrB_INVALID_VALUE}
    return info == lib.GrB_INVALID_VALUE


@contextmanager
def _engaged(nthreads):
    """Engage a new Context that uses `nthreads` threads on this thread."""
    ctx = context.context_new()
    context.context_set_nthreads(ctx, nthreads)
    context.context_engage(ctx)
    try:
        yield
    finally:
        context.context_disengage(ctx)


def _assert_vectors_equal(v, w):
    assert vector.vector_type(w) == vector.vector_type(v)
    assert vector.vector_size(w) == vector.vector_size(v)
    for x, y in zip(
        vector.vector_extract_tuples_int64(v), vector.vector_extract_tuples_int64(w), strict=True
    ):
        np.testing.assert_array_equal(x, y)


def _big(kind):
    """A Matrix or Vector whose blob shows how many threads serialized it.

    Returns the object, the functions that serialize and deserialize it, and one that asserts
    that two of them are equal.  GraphBLAS compresses an array as a single block if it uses
    one thread, and as blocks of at least 64 KiB if it uses more (see GB_serialize_array.c).
    It uses one thread for each GxB_CHUNK bytes (64 KiB by default), up to the number of
    threads of the Context, and the 512 KiB of values here are enough for several.
    """
    if kind == "matrix":
        A = _numbered_matrix(256)
        return A, serialize.serialize_matrix, serialize.deserialize_matrix, _assert_matrices_equal
    n = 256 * 256
    v = vector.vector_new(lib.GrB_INT64, n)
    indices = np.arange(n, dtype=np.uint64)
    vector.vector_build_int64(v, indices, np.arange(n, dtype=np.int64), n, lib.GrB_PLUS_INT64)
    return v, serialize.serialize_vector, serialize.deserialize_vector, _assert_vectors_equal


@pytest.mark.parametrize("kind", ["matrix", "vector"])
def test_serialize_nthreads(kind):
    # GraphBLAS 10 rejects the number of threads in a descriptor: it's a Context setting.
    # Serializing with `nthreads` gives what serializing under a Context with that many
    # threads gives, and one thread gives a different result from two.
    probe = context.context_new()
    A, ser, deser, assert_equal = _big(kind)
    blobs = {}
    for nthreads in (1, 2):
        with _engaged(nthreads):
            expected = ser(A).tobytes()
        blobs[nthreads] = ser(A, nthreads=nthreads).tobytes()
        assert blobs[nthreads] == expected
    assert blobs[1] != blobs[2]
    assert ser(A, nthreads=np.int64(2)).tobytes() == blobs[2]  # any kind of integer will do
    # How many threads deserialize uses doesn't show in its result
    for blob in blobs.values():
        for nthreads in (1, 2):
            assert_equal(A, deser(blob, nthreads=nthreads))
    assert not _context_engaged(probe)


@pytest.mark.parametrize("nthreads", [None, 0, -1, -(2**40)])  # -2**40: beyond int32_t
@pytest.mark.parametrize("kind", ["matrix", "vector"])
def test_serialize_nthreads_default(kind, nthreads):
    # These use the Context engaged on this thread, if any, or else GxB_CONTEXT_WORLD
    probe = context.context_new()
    A, ser, deser, assert_equal = _big(kind)
    blob = ser(A).tobytes()
    assert ser(A, nthreads=nthreads).tobytes() == blob
    assert_equal(A, deser(blob, nthreads=nthreads))
    expected = {engaged: ser(A, nthreads=engaged).tobytes() for engaged in (1, 2)}
    for engaged in (1, 2):
        with _engaged(engaged):
            assert ser(A, nthreads=nthreads).tobytes() == expected[engaged]
            assert_equal(A, deser(blob, nthreads=nthreads))
            assert _context_engaged(probe)
    assert not _context_engaged(probe)


@pytest.mark.parametrize("kind", ["matrix", "vector"])
def test_serialize_nthreads_with_engaged_context(kind):
    # GraphBLAS can't say which Context is engaged on this thread, so a call that engaged
    # its own for `nthreads` couldn't engage that one again afterwards.  It refuses instead,
    # and the caller's Context stays engaged.
    probe = context.context_new()
    A, ser, deser, assert_equal = _big(kind)
    blob = ser(A)
    ctx = context.context_new()
    context.context_engage(ctx)
    try:
        for call in (lambda: ser(A, nthreads=2), lambda: deser(blob, nthreads=2)):
            with pytest.raises(InvalidValue, match="engaged on this thread"):
                call()
            assert _context_engaged(probe)
        # ...and it's `ctx`: disengaging it fails if another Context is engaged
        context.context_disengage(ctx)
        assert not _context_engaged(probe)
    finally:
        lib.GxB_Context_disengage(NULL)  # whichever is engaged, if the test failed


@pytest.mark.parametrize("kind", ["matrix", "vector"])
def test_serialize_nthreads_invalid(kind):
    probe = context.context_new()
    A, ser, deser, assert_equal = _big(kind)
    blob = ser(A)
    for call in (lambda n: ser(A, nthreads=n), lambda n: deser(blob, nthreads=n)):
        for nthreads in (2.0, 0.0, "2"):  # not even 0.0, which would mean the default
            with pytest.raises(TypeError):
                call(nthreads)
        for nthreads in (True, False, np.True_):  # nor a bool, which would mean 1 or 0
            with pytest.raises(TypeError, match="bool"):
                call(nthreads)
        with pytest.raises(OverflowError):
            call(2**31)  # more than int32_t holds
        assert not _context_engaged(probe)
    # Nor do failures.  This one is while the Context is engaged.
    other = vector.vector_new(lib.GrB_INT64) if kind == "matrix" else _numbered_matrix(2)
    with pytest.raises(TypeError):
        ser(other, nthreads=2)  # the wrong kind of object
    assert not _context_engaged(probe)
    with pytest.raises(InvalidObject):
        deser(bytes(64), nthreads=2)
    assert not _context_engaged(probe)


def test_call_with_nthreads_interrupted():
    # The call runs under the Context, which is disengaged whatever it raises
    probe = context.context_new()

    def interrupted():
        assert _context_engaged(probe)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        serialize._call_with_nthreads(2, interrupted)
    assert not _context_engaged(probe)


class _InterruptedLib:
    """`lib`, except that the `n`th lookup of GxB_Context_disengage raises KeyboardInterrupt."""

    def __init__(self, n):
        self.lookups = 0
        self.n = n

    def __getattr__(self, name):
        if name == "GxB_Context_disengage":
            self.lookups += 1
            if self.lookups == self.n:
                raise KeyboardInterrupt
        return getattr(lib, name)


@pytest.mark.thread_unsafe(reason="replaces `lib` in the serialize module")
@pytest.mark.parametrize("n", [2, 3], ids=["inner", "outer"])
def test_call_with_nthreads_interrupted_in_finally(monkeypatch, n):
    # A KeyboardInterrupt can come at the first bytecode of a `finally` block (PyPy can stop
    # it there), before the block disengages the Context.  Each of the two `finally` blocks
    # disengages it, so the other one still does.  The lookups of GxB_Context_disengage are
    # for the probe, then the inner `finally` block, then the outer one.
    probe = context.context_new()
    interrupted = _InterruptedLib(n)
    monkeypatch.setattr(serialize, "lib", interrupted)
    with pytest.raises(KeyboardInterrupt):
        serialize._call_with_nthreads(2, lambda: None)
    monkeypatch.undo()
    assert not _context_engaged(probe)
    assert interrupted.lookups == 3


@functools.cache
def _arena_like_0():
    """Arena 1, which allocates as arena 0 does.  It's made once: arenas can't be changed.

    Other tests that run in this process keep arena 5 uninitialized (test_can_unclaim_buffer)
    and use arena 6 (`recording_arena` in test_memory.py).
    """
    k = 1
    malloc, free = ffi.new("void**"), ffi.new("void**")
    info = lib.GrB_Global_get_VOID(lib.GrB_GLOBAL, malloc, lib.GxB_ARENA_MALLOC)
    assert info == lib.GrB_SUCCESS
    info = lib.GrB_Global_get_VOID(lib.GrB_GLOBAL, free, lib.GxB_ARENA_FREE)
    assert info == lib.GrB_SUCCESS
    malloc = ffi.cast("void *(*)(size_t)", malloc[0])
    free = ffi.cast("void (*)(void *)", free[0])
    assert lib.GxB_arena_init(k, malloc, NULL, NULL, free) == lib.GrB_SUCCESS
    return k


@pytest.mark.thread_unsafe(reason="changes the global header and data arenas")
@pytest.mark.parametrize("kind", ["matrix", "vector"])
def test_deserialize_nthreads_arenas(kind):
    # The result is in the global arenas with or without `nthreads`.  A new Context has arena
    # 0 for both, so the one that `nthreads` engages is given the global arenas.
    A, ser, deser, assert_equal = _big(kind)
    blob = ser(A)
    get = lib.GrB_Matrix_get_INT32 if kind == "matrix" else lib.GrB_Vector_get_INT32
    fields = (lib.GxB_ARENA_HEADER, lib.GxB_ARENA_DATA)
    k = _arena_like_0()
    old = [global_get_int32(field) for field in fields]
    try:
        for field in fields:
            assert lib.GrB_Global_set_INT32(lib.GrB_GLOBAL, k, field) == lib.GrB_SUCCESS
        for nthreads in (None, 2):
            B = deser(blob, nthreads=nthreads)
            arena = ffi.new("int32_t*")
            for field in fields:
                check_status(B, get(B[0], arena, field))
                assert arena[0] == k
            assert_equal(A, B)
    finally:
        for field, value in zip(fields, old, strict=True):
            assert lib.GrB_Global_set_INT32(lib.GrB_GLOBAL, value, field) == lib.GrB_SUCCESS
