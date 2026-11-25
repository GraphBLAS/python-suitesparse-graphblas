"""User-defined operators of the functional API, made from cffi callbacks."""

import pytest

from suitesparse_graphblas import exceptions, ffi, lib
from suitesparse_graphblas.api import (
    binaryop,
    indexbinaryop,
    indexunaryop,
    matrix,
    monoid,
    scalar,
    selectop,
    unaryop,
    vector,
)
from suitesparse_graphblas.api.utils import _capture_c_output

INT64 = lib.GrB_INT64

# Whether a constructor hands its object to the garbage collector (its default) or leaves
# freeing it to the caller (`free=None`)
GC = [pytest.param(True, id="gc"), pytest.param(False, id="manual")]

# GraphBLAS calls these through the function pointers cffi makes for them.  They live at
# module level so that they outlive every operator built from them.


@ffi.callback("GxB_unary_function")
def _square_plus_half(z, x):
    # FP64 <- INT32
    value = ffi.cast("int32_t*", x)[0]
    ffi.cast("double*", z)[0] = value * value + 0.5


@ffi.callback("GxB_binary_function")
def _times10_plus(z, x, y):
    # INT64 <- INT64, INT64
    ffi.cast("int64_t*", z)[0] = 10 * ffi.cast("int64_t*", x)[0] + ffi.cast("int64_t*", y)[0]


@ffi.callback("GxB_index_unary_function")
def _plus_position(z, x, i, j, y):
    # INT64 <- INT64 at (i, j), INT64 thunk
    x = ffi.cast("int64_t*", x)[0]
    y = ffi.cast("int64_t*", y)[0]
    ffi.cast("int64_t*", z)[0] = x + 100 * i + 10 * j + y


@ffi.callback("GxB_index_binary_function")
def _plus_positions(z, x, ix, jx, y, iy, jy, theta):
    # INT64 <- INT64 at (ix, jx), INT64 at (iy, jy), INT64 theta
    x = ffi.cast("int64_t*", x)[0]
    y = ffi.cast("int64_t*", y)[0]
    theta = ffi.cast("int64_t*", theta)[0]
    ffi.cast("int64_t*", z)[0] = x + y + theta * (1000 * ix + 100 * jx + 10 * iy + jy)


# C definitions of the callbacks, as the JIT would want them
DEFINITIONS = {
    "unaryop": (
        "square_plus_half",
        "void square_plus_half (double *z, const int32_t *x) "
        "{ (*z) = ((double) (*x)) * (*x) + 0.5 ; }",
    ),
    "binaryop": (
        "times10_plus",
        "void times10_plus (int64_t *z, const int64_t *x, const int64_t *y) "
        "{ (*z) = 10 * (*x) + (*y) ; }",
    ),
    "indexunaryop": (
        "plus_position",
        "void plus_position (int64_t *z, const int64_t *x, GrB_Index i, GrB_Index j, "
        "const int64_t *y) { (*z) = (*x) + 100 * i + 10 * j + (*y) ; }",
    ),
    "indexbinaryop": (
        "plus_positions",
        "void plus_positions (int64_t *z, const int64_t *x, GrB_Index ix, GrB_Index jx, "
        "const int64_t *y, GrB_Index iy, GrB_Index jy, const int64_t *theta) "
        "{ (*z) = (*x) + (*y) + (*theta) * (1000 * ix + 100 * jx + 10 * iy + jy) ; }",
    ),
}

MODULES = {
    "unaryop": unaryop,
    "binaryop": binaryop,
    "indexunaryop": indexunaryop,
    "indexbinaryop": indexbinaryop,
    "monoid": monoid,
}


def _user_defined(kind, *, gc=True):
    """Return a new user-defined object of `kind`, without a name."""
    kwargs = {} if gc else {"free": None}
    if kind == "unaryop":
        return unaryop.unaryop_new(_square_plus_half, lib.GrB_FP64, lib.GrB_INT32, **kwargs)
    if kind == "binaryop":
        return binaryop.binaryop_new(_times10_plus, INT64, INT64, INT64, **kwargs)
    if kind == "indexunaryop":
        return indexunaryop.indexunaryop_new(_plus_position, INT64, INT64, INT64, **kwargs)
    if kind == "indexbinaryop":
        return indexbinaryop.indexbinaryop_new(
            _plus_positions, INT64, INT64, INT64, INT64, ffi.NULL, ffi.NULL, **kwargs
        )
    assert kind == "monoid"
    return monoid.monoid_new_int64(lib.GrB_PLUS_INT64, 0, **kwargs)


def _named(kind, name, defn, *, gc=True):
    """Return a new operator of `kind` with a JIT name and definition."""
    kwargs = {} if gc else {"free": None}
    if kind == "unaryop":
        return unaryop.unaryop_new_named(
            _square_plus_half, lib.GrB_FP64, lib.GrB_INT32, name, defn, **kwargs
        )
    if kind == "binaryop":
        return binaryop.binaryop_new_named(_times10_plus, INT64, INT64, INT64, name, defn, **kwargs)
    if kind == "indexunaryop":
        return indexunaryop.indexunaryop_new_named(
            _plus_position, INT64, INT64, INT64, name, defn, **kwargs
        )
    assert kind == "indexbinaryop"
    return indexbinaryop.indexbinaryop_new(
        _plus_positions, INT64, INT64, INT64, INT64, name, defn, **kwargs
    )


def _func(kind, name):
    """Return the function `<kind>_<name>` of the module for `kind`."""
    return getattr(MODULES[kind], f"{kind}_{name}")


def _free(kind, obj, gc):
    """Free `obj` if it isn't garbage collected, and check that it's gone."""
    if not gc:
        _func(kind, "free")(obj)
        assert obj[0] == ffi.NULL


def _entries(A):
    """Return the entries of an INT64 matrix as sorted (row, col, value) tuples."""
    rows, cols, values = matrix.matrix_extract_tuples_int64(A)
    return sorted(zip(rows.tolist(), cols.tolist(), values.tolist()))


@pytest.mark.parametrize("kind", list(MODULES))
def test_get_string_long_name(kind):
    # GraphBLAS doesn't limit the length of a name the user sets
    obj = _user_defined(kind)
    name = "long_name_" + "x" * 500
    _func(kind, "set_string")(obj, lib.GrB_NAME, name)
    assert _func(kind, "get_size")(obj, lib.GrB_NAME) == len(name) + 1
    assert _func(kind, "get_string")(obj, lib.GrB_NAME) == name


@pytest.mark.parametrize("kind", list(DEFINITIONS))
def test_get_string_long_definition(kind):
    # A definition is as long as its C code
    name, defn = DEFINITIONS[kind]
    defn = f"/* {'x' * 500} */\n{defn}"
    op = _named(kind, name, defn)
    assert _func(kind, "get_string")(op, lib.GxB_JIT_C_NAME) == name
    assert _func(kind, "get_size")(op, lib.GxB_JIT_C_DEFINITION) == len(defn) + 1
    assert _func(kind, "get_string")(op, lib.GxB_JIT_C_DEFINITION) == defn


@pytest.mark.parametrize("gc", GC)
def test_unaryop_new(gc):
    op = _user_defined("unaryop", gc=gc)
    assert unaryop.unaryop_xtype(op) == lib.GrB_INT32
    assert unaryop.unaryop_ztype(op) == lib.GrB_FP64
    assert unaryop.unaryop_xtype_name(op) == "int32_t"
    assert unaryop.unaryop_ztype_name(op) == "double"
    assert unaryop.unaryop_get_int32(op, lib.GrB_INP0_TYPE_CODE) == lib.GrB_INT32_CODE
    assert unaryop.unaryop_get_int32(op, lib.GrB_OUTP_TYPE_CODE) == lib.GrB_FP64_CODE
    assert unaryop.unaryop_get_string(op, lib.GrB_NAME) == ""
    unaryop.unaryop_wait(op)

    u = vector.vector_new(lib.GrB_INT32, 4)
    vector.set_int32(u, 3, 1)
    vector.set_int32(u, -2, 3)
    w = vector.vector_new(lib.GrB_FP64, 4)
    vector.vector_apply(w, op[0], u)
    indices, values = vector.vector_extract_tuples_fp64(w)
    assert indices.tolist() == [1, 3]
    assert values.tolist() == [9.5, 4.5]
    _free("unaryop", op, gc)


@pytest.mark.parametrize("gc", GC)
def test_binaryop_new(gc):
    op = _user_defined("binaryop", gc=gc)
    assert binaryop.binaryop_xtype(op) == INT64
    assert binaryop.binaryop_ytype(op) == INT64
    assert binaryop.binaryop_ztype(op) == INT64
    assert binaryop.binaryop_xtype_name(op) == "int64_t"
    assert binaryop.binaryop_ytype_name(op) == "int64_t"
    assert binaryop.binaryop_ztype_name(op) == "int64_t"
    for field in (lib.GrB_INP0_TYPE_CODE, lib.GrB_INP1_TYPE_CODE, lib.GrB_OUTP_TYPE_CODE):
        assert binaryop.binaryop_get_int32(op, field) == lib.GrB_INT64_CODE
    binaryop.binaryop_wait(op)

    A = matrix.matrix_new(INT64, 2, 2)
    B = matrix.matrix_new(INT64, 2, 2)
    matrix.set_int64(A, 3, 0, 0)
    matrix.set_int64(B, 4, 0, 0)
    matrix.set_int64(B, 7, 0, 1)
    matrix.set_int64(A, 5, 1, 1)
    C = matrix.matrix_new(INT64, 2, 2)
    # The operator applies only where both have an entry
    matrix.matrix_ewise_add(C, op[0], A, B)
    assert _entries(C) == [(0, 0, 34), (0, 1, 7), (1, 1, 5)]
    _free("binaryop", op, gc)


@pytest.mark.parametrize("gc", GC)
def test_binaryop_new_indexop(gc):
    idxop = _user_defined("indexbinaryop")
    theta = scalar.scalar_new(INT64)
    scalar.set_int64(theta, 2)
    kwargs = {} if gc else {"free": None}
    op = binaryop.binaryop_new_indexop(idxop[0], theta[0], **kwargs)
    assert binaryop.binaryop_ztype(op) == INT64

    A = matrix.matrix_new(INT64, 3, 3)
    B = matrix.matrix_new(INT64, 3, 3)
    matrix.set_int64(A, 1, 1, 2)
    matrix.set_int64(B, 2, 1, 2)
    matrix.set_int64(A, 3, 2, 0)
    matrix.set_int64(B, 4, 2, 0)
    matrix.set_int64(A, 5, 0, 0)
    C = matrix.matrix_new(INT64, 3, 3)
    # x + y + theta * (1000 * ix + 100 * jx + 10 * iy + jy), where both have an entry
    matrix.matrix_ewise_mult(C, op[0], A, B)
    assert _entries(C) == [(1, 2, 1 + 2 + 2 * 1212), (2, 0, 3 + 4 + 2 * 2020)]
    _free("binaryop", op, gc)


@pytest.mark.parametrize("gc", GC)
def test_indexunaryop_new(gc):
    op = _user_defined("indexunaryop", gc=gc)
    assert indexunaryop.indexunaryop_xtype_name(op) == "int64_t"
    assert indexunaryop.indexunaryop_ytype_name(op) == "int64_t"
    assert indexunaryop.indexunaryop_ztype_name(op) == "int64_t"
    for field in (lib.GrB_INP0_TYPE_CODE, lib.GrB_INP1_TYPE_CODE, lib.GrB_OUTP_TYPE_CODE):
        assert indexunaryop.indexunaryop_get_int32(op, field) == lib.GrB_INT64_CODE
    indexunaryop.indexunaryop_wait(op)

    A = matrix.matrix_new(INT64, 3, 3)
    matrix.set_int64(A, 1, 1, 2)
    matrix.set_int64(A, 5, 2, 0)
    thunk = scalar.scalar_new(INT64)
    scalar.set_int64(thunk, 7)
    C = matrix.matrix_new(INT64, 3, 3)
    # x + 100 * i + 10 * j + thunk
    matrix.matrix_apply_indexop(C, op[0], A, thunk)
    assert _entries(C) == [(1, 2, 128), (2, 0, 212)]
    _free("indexunaryop", op, gc)


@pytest.mark.parametrize("gc", GC)
def test_indexbinaryop_new(gc):
    name, defn = DEFINITIONS["indexbinaryop"]
    op = _named("indexbinaryop", name, defn, gc=gc)
    for field in (
        lib.GrB_INP0_TYPE_CODE,
        lib.GrB_INP1_TYPE_CODE,
        lib.GrB_OUTP_TYPE_CODE,
        lib.GxB_THETA_TYPE_CODE,
    ):
        assert indexbinaryop.indexbinaryop_get_int32(op, field) == lib.GrB_INT64_CODE
    for field in (
        lib.GrB_INP0_TYPE_STRING,
        lib.GrB_INP1_TYPE_STRING,
        lib.GrB_OUTP_TYPE_STRING,
        lib.GxB_THETA_TYPE_STRING,
    ):
        assert indexbinaryop.indexbinaryop_get_string(op, field) == "GrB_INT64"
    indexbinaryop.indexbinaryop_wait(op)
    _free("indexbinaryop", op, gc)


@pytest.mark.parametrize("kind", list(DEFINITIONS))
@pytest.mark.parametrize("gc", GC)
@pytest.mark.parametrize("encode", [False, True], ids=["str", "bytes"])
def test_new_named(kind, gc, encode):
    # Only the strings: using an operator that has a definition could start the JIT
    name, defn = DEFINITIONS[kind]
    if encode:
        op = _named(kind, name.encode(), defn.encode(), gc=gc)
    else:
        op = _named(kind, name, defn, gc=gc)
    get_string = _func(kind, "get_string")
    assert get_string(op, lib.GxB_JIT_C_NAME) == name
    assert get_string(op, lib.GxB_JIT_C_DEFINITION) == defn
    assert get_string(op, lib.GrB_NAME) == ""
    _free(kind, op, gc)


@pytest.mark.parametrize("kind", list(MODULES))
def test_set_int32_not_allowed(kind):
    # The types of an operator or a monoid are fixed when it's made
    get_int32 = _func(kind, "get_int32")
    obj = _user_defined(kind)
    code = get_int32(obj, lib.GrB_OUTP_TYPE_CODE)
    with pytest.raises(exceptions.InvalidValue):
        _func(kind, "set_int32")(obj, lib.GrB_OUTP_TYPE_CODE, lib.GrB_BOOL_CODE)
    assert get_int32(obj, lib.GrB_OUTP_TYPE_CODE) == code


@pytest.mark.parametrize(
    ("kind", "name"),
    [
        ("unaryop", "GrB_AINV_INT64"),
        ("binaryop", "GrB_PLUS_INT64"),
        ("indexunaryop", "GrB_VALUEEQ_INT64"),
        ("monoid", "GrB_PLUS_MONOID_INT64"),
    ],
)
def test_set_string_of_builtin(kind, name):
    # A built-in keeps its name
    ctype = {
        "unaryop": "GrB_UnaryOp*",
        "binaryop": "GrB_BinaryOp*",
        "indexunaryop": "GrB_IndexUnaryOp*",
        "monoid": "GrB_Monoid*",
    }[kind]
    obj = ffi.new(ctype, getattr(lib, name))
    with pytest.raises(exceptions.InvalidValue):
        _func(kind, "set_string")(obj, lib.GrB_NAME, "renamed")
    assert _func(kind, "get_string")(obj, lib.GrB_NAME) == name


@pytest.mark.parametrize(
    "new",
    [
        lambda: unaryop.unaryop_new(_square_plus_half, ffi.NULL, lib.GrB_INT32),
        lambda: binaryop.binaryop_new(_times10_plus, ffi.NULL, INT64, INT64),
        lambda: indexunaryop.indexunaryop_new(_plus_position, ffi.NULL, INT64, INT64),
        lambda: indexbinaryop.indexbinaryop_new(
            _plus_positions, ffi.NULL, INT64, INT64, INT64, ffi.NULL, ffi.NULL
        ),
        lambda: binaryop.binaryop_new_indexop(ffi.NULL, ffi.NULL),
    ],
    ids=["unaryop", "binaryop", "indexunaryop", "indexbinaryop", "binaryop_indexop"],
)
def test_new_fails(new):
    with pytest.raises(exceptions.NullPointer):
        new()


@pytest.mark.thread_unsafe(reason="redirects the process's stdout file descriptor")
@pytest.mark.parametrize("kind", list(DEFINITIONS))
def test_print(kind):
    name, defn = DEFINITIONS[kind]
    op = _named(kind, name, defn)
    out = _capture_c_output(_func(kind, "print"), op, "my_op", lib.GxB_COMPLETE)
    assert "my_op (user-defined)" in out
    assert f"z={name}(" in out
    assert defn in out

    def fprint_silent_then_short():
        # A capture that gets no output at all fails on Emscripten, where reading the empty
        # pipe raises BlockingIOError
        _func(kind, "fprint")(op, ffi.NULL, "silent_op", lib.GxB_SILENT)
        _func(kind, "fprint")(op, ffi.NULL, b"short_op", lib.GxB_SHORT)

    out = _capture_c_output(fprint_silent_then_short)
    assert "silent_op" not in out
    assert "short_op (user-defined)" in out


@pytest.mark.parametrize(
    "func",
    [
        selectop.selectop_xtype,
        selectop.selectop_ttype,
        selectop.selectop_print,
        lambda op: selectop.selectop_fprint(op, ffi.NULL),
    ],
    ids=["xtype", "ttype", "print", "fprint"],
)
def test_selectop_null(func):
    # These check the status themselves, rather than with check_status
    with pytest.raises(exceptions.NullPointer, match="GxB_SelectOp_.* failed with info=-2"):
        func(ffi.new("GxB_SelectOp*"))
