"""User-defined operators of the functional API, made from cffi callbacks."""

import pytest

from suitesparse_graphblas import ffi, lib
from suitesparse_graphblas.api import binaryop, indexbinaryop, indexunaryop, monoid, unaryop

INT64 = lib.GrB_INT64

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


def _user_defined(kind):
    """Return a new user-defined object of `kind`, without a name."""
    if kind == "unaryop":
        return unaryop.unaryop_new(_square_plus_half, lib.GrB_FP64, lib.GrB_INT32)
    if kind == "binaryop":
        return binaryop.binaryop_new(_times10_plus, INT64, INT64, INT64)
    if kind == "indexunaryop":
        return indexunaryop.indexunaryop_new(_plus_position, INT64, INT64, INT64)
    if kind == "indexbinaryop":
        return indexbinaryop.indexbinaryop_new(
            _plus_positions, INT64, INT64, INT64, INT64, ffi.NULL, ffi.NULL
        )
    assert kind == "monoid"
    return monoid.monoid_new_int64(lib.GrB_PLUS_INT64, 0)


def _named(kind, name, defn):
    """Return a new operator of `kind` with a JIT name and definition."""
    if kind == "unaryop":
        return unaryop.unaryop_new_named(_square_plus_half, lib.GrB_FP64, lib.GrB_INT32, name, defn)
    if kind == "binaryop":
        return binaryop.binaryop_new_named(_times10_plus, INT64, INT64, INT64, name, defn)
    if kind == "indexunaryop":
        return indexunaryop.indexunaryop_new_named(_plus_position, INT64, INT64, INT64, name, defn)
    assert kind == "indexbinaryop"
    return indexbinaryop.indexbinaryop_new(_plus_positions, INT64, INT64, INT64, INT64, name, defn)


@pytest.mark.parametrize("kind", list(MODULES))
def test_get_string_long_name(kind):
    # GraphBLAS doesn't limit the length of a name the user sets
    get_size, get_string, set_string = (
        getattr(MODULES[kind], f"{kind}_{func}")
        for func in ("get_size", "get_string", "set_string")
    )
    obj = _user_defined(kind)
    name = "long_name_" + "x" * 500
    set_string(obj, lib.GrB_NAME, name)
    assert get_size(obj, lib.GrB_NAME) == len(name) + 1
    assert get_string(obj, lib.GrB_NAME) == name


@pytest.mark.parametrize("kind", list(DEFINITIONS))
def test_get_string_long_definition(kind):
    # A definition is as long as its C code
    get_size, get_string = (
        getattr(MODULES[kind], f"{kind}_{func}") for func in ("get_size", "get_string")
    )
    name, defn = DEFINITIONS[kind]
    defn = f"/* {'x' * 500} */\n{defn}"
    op = _named(kind, name, defn)
    assert get_string(op, lib.GxB_JIT_C_NAME) == name
    assert get_size(op, lib.GxB_JIT_C_DEFINITION) == len(defn) + 1
    assert get_string(op, lib.GxB_JIT_C_DEFINITION) == defn
