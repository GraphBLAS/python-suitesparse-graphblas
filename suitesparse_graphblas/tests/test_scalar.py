import pytest

from suitesparse_graphblas import ffi, lib, scalar, supports_complex


@pytest.mark.skipif("not supports_complex()")
def test_complex():
    s = ffi.new("GrB_Scalar*")
    success = lib.GrB_SUCCESS
    assert lib.GrB_Scalar_new(s, lib.GxB_FC64) == success
    assert lib.GxB_Scalar_setElement_FC64(s[0], 1j) == success
    assert lib.GrB_Scalar_free(s) == success


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


def _values(name):
    """Values for a scalar of type `name`, including its limits, that it holds exactly."""
    if name == "bool":
        return [True, False]
    if name.startswith("int"):
        bits = int(name[3:])
        return [-(2 ** (bits - 1)), -1, 2 ** (bits - 1) - 1]
    if name.startswith("uint"):
        bits = int(name[4:])
        return [0, 1, 2**bits - 1]
    if name.startswith("fp"):
        return [-0.5, 2.0**100, 2.0**-20]
    return [complex(-0.5, 2.0), complex(2.0**100, -1.0), 2.0j]


@pytest.mark.parametrize("T", sorted(_type_names, key=_type_names.get), ids=_type_names.get)
def test_set_get(T):
    name = _type_names[T]
    set_value = getattr(scalar, f"set_{name}")
    get_value = getattr(scalar, f"get_{name}")
    s = scalar.scalar_new(T)
    assert scalar.scalar_type(s) == T
    assert get_value(s) is None  # no value yet
    for value in _values(name):
        set_value(s, value)
        got = get_value(s)
        assert got == value
        assert type(got) is type(value)

    t = scalar.scalar_dup(s)
    assert scalar.scalar_type(t) == T
    scalar.scalar_clear(s)
    assert get_value(s) is None
    assert get_value(t) == value  # the duplicate has its own value


def test_free():
    # With free=None, the caller frees the scalar
    s = scalar.scalar_new(lib.GrB_INT64, free=None)
    scalar.set_int64(s, 3)
    t = scalar.scalar_dup(s, free=None)
    assert scalar.get_int64(t) == 3
    scalar.scalar_free(s)
    scalar.scalar_free(t)
    assert s[0] == ffi.NULL
    assert t[0] == ffi.NULL

    # Otherwise `free` is called with it once it is released
    freed = []

    def free(s):
        freed.append(scalar.get_int64(s))
        scalar.scalar_free(s)

    s = scalar.scalar_new(lib.GrB_INT64, free=free)
    scalar.set_int64(s, 4)
    t = scalar.scalar_dup(s, free=free)
    scalar.set_int64(t, 5)
    assert freed == []
    ffi.release(s)
    ffi.release(t)
    assert freed == [4, 5]


def test_get_casts():
    # A getter of another type casts the value, as C does
    s = scalar.scalar_new(lib.GrB_INT32)
    scalar.set_int32(s, -7)
    assert scalar.get_fp64(s) == -7.0
    assert scalar.get_int8(s) == -7
    assert scalar.get_bool(s) is True
    scalar.set_fp64(s, 2.75)
    assert scalar.get_int32(s) == 2
