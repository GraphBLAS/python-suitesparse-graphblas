"""Monoids of the functional API: the typed constructors, and monoids of user-defined types."""

import math

import pytest

from suitesparse_graphblas import check_status, exceptions, ffi, lib, libget, supports_complex
from suitesparse_graphblas.api import binaryop, grb_type, monoid, vector

# Whether a constructor hands its monoid to the garbage collector (its default) or leaves
# freeing it to the caller (`free=None`)
GC = [pytest.param(True, id="gc"), pytest.param(False, id="manual")]

needs_complex = pytest.mark.skipif(not supports_complex(), reason="no complex types")

# suffix, C type, identity, terminal.  GraphBLAS ignores the identity and terminal given with
# the operator of a known monoid, such as MAX, so these monoids use FIRST, which isn't one.
TYPES = [
    ("bool", "bool", True, False),
    ("int8", "int8_t", -(2**7), 2**7 - 1),
    ("int16", "int16_t", -(2**15), 2**15 - 1),
    ("int32", "int32_t", -(2**31), 2**31 - 1),
    ("int64", "int64_t", -(2**63), 2**63 - 1),
    ("uint8", "uint8_t", 2**8 - 1, 1),
    ("uint16", "uint16_t", 2**16 - 1, 1),
    ("uint32", "uint32_t", 2**32 - 1, 1),
    ("uint64", "uint64_t", 2**64 - 1, 1),
    ("fp32", "float", -math.inf, 2.25),
    ("fp64", "double", -math.inf, 1e300),
    pytest.param("fc32", "GxB_FC32_t", 1.5 - 2j, -3 + 0.5j, marks=needs_complex),
    pytest.param("fc64", "GxB_FC64_t", 1.5 - 2j, -3 + 1e300j, marks=needs_complex),
]

BIG = 64  # int64s in a value of a large user-defined type


@ffi.callback("GxB_binary_function")
def _max_big(z, x, y):
    # Elementwise max
    x = ffi.cast("int64_t*", x)
    y = ffi.cast("int64_t*", y)
    z = ffi.cast("int64_t*", z)
    for k in range(BIG):
        z[k] = max(x[k], y[k])


@pytest.mark.parametrize("gc", GC)
@pytest.mark.parametrize("with_terminal", [False, True], ids=["new", "terminal_new"])
@pytest.mark.parametrize(("suffix", "ctype", "identity", "terminal"), TYPES)
def test_monoid_new(gc, with_terminal, suffix, ctype, identity, terminal):
    op = libget(f"GrB_FIRST_{suffix.upper()}")
    kwargs = {} if gc else {"free": None}
    if with_terminal:
        new = getattr(monoid, f"monoid_terminal_new_{suffix}")
        m = new(op, identity, terminal, **kwargs)
    else:
        m = getattr(monoid, f"monoid_new_{suffix}")(op, identity, **kwargs)
    assert monoid.monoid_operator(m) == op
    code = libget(f"GrB_{suffix.upper()}_CODE")
    assert monoid.monoid_get_int32(m, lib.GrB_OUTP_TYPE_CODE) == code
    value = monoid.monoid_identity(m)
    assert ffi.cast(f"{ctype}*", value)[0] == identity
    has_terminal, value = monoid.monoid_terminal(m)
    assert has_terminal == with_terminal
    if with_terminal:
        assert ffi.cast(f"{ctype}*", value)[0] == terminal

    # An empty vector reduces to the identity
    u = vector.vector_new(libget(f"GrB_{suffix.upper()}"), 2)
    result = ffi.new(f"{ctype}*")
    reduce = libget(f"GrB_Vector_reduce_{suffix.upper()}")
    check_status(u, reduce(result, ffi.NULL, m[0], u[0], ffi.NULL))
    assert result[0] == identity
    if not gc:
        monoid.monoid_free(m)
        assert m[0] == ffi.NULL


@pytest.mark.parametrize("gc", GC)
@pytest.mark.parametrize("with_terminal", [False, True], ids=["new", "terminal_new"])
def test_monoid_of_large_type(gc, with_terminal):
    size = 8 * BIG
    T = grb_type.grb_type_new(size)
    op = binaryop.binaryop_new(_max_big, T[0], T[0], T[0])
    identity = ffi.new("int64_t[]", [-(2**63)] * BIG)
    terminal = ffi.new("int64_t[]", [2**63 - 1] * BIG)
    kwargs = {} if gc else {"free": None}
    if with_terminal:
        m = monoid.monoid_terminal_new_udt(op[0], identity, terminal, **kwargs)
    else:
        m = monoid.monoid_new_udt(op[0], identity, **kwargs)
    # GraphBLAS copies one value of the monoid's type, whatever its size
    value = monoid.monoid_identity(m)
    assert len(ffi.buffer(value)) >= size
    assert ffi.buffer(value)[:size] == ffi.buffer(identity)[:]
    has_terminal, value = monoid.monoid_terminal(m)
    assert has_terminal == with_terminal
    if with_terminal:
        assert len(ffi.buffer(value)) >= size
        assert ffi.buffer(value)[:size] == ffi.buffer(terminal)[:]

    u = vector.vector_new(T[0], 3)
    first = ffi.new("int64_t[]", list(range(BIG)))
    last = ffi.new("int64_t[]", list(range(BIG, 0, -1)))
    check_status(u, lib.GrB_Vector_setElement_UDT(u[0], first, 0))
    check_status(u, lib.GrB_Vector_setElement_UDT(u[0], last, 2))
    result = ffi.new("int64_t[]", BIG)
    check_status(u, lib.GrB_Vector_reduce_UDT(result, ffi.NULL, m[0], u[0], ffi.NULL))
    assert list(result) == [max(k, BIG - k) for k in range(BIG)]
    if not gc:
        monoid.monoid_free(m)
        assert m[0] == ffi.NULL


def test_monoid_new_fails():
    # The operator's inputs and output must have the same type
    with pytest.raises(exceptions.DomainMismatch):
        monoid.monoid_new_int64(lib.GrB_LT_INT64, 0)
