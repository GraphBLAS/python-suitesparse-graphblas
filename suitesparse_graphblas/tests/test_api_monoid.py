"""Monoids of the functional API: the typed constructors, and monoids of user-defined types."""

from suitesparse_graphblas import check_status, ffi, lib
from suitesparse_graphblas.api import binaryop, grb_type, monoid, vector

BIG = 64  # int64s in a value of a large user-defined type


@ffi.callback("GxB_binary_function")
def _max_big(z, x, y):
    # Elementwise max
    x = ffi.cast("int64_t*", x)
    y = ffi.cast("int64_t*", y)
    z = ffi.cast("int64_t*", z)
    for k in range(BIG):
        z[k] = max(x[k], y[k])


def test_monoid_of_large_type():
    size = 8 * BIG
    T = grb_type.grb_type_new(size)
    op = binaryop.binaryop_new(_max_big, T[0], T[0], T[0])
    identity = ffi.new("int64_t[]", [-(2**63)] * BIG)
    terminal = ffi.new("int64_t[]", [2**63 - 1] * BIG)
    m = monoid.monoid_terminal_new_udt(op[0], identity, terminal)
    # GraphBLAS copies one value of the monoid's type, whatever its size
    value = monoid.monoid_identity(m)
    assert len(ffi.buffer(value)) >= size
    assert ffi.buffer(value)[:size] == ffi.buffer(identity)[:]
    has_terminal, value = monoid.monoid_terminal(m)
    assert has_terminal
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
