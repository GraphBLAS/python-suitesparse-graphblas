import pytest

from suitesparse_graphblas import exceptions, ffi, lib
from suitesparse_graphblas.api import (
    container,
    context,
    descriptor,
    global_options,
    grb_type,
    matrix,
    semiring,
    vector,
)


def test_grb_type_definition_longer_than_256():
    fields = "; ".join(f"double x{i}" for i in range(40))
    defn = f"typedef struct {{ {fields}; }} wide_type ;"
    assert len(defn) > 256
    T = grb_type.grb_type_new_named(40 * 8, "wide_type", defn)
    assert grb_type.grb_type_get_string(T, lib.GxB_JIT_C_DEFINITION) == defn
    assert grb_type.grb_type_get_size(T, lib.GxB_JIT_C_DEFINITION) == len(defn) + 1
    assert grb_type.grb_type_get_string(T, lib.GxB_JIT_C_NAME) == "wide_type"


# `free=None` returns a handle that isn't freed when it is garbage collected, so the caller
# frees it.  Freeing sets the handle to NULL.


def test_context_free_none():
    ctx = context.context_new(free=None)
    assert ctx[0] != ffi.NULL
    context.context_set_nthreads(ctx, 3)
    assert context.context_get_nthreads(ctx) == 3
    context.context_free(ctx)
    assert ctx[0] == ffi.NULL
    # A freed context is an error to use, not a crash
    with pytest.raises(exceptions.NullPointer):
        context.context_get_nthreads(ctx)


def test_container_free_none():
    c = container.container_new(free=None)
    assert c[0] != ffi.NULL
    container.container_free(c)
    assert c[0] == ffi.NULL


def test_grb_type_free_none():
    T = grb_type.grb_type_new(16, free=None)
    assert grb_type.grb_type_size(T) == 16
    grb_type.grb_type_free(T)
    assert T[0] == ffi.NULL

    defn = "typedef struct { double a ; double b ; } pair_type ;"
    T = grb_type.grb_type_new_named(16, "pair_type", defn, free=None)
    assert grb_type.grb_type_size(T) == 16
    assert grb_type.grb_type_name(T) == "pair_type"
    assert grb_type.grb_type_get_string(T, lib.GxB_JIT_C_DEFINITION) == defn
    grb_type.grb_type_free(T)
    assert T[0] == ffi.NULL


def test_semiring_free_none():
    s = semiring.semiring_new(lib.GrB_MAX_MONOID_FP64, lib.GrB_PLUS_FP64, free=None)
    assert semiring.semiring_add(s) == lib.GrB_MAX_MONOID_FP64
    assert semiring.semiring_multiply(s) == lib.GrB_PLUS_FP64
    semiring.semiring_free(s)
    assert s[0] == ffi.NULL


def test_context_invalid_field():
    ctx = context.context_new()
    with pytest.raises(exceptions.InvalidValue):
        context.context_set_int32(ctx, 999999, 1)
    with pytest.raises(exceptions.InvalidValue):
        context.context_get_fp64(ctx, 999999)


def test_descriptor_set_get():
    # Not a fixture: pytest-run-parallel would share it between threads
    D = ffi.new("GrB_Descriptor*")
    assert lib.GrB_Descriptor_new(D) == lib.GrB_SUCCESS
    D = ffi.gc(D, lib.GrB_Descriptor_free)
    desc = D[0]
    assert descriptor.descriptor_get_int32(desc, lib.GrB_OUTP_FIELD) == lib.GrB_DEFAULT
    descriptor.descriptor_set_int32(desc, lib.GrB_OUTP_FIELD, lib.GrB_REPLACE)
    assert descriptor.descriptor_get_int32(desc, lib.GrB_OUTP_FIELD) == lib.GrB_REPLACE
    descriptor.descriptor_set_int32(desc, lib.GrB_MASK_FIELD, lib.GrB_STRUCTURE)
    assert descriptor.descriptor_get_int32(desc, lib.GrB_MASK_FIELD) == lib.GrB_STRUCTURE

    descriptor.descriptor_set_string(desc, lib.GrB_NAME, "my_descriptor")
    assert descriptor.descriptor_get_string(desc, lib.GrB_NAME) == "my_descriptor"
    assert descriptor.descriptor_get_size(desc, lib.GrB_NAME) == len("my_descriptor") + 1
    descriptor.descriptor_wait(desc)


def test_builtin_descriptor_is_read_only():
    with pytest.raises(exceptions.InvalidValue):
        descriptor.descriptor_set_int32(lib.GrB_DESC_S, lib.GrB_OUTP_FIELD, lib.GrB_REPLACE)
    with pytest.raises(exceptions.InvalidValue):
        descriptor.descriptor_set_string(lib.GrB_DESC_S, lib.GrB_NAME, "renamed")
    assert descriptor.descriptor_get_int32(lib.GrB_DESC_S, lib.GrB_OUTP_FIELD) == lib.GrB_DEFAULT
    assert descriptor.descriptor_get_string(lib.GrB_DESC_S, lib.GrB_NAME) == "GrB_DESC_S"


def test_grb_type_name():
    T = grb_type.grb_type_new(8)
    # A new type has no name until it is given one
    assert grb_type.grb_type_get_string(T, lib.GrB_NAME) == ""
    grb_type.grb_type_set_string(T, lib.GrB_NAME, "my_type")
    assert grb_type.grb_type_get_string(T, lib.GrB_NAME) == "my_type"
    assert grb_type.grb_type_get_size(T, lib.GrB_NAME) == len("my_type") + 1
    # GrB_SIZE is a number, not a string
    with pytest.raises(exceptions.InvalidValue):
        grb_type.grb_type_get_string(T, lib.GrB_SIZE)


def test_semiring_set():
    s = semiring.semiring_new(lib.GrB_MAX_MONOID_FP64, lib.GrB_PLUS_FP64)
    semiring.semiring_set_string(s, lib.GrB_NAME, "max_plus")
    assert semiring.semiring_get_string(s, lib.GrB_NAME) == "max_plus"
    # A semiring's integer properties (its types) are read-only
    code = semiring.semiring_get_int32(s, lib.GrB_INP0_TYPE_CODE)
    assert code == lib.GrB_FP64_CODE
    with pytest.raises(exceptions.InvalidValue):
        semiring.semiring_set_int32(s, lib.GrB_INP0_TYPE_CODE, lib.GrB_INT64_CODE)
    assert semiring.semiring_get_int32(s, lib.GrB_INP0_TYPE_CODE) == code
    # Built-in semirings can't be renamed
    builtin = ffi.new("GrB_Semiring*", lib.GrB_PLUS_TIMES_SEMIRING_INT64)
    with pytest.raises(exceptions.InvalidValue):
        semiring.semiring_set_string(builtin, lib.GrB_NAME, "renamed")


# GraphBLAS doesn't limit the length of a name set with GrB_NAME


def test_descriptor_name_longer_than_256():
    D = ffi.new("GrB_Descriptor*")
    assert lib.GrB_Descriptor_new(D) == lib.GrB_SUCCESS
    D = ffi.gc(D, lib.GrB_Descriptor_free)
    name = "d" * 1000
    descriptor.descriptor_set_string(D[0], lib.GrB_NAME, name)
    assert descriptor.descriptor_get_string(D[0], lib.GrB_NAME) == name
    assert descriptor.descriptor_get_size(D[0], lib.GrB_NAME) == len(name) + 1
    # GrB_NAME is a descriptor's only string.  (GrB_Descriptor_get_String itself ignores the
    # field and returns the name, or "" when none is set, but GrB_Descriptor_get_SIZE rejects
    # every field but GrB_NAME.)
    with pytest.raises(exceptions.InvalidValue):
        descriptor.descriptor_get_string(D[0], lib.GrB_OUTP_FIELD)


def test_semiring_name_longer_than_256():
    s = semiring.semiring_new(lib.GrB_MAX_MONOID_FP64, lib.GrB_PLUS_FP64)
    name = "s" * 1000
    semiring.semiring_set_string(s, lib.GrB_NAME, name)
    assert semiring.semiring_get_string(s, lib.GrB_NAME) == name
    assert semiring.semiring_get_size(s, lib.GrB_NAME) == len(name) + 1


def _matrix_setting(set_, get, value):
    A = matrix.matrix_new(lib.GrB_INT64, 2, 2)
    return value, lambda v: set_(A, v), lambda: get(A)


def _descriptor_setting(field, value):
    D = ffi.new("GrB_Descriptor*")
    assert lib.GrB_Descriptor_new(D) == lib.GrB_SUCCESS
    D = ffi.gc(D, lib.GrB_Descriptor_free)
    return (
        value,
        lambda v: descriptor.descriptor_set_int32(D[0], field, v),
        lambda: descriptor.descriptor_get_int32(D[0], field),
    )


def _setting(new, set_, get, field, value):
    obj = new()
    return value, lambda v: set_(obj, field, v), lambda: get(obj, field)


def _global_setting(set_, get, field):
    return 1 - get(field), lambda v: set_(field, v), lambda: get(field)


# Each function that sets an int32 option or property: `(value, set, get)` for that setting of
# a new object, where `value` is in range and accepted (or rejected by GraphBLAS anyway, for the
# read-only properties of a type or a semiring)
_int32_settings = {
    "context_set_int32": lambda: _setting(
        context.context_new,
        context.context_set_int32,
        context.context_get_int32,
        lib.GxB_CONTEXT_NTHREADS,
        3,
    ),
    "context_set_nthreads": lambda: _setting(
        context.context_new,
        lambda ctx, _, v: context.context_set_nthreads(ctx, v),
        lambda ctx, _: context.context_get_nthreads(ctx),
        None,
        3,
    ),
    "matrix_set_format": lambda: _matrix_setting(
        matrix.matrix_set_format, matrix.matrix_format, lib.GxB_BY_COL
    ),
    "matrix_set_sparsity_control": lambda: _matrix_setting(
        matrix.matrix_set_sparsity_control, matrix.matrix_sparsity_control, lib.GxB_SPARSE
    ),
    "matrix_option_set_int32": lambda: _setting(
        lambda: matrix.matrix_new(lib.GrB_INT64, 2, 2),
        matrix.matrix_option_set_int32,
        matrix.matrix_option_get_int32,
        lib.GxB_FORMAT,
        lib.GxB_BY_COL,
    ),
    "vector_option_set_int32": lambda: _setting(
        lambda: vector.vector_new(lib.GrB_INT64, 2),
        vector.vector_option_set_int32,
        vector.vector_option_get_int32,
        lib.GxB_SPARSITY_CONTROL,
        lib.GxB_SPARSE,
    ),
    "descriptor_set_int32": lambda: _descriptor_setting(lib.GrB_OUTP_FIELD, lib.GrB_REPLACE),
    "grb_type_set_int32": lambda: _setting(
        lambda: grb_type.grb_type_new(8),
        grb_type.grb_type_set_int32,
        grb_type.grb_type_get_int32,
        lib.GrB_EL_TYPE_CODE,
        lib.GrB_INT64_CODE,
    ),
    "semiring_set_int32": lambda: _setting(
        lambda: semiring.semiring_new(lib.GrB_MAX_MONOID_FP64, lib.GrB_PLUS_FP64),
        semiring.semiring_set_int32,
        semiring.semiring_get_int32,
        lib.GrB_INP0_TYPE_CODE,
        lib.GrB_INT64_CODE,
    ),
    "global_option_set_int32": lambda: _global_setting(
        global_options.global_option_set_int32,
        global_options.global_option_get_int32,
        lib.GxB_PRINT_1BASED,
    ),
    "global_set_int32": lambda: _global_setting(
        global_options.global_set_int32, global_options.global_get_int32, lib.GxB_PRINT_1BASED
    ),
}

_global_unsafe = pytest.mark.thread_unsafe(
    reason="a value that wrapped around would set a global option"
)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(name, marks=_global_unsafe if name.startswith("global") else ())
        for name in _int32_settings
    ],
)
def test_set_int32_rejects_non_int32(name):
    # A value that doesn't fit in an int32 is an error, rather than wrapping around to one that
    # does (2**32 + 3 would set 3)
    value, set_, get = _int32_settings[name]()
    old = get()
    try:
        for wrapped in (value + 2**32, value - 2**32, 2**31, -(2**31) - 1):
            with pytest.raises(OverflowError):
                set_(wrapped)
            assert get() == old
        # So is a float or a str, rather than truncated (4.7 would set 4) or taken as a
        # character code ("3" would set 51)
        for wrong in (value + 0.7, "3"):
            with pytest.raises(TypeError):
                set_(wrong)
            assert get() == old
    finally:
        if get() != old:  # put a global option back
            set_(old)
