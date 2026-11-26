import pytest

from suitesparse_graphblas import exceptions, ffi, lib
from suitesparse_graphblas.api import container, context, descriptor, grb_type, semiring


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
