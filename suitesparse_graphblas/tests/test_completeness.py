"""Guards that the package covers everything the GraphBLAS it is built against declares.

Each expected set is derived from GraphBLAS itself (the cffi `lib` and `ffi`), never from a hand
list, so a GraphBLAS that adds a type, a compression codec or a sparsity format fails a test here
and prompts a decision, instead of the package quietly falling behind.
"""

import importlib
import re

import pytest

import suitesparse_graphblas as sg
from suitesparse_graphblas import ffi, lib
from suitesparse_graphblas.api.io import binary

_TYPE_NAME = re.compile(r"G[rx]B_(BOOL|U?INT\d+|FP\d+|FC\d+)")

_GrB_Type = ffi.typeof("GrB_Type")

# The complex types are present exactly when the GraphBLAS was built with them
_builtin_names = sorted(n for n in dir(lib) if _TYPE_NAME.fullmatch(n))
_builtin_types = {n: getattr(lib, n) for n in _builtin_names}


def _type_code(T):
    value = ffi.new("int32_t*")
    assert lib.GrB_Type_get_INT32(T, value, lib.GrB_EL_TYPE_CODE) == lib.GrB_SUCCESS
    return value[0]


def _type_name(T):
    size = ffi.new("size_t*")
    assert lib.GrB_Type_get_SIZE(T, size, lib.GrB_NAME) == lib.GrB_SUCCESS
    buf = ffi.new("char[]", size[0])
    assert lib.GrB_Type_get_String(T, buf, lib.GrB_NAME) == lib.GrB_SUCCESS
    return ffi.string(buf).decode()


def test_builtin_type_names_are_all_types():
    # The pattern that finds the built-in types must not pick up anything else
    for name in _builtin_names:
        assert ffi.typeof(getattr(lib, name)) is _GrB_Type, name
    assert len(_builtin_names) == (13 if sg.supports_complex() else 11)
    # ... and must miss none: every GrB_Type constant is a built-in type
    all_types = {
        n
        for n in dir(lib)
        if n.startswith(("GrB_", "GxB_"))
        and isinstance(getattr(lib, n), ffi.CData)
        and ffi.typeof(getattr(lib, n)) is _GrB_Type
    }
    assert all_types == set(_builtin_names)


def test_grb_types_has_every_builtin_type():
    assert sg.grb_types == frozenset(_builtin_types.values())


def test_type_families_partition_by_type_code():
    codes = ffi.typeof("GrB_Type_Code").relements
    families = {
        "BOOL": sg.bool_types,
        "INT": sg.signed_integer_types,
        "UINT": sg.unsigned_integer_types,
        "FP": sg.real_types,
        "FC": sg.complex_types,
    }
    expected = {name: set() for name in families}
    for name, T in _builtin_types.items():
        code = _type_code(T)
        (code_name,) = (n for n, v in codes.items() if v == code)
        assert code_name == f"{name}_CODE"
        family = re.fullmatch(r"G[rx]B_(BOOL|U?INT|FP|FC)\d*", name).group(1)
        expected[family].add(T)
    for family, types in families.items():
        assert types == expected[family], family
    assert sg.integer_types == sg.signed_integer_types | sg.unsigned_integer_types
    # The type codes are the built-in types plus the user-defined one
    assert set(codes) == {f"{n}_CODE" for n in _builtin_names} | {"GrB_UDT_CODE"}


def test_binary_type_tables_cover_every_type():
    assert set(binary._ss_typecodes) == set(binary._ss_typenames) == set(sg.grb_types)
    # The on-disk ordinals are unique, and decode back to the type
    assert len(set(binary._ss_typecodes.values())) == len(binary._ss_typecodes)
    assert {binary._ss_codetypes[c] for c in binary._ss_typecodes.values()} == set(sg.grb_types)


@pytest.mark.parametrize("name", _builtin_names)
def test_binary_type_name_matches_graphblas(name):
    T = _builtin_types[name]
    assert binary._ss_typenames[T] == _type_name(T)


_compression_names = sorted(
    n for n in dir(lib) if n.startswith("GxB_COMPRESSION_") and not callable(getattr(lib, n))
)


def test_compression_codecs_are_known():
    # A new codec needs a decision: whether it takes a level, as LZ4HC and ZSTD do (see
    # `serialize.get_serialize_desc`), and a case in the `_serialize_options` tests in test_io.py
    assert _compression_names == [
        "GxB_COMPRESSION_DEFAULT",
        "GxB_COMPRESSION_LZ4",
        "GxB_COMPRESSION_LZ4HC",
        "GxB_COMPRESSION_NONE",
        "GxB_COMPRESSION_ZSTD",
    ]


@pytest.mark.parametrize("name", _compression_names)
def test_every_compression_codec_is_tested(name):
    from suitesparse_graphblas.tests.test_io import _serialize_options

    assert getattr(lib, name) in {compression for compression, _ in _serialize_options.values()}


_ALL_TYPES = [n[4:].lower() for n in _builtin_names]

# (module, family prefix, whether there is also a user-defined-type variant)
_families = [
    ("matrix", "get", False),
    ("matrix", "set", False),
    ("matrix", "matrix_build", False),
    ("matrix", "matrix_extract_tuples", False),
    ("vector", "get", False),
    ("vector", "set", False),
    ("vector", "vector_build", False),
    ("vector", "vector_extract_tuples", False),
    ("scalar", "get", False),
    ("scalar", "set", False),
    ("iterator", "iterator_get", False),
    ("monoid", "monoid_new", True),
    ("monoid", "monoid_terminal_new", True),
]


@pytest.mark.parametrize("module, prefix, udt", _families)
def test_wrapper_family_covers_every_type(module, prefix, udt):
    mod = importlib.import_module(f"suitesparse_graphblas.api.{module}")
    suffixes = _ALL_TYPES + (["udt"] if udt else [])
    # Functions named `<prefix>_<type>` (functions of other families, such as
    # `matrix_option_get_int32`, are named differently)
    found = {
        name[len(prefix) + 1 :]
        for name in vars(mod)
        if name.startswith(prefix + "_")
        and callable(getattr(mod, name))
        and re.fullmatch(r"(bool|u?int\d+|fp\d+|fc\d+|udt)", name[len(prefix) + 1 :])
    }
    assert found == set(suffixes)


def test_sparsity_formats_are_known():
    from suitesparse_graphblas.tests.test_io import _sparsities

    found = {
        getattr(lib, n)
        for n in dir(lib)
        if re.fullmatch(r"GxB_(HYPERSPARSE|SPARSE|BITMAP|FULL)", n)
    }
    assert found == set(_sparsities)
    from suitesparse_graphblas.tests import test_iterator

    assert set(test_iterator._sparsities) == found
