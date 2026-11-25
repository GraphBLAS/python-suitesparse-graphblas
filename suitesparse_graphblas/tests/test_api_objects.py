from suitesparse_graphblas import lib
from suitesparse_graphblas.api import grb_type


def test_grb_type_definition_longer_than_256():
    fields = "; ".join(f"double x{i}" for i in range(40))
    defn = f"typedef struct {{ {fields}; }} wide_type ;"
    assert len(defn) > 256
    T = grb_type.grb_type_new_named(40 * 8, "wide_type", defn)
    assert grb_type.grb_type_get_string(T, lib.GxB_JIT_C_DEFINITION) == defn
    assert grb_type.grb_type_get_size(T, lib.GxB_JIT_C_DEFINITION) == len(defn) + 1
    assert grb_type.grb_type_get_string(T, lib.GxB_JIT_C_NAME) == "wide_type"
