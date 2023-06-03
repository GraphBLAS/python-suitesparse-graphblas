from suitesparse_graphblas import ffi, lib


def test_print_jit_config():
    print()
    print("===================================")
    print("Printing default JIT configurations")
    print("-----------------------------------")
    jit_c_control = {
        lib.GxB_JIT_OFF: "off",
        lib.GxB_JIT_PAUSE: "pause",
        lib.GxB_JIT_RUN: "run",
        lib.GxB_JIT_LOAD: "load",
        lib.GxB_JIT_ON: "on",
    }
    val_ptr = ffi.new("int32_t*")
    assert lib.GxB_Global_Option_get_INT32(lib.GxB_JIT_C_CONTROL, val_ptr) == lib.GrB_SUCCESS
    print("JIT_C_CONTROL", jit_c_control[val_ptr[0]])

    assert lib.GxB_Global_Option_get_INT32(lib.GxB_JIT_USE_CMAKE, val_ptr) == lib.GrB_SUCCESS
    print("JIT_USE_CMAKE", bool(val_ptr[0]))

    func = lib.GxB_Global_Option_get_CHAR
    names = [
        "JIT_C_COMPILER_NAME",
        "JIT_C_COMPILER_FLAGS",
        "JIT_C_LINKER_FLAGS",
        "JIT_C_LIBRARIES",
        "JIT_C_CMAKE_LIBS",
        "JIT_C_PREFACE",
        "JIT_ERROR_LOG",
        "JIT_CACHE_PATH",
    ]
    val_ptr = ffi.new("char**")
    for name in names:
        obj = getattr(lib, f"GxB_{name}")
        assert func(obj, val_ptr) == lib.GrB_SUCCESS
        print(name, ffi.string(val_ptr[0]).decode())
    print("===================================")
