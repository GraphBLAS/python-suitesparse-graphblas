import sysconfig

from suitesparse_graphblas import _error_code_lookup, ffi, lib

_SUITESPARSE_ORIG_CONFIG = {}


def _get_suitesparse_original_configs():
    """Save the initial configuration for the SuiteSparse:GraphBLAS JIT."""
    if _SUITESPARSE_ORIG_CONFIG:
        raise RuntimeError("suitesparse defaults already gotten!")
    val_ptr = ffi.new("int32_t*")
    # GxB_JIT_C_CONTROL
    info = lib.GxB_Global_Option_get_INT32(lib.GxB_JIT_C_CONTROL, val_ptr)
    if info != lib.GrB_SUCCESS:
        raise _error_code_lookup[info]("Failed to get config for GxB_JIT_C_CONTROL.")
    _SUITESPARSE_ORIG_CONFIG["C_CONTROL"] = val_ptr[0]
    # GxB_JIT_USE_CMAKE
    info = lib.GxB_Global_Option_get_INT32(lib.GxB_JIT_USE_CMAKE, val_ptr)
    if info != lib.GrB_SUCCESS:
        raise _error_code_lookup[info]("Failed to get config for GxB_JIT_USE_CMAKE.")
    _SUITESPARSE_ORIG_CONFIG["USE_CMAKE"] = bool(val_ptr[0])

    val_ptr = ffi.new("char**")
    get_config = lib.GxB_Global_Option_get_CHAR
    for config_name in (
        "C_COMPILER_NAME",
        "C_COMPILER_FLAGS",
        "C_LINKER_FLAGS",
        "C_LIBRARIES",
        "C_CMAKE_LIBS",
        "C_PREFACE",
        "ERROR_LOG",
        "CACHE_PATH",
    ):
        key_obj = getattr(lib, f"GxB_JIT_{config_name}")
        info = get_config(key_obj, val_ptr)
        if info != lib.GrB_SUCCESS:
            raise _error_code_lookup[info](f"Failed to get config for GxB_JIT_{config_name}.")
        _SUITESPARSE_ORIG_CONFIG[config_name] = ffi.string(val_ptr[0]).decode()


def _set_defaults_common(**override):
    set_config = lib.GxB_Global_Option_set_INT32
    # GxB_JIT_C_CONTROL (turn on)
    info = set_config(lib.GxB_JIT_C_CONTROL, ffi.cast("int32_t", lib.GxB_JIT_ON))
    if info != lib.GrB_SUCCESS:
        raise _error_code_lookup[info]("Failed to set config for GxB_JIT_C_CONTROL")
    # GxB_JIT_USE_CMAKE
    info = set_config(
        lib.GxB_JIT_USE_CMAKE, ffi.cast("int32_t", _SUITESPARSE_ORIG_CONFIG["USE_CMAKE"])
    )
    if info != lib.GrB_SUCCESS:
        raise _error_code_lookup[info]("Failed to set config for GxB_JIT_USE_CMAKE")

    set_config = lib.GxB_Global_Option_set_CHAR
    for config_name in (
        "C_COMPILER_NAME",
        "C_COMPILER_FLAGS",
        "C_LINKER_FLAGS",
        "C_LIBRARIES",
        "C_CMAKE_LIBS",
        "C_PREFACE",
        "ERROR_LOG",
        "CACHE_PATH",
    ):
        key_obj = getattr(lib, f"GxB_JIT_{config_name}")
        if config_name in override:
            val = override[config_name]
        else:
            val = _SUITESPARSE_ORIG_CONFIG[config_name]
        val_obj = ffi.new("char[]", val.encode())
        info = set_config(key_obj, val_obj)
        if info != lib.GrB_SUCCESS:
            raise _error_code_lookup[info](f"Failed to set config for GxB_JIT_{config_name}")


def set_suitesparse_defaults():
    """Enable the JIT and set all JIT configs to the SuiteSparse:GraphBLAS defaults.

    Using the SuiteSparse:GraphBLAS JIT requires a C compiler.

    This sets values for:
    - GxB_JIT_C_CONTROL
    - GxB_JIT_USE_CMAKE
    - GxB_JIT_C_COMPILER_NAME
    - GxB_JIT_C_COMPILER_FLAGS
    - GxB_JIT_C_LINKER_FLAGS
    - GxB_JIT_C_LIBRARIES
    - GxB_JIT_C_CMAKE_LIBS
    - GxB_JIT_C_PREFACE
    - GxB_JIT_ERROR_LOG
    - GxB_JIT_CACHE_PATH

    See Also
    --------
    set_python_defaults : set compiler config from Python's ``sysconfig`` module
    """
    _set_defaults_common()


def set_python_defaults():
    """Enable the JIT and set configs with compiler configs from ``sysconfig`` module.

    Using the SuiteSparse:GraphBLAS JIT requires a C compiler.

    This uses ``sysconfig`` to set the values for:
    - GxB_JIT_C_COMPILER_NAME
    - GxB_JIT_C_COMPILER_FLAGS
    - GxB_JIT_C_LIBRARIES

    and uses the SuiteSparse:GraphBLAS defaults for:
    - GxB_JIT_USE_CMAKE
    - GxB_JIT_C_LINKER_FLAGS
    - GxB_JIT_C_CMAKE_LIBS
    - GxB_JIT_C_PREFACE
    - GxB_JIT_ERROR_LOG
    - GxB_JIT_CACHE_PATH

    See Also
    --------
    set_suitesparse_defaults : reset JIT config with SuiteSparse:GraphBLAS defaults.
    """
    cc = sysconfig.get_config_var("CC")
    cflags = sysconfig.get_config_var("CFLAGS")
    include = sysconfig.get_path("include")
    libs = sysconfig.get_config_var("LIBS")
    if cc is None or cflags is None or include is None or libs is None:
        raise RuntimeError("C compiler configuration not found by `sysconfig` module")
    _set_defaults_common(
        C_COMPILER_NAME=cc,
        C_COMPILER_FLAGS=f"{cflags} -I{include}",
        C_LIBRARIES=libs,
    )


def disable():
    """Completely disable the SuiteSparse:GraphBLAS JIT.

    This sets GxB_JIT_C_CONTROL to GxB_JIT_OFF.
    """
    set_config = lib.GxB_Global_Option_set_INT32
    info = set_config(lib.GxB_JIT_C_CONTROL, ffi.cast("int32_t", lib.GxB_JIT_OFF))
    if info != lib.GrB_SUCCESS:
        raise _error_code_lookup[info]("Failed to set config for GxB_JIT_C_CONTROL")


def load():
    """Allow the SuiteSparse:GraphBLAS JIT to load JIT kernels, but don't compile.

    This sets GxB_JIT_C_CONTROL to GxB_JIT_LOAD.
    """
    set_config = lib.GxB_Global_Option_set_INT32
    info = set_config(lib.GxB_JIT_C_CONTROL, ffi.cast("int32_t", lib.GxB_JIT_LOAD))
    if info != lib.GrB_SUCCESS:
        raise _error_code_lookup[info]("Failed to set config for GxB_JIT_C_CONTROL")
