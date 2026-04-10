from suitesparse_graphblas import _error_code_lookup, ffi, lib


def global_option_get_int32(field):
    """Get a global option as an int32.

    >>> global_option_get_int32(lib.GxB_BURBLE) in (0, 1)
    True

    """
    val = ffi.new("int32_t*")
    info = lib.GxB_Global_Option_get_INT32(field, val)
    if info != lib.GrB_SUCCESS:
        raise _error_code_lookup.get(info, RuntimeError)(
            f"GxB_Global_Option_get_INT32 failed with info={info}"
        )
    return val[0]


def global_option_set_int32(field, value):
    """Set a global option from an int32.

    >>> global_option_set_int32(lib.GxB_BURBLE, 0)

    """
    info = lib.GxB_Global_Option_set_INT32(field, ffi.cast("int32_t", value))
    if info != lib.GrB_SUCCESS:
        raise _error_code_lookup.get(info, RuntimeError)(
            f"GxB_Global_Option_set_INT32 failed with info={info}"
        )


def global_option_get_fp64(field):
    """Get a global option as a float64.

    >>> isinstance(global_option_get_fp64(lib.GxB_HYPER_SWITCH), float)
    True

    """
    val = ffi.new("double*")
    info = lib.GxB_Global_Option_get_FP64(field, val)
    if info != lib.GrB_SUCCESS:
        raise _error_code_lookup.get(info, RuntimeError)(
            f"GxB_Global_Option_get_FP64 failed with info={info}"
        )
    return val[0]


def global_option_set_fp64(field, value):
    """Set a global option from a float64.

    >>> default = global_option_get_fp64(lib.GxB_HYPER_SWITCH)
    >>> global_option_set_fp64(lib.GxB_HYPER_SWITCH, default)

    """
    info = lib.GxB_Global_Option_set_FP64(field, ffi.cast("double", value))
    if info != lib.GrB_SUCCESS:
        raise _error_code_lookup.get(info, RuntimeError)(
            f"GxB_Global_Option_set_FP64 failed with info={info}"
        )


def global_option_get_char(field):
    """Get a global option as a string.

    >>> 'SuiteSparse:GraphBLAS' in global_option_get_char(lib.GxB_LIBRARY_NAME)
    True

    """
    val = ffi.new("char**")
    info = lib.GxB_Global_Option_get_CHAR(field, val)
    if info != lib.GrB_SUCCESS:
        raise _error_code_lookup.get(info, RuntimeError)(
            f"GxB_Global_Option_get_CHAR failed with info={info}"
        )
    return ffi.string(val[0]).decode()
