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


def global_get_int32(field):
    """Get a global property as an int32.

    >>> isinstance(global_get_int32(lib.GxB_NTHREADS), int)
    True

    """
    val = ffi.new("int32_t*")
    info = lib.GrB_Global_get_INT32(lib.GrB_GLOBAL, val, field)
    if info != lib.GrB_SUCCESS:
        raise _error_code_lookup.get(info, RuntimeError)(
            f"GrB_Global_get_INT32 failed with info={info}"
        )
    return val[0]


def global_set_int32(field, value):
    """Set a global property from an int32.

    >>> nthreads = global_get_int32(lib.GxB_NTHREADS)
    >>> global_set_int32(lib.GxB_NTHREADS, nthreads)

    """
    info = lib.GrB_Global_set_INT32(lib.GrB_GLOBAL, ffi.cast("int32_t", value), field)
    if info != lib.GrB_SUCCESS:
        raise _error_code_lookup.get(info, RuntimeError)(
            f"GrB_Global_set_INT32 failed with info={info}"
        )


def global_get_size(field):
    """Get a global property as a size_t.

    >>> isinstance(global_get_size(lib.GrB_NAME), int)
    True

    """
    val = ffi.new("size_t*")
    info = lib.GrB_Global_get_SIZE(lib.GrB_GLOBAL, val, field)
    if info != lib.GrB_SUCCESS:
        raise _error_code_lookup.get(info, RuntimeError)(
            f"GrB_Global_get_SIZE failed with info={info}"
        )
    return val[0]


def global_get_string(field):
    """Get a global property as a string.

    >>> 'SuiteSparse:GraphBLAS' in global_get_string(lib.GrB_NAME)
    True

    """
    val = ffi.new("char[256]")
    info = lib.GrB_Global_get_String(lib.GrB_GLOBAL, val, field)
    if info != lib.GrB_SUCCESS:
        raise _error_code_lookup.get(info, RuntimeError)(
            f"GrB_Global_get_String failed with info={info}"
        )
    return ffi.string(val).decode()


def global_set_string(field, value):
    """Set a global property from a string."""
    info = lib.GrB_Global_set_String(lib.GrB_GLOBAL, value.encode(), field)
    if info != lib.GrB_SUCCESS:
        raise _error_code_lookup.get(info, RuntimeError)(
            f"GrB_Global_set_String failed with info={info}"
        )
