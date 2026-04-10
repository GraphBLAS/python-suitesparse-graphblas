from suitesparse_graphblas import check_status, ffi, lib

from .utils import _capture_c_output  # noqa: F401


def grb_type_free(T):
    """Free a type.

    >>> T = grb_type_new(8)
    >>> grb_type_free(T)

    """
    check_status(T, lib.GrB_Type_free(T))


def grb_type_new(sizeof_ctype, *, free=grb_type_free):
    """Create a new ``GrB_Type`` and initialize it.

    The ``free`` argument is called when the object is garbage
    collected, the default is ``grb_type.grb_type_free()``.  If ``free`` is
    None then there is no automatic garbage collection and it is up to
    the user to free the type.

    >>> T = grb_type_new(8)

    """
    T = ffi.new("GrB_Type*")
    check_status(T, lib.GrB_Type_new(T, sizeof_ctype))
    if free:
        return ffi.gc(T, free)
    return T


def grb_type_new_named(sizeof_ctype, type_name, type_defn, *, free=grb_type_free):
    """Create a new named ``GrB_Type`` and initialize it.

    ``type_name`` and ``type_defn`` are strings describing the type.

    >>> T = grb_type_new_named(8, "my_type", "int64_t")

    """
    T = ffi.new("GrB_Type*")
    check_status(T, lib.GxB_Type_new(T, sizeof_ctype, type_name.encode(), type_defn.encode()))
    if free:
        return ffi.gc(T, free)
    return T


def grb_type_wait(T, waitmode=lib.GrB_COMPLETE):
    """Wait for a type to complete pending operations.

    >>> T = grb_type_new(8)
    >>> grb_type_wait(T)

    """
    check_status(T, lib.GrB_Type_wait(T[0], waitmode))


def grb_type_print(T, name="", level=lib.GxB_COMPLETE):
    """Print a type to stdout.

    ``level`` controls verbosity: ``lib.GxB_SILENT``, ``lib.GxB_SUMMARY``,
    ``lib.GxB_SHORT``, ``lib.GxB_COMPLETE``, ``lib.GxB_SHORT_VERBOSE``,
    or ``lib.GxB_COMPLETE_VERBOSE``.

    >>> T = ffi.new("GrB_Type*", lib.GrB_INT64)
    >>> out = _capture_c_output(grb_type_print, T, 'T', lib.GxB_SHORT)
    >>> 'int64_t' in out
    True

    """
    check_status(T, lib.GxB_Type_fprint(
        T[0], name.encode() if isinstance(name, str) else name,
        level, ffi.NULL,
    ))


def grb_type_fprint(T, f, name="", level=lib.GxB_COMPLETE):
    """Print a type to a C FILE* stream.

    Pass ``ffi.NULL`` for ``f`` to print to stdout.
    ``level`` controls verbosity (see :func:`grb_type_print`).

    >>> T = ffi.new("GrB_Type*", lib.GrB_INT64)
    >>> out = _capture_c_output(grb_type_fprint, T, ffi.NULL, 'T', lib.GxB_SHORT)
    >>> 'int64_t' in out
    True

    """
    check_status(T, lib.GxB_Type_fprint(
        T[0], name.encode() if isinstance(name, str) else name,
        level, f,
    ))


# ---------------------------------------------------------------------------
# Get / Set
# ---------------------------------------------------------------------------


def grb_type_get_int32(T, field):
    """Get a type property as an int32.

    >>> T = ffi.new("GrB_Type*", lib.GrB_INT64)
    >>> isinstance(grb_type_get_int32(T, lib.GrB_EL_TYPE_CODE), int)
    True

    """
    val = ffi.new("int32_t*")
    check_status(T, lib.GrB_Type_get_INT32(T[0], val, field))
    return val[0]


def grb_type_set_int32(T, field, value):
    """Set a type property from an int32.

    >>> T = grb_type_new(8)
    >>> grb_type_set_int32(T, lib.GrB_EL_TYPE_CODE, 11)
    Traceback (most recent call last):
        ...
    suitesparse_graphblas.exceptions.InvalidValue

    """
    check_status(T, lib.GrB_Type_set_INT32(T[0], ffi.cast("int32_t", value), field))


def grb_type_get_size(T, field):
    """Get a type property as a size_t.

    >>> T = ffi.new("GrB_Type*", lib.GrB_INT64)
    >>> grb_type_get_size(T, lib.GrB_SIZE)
    8

    """
    val = ffi.new("size_t*")
    check_status(T, lib.GrB_Type_get_SIZE(T[0], val, field))
    return val[0]


def grb_type_get_string(T, field):
    """Get a type property as a string.

    >>> T = ffi.new("GrB_Type*", lib.GrB_INT64)
    >>> grb_type_get_string(T, lib.GrB_NAME)
    'GrB_INT64'

    """
    val = ffi.new("char[256]")
    check_status(T, lib.GrB_Type_get_String(T[0], val, field))
    return ffi.string(val).decode()


def grb_type_set_string(T, field, value):
    """Set a type property from a string."""
    check_status(T, lib.GrB_Type_set_String(T[0], value.encode(), field))


# ---------------------------------------------------------------------------
# Legacy GxB accessors
# ---------------------------------------------------------------------------


def grb_type_size(T):
    """Return the size of the type in bytes.

    >>> T = ffi.new("GrB_Type*", lib.GrB_INT64)
    >>> grb_type_size(T)
    8

    """
    sz = ffi.new("size_t*")
    check_status(T, lib.GxB_Type_size(sz, T[0]))
    return sz[0]


def grb_type_name(T):
    """Return the name of the type.

    >>> T = ffi.new("GrB_Type*", lib.GrB_INT64)
    >>> grb_type_name(T)
    'int64_t'

    """
    name = ffi.new("char[256]")
    check_status(T, lib.GxB_Type_name(name, T[0]))
    return ffi.string(name).decode()


def grb_type_from_name(type_name):
    """Look up a built-in type by name.

    >>> grb_type_from_name("GrB_INT64") == lib.GrB_INT64
    True

    """
    T = ffi.new("GrB_Type*")
    check_status(T, lib.GxB_Type_from_name(T, type_name.encode()))
    return T[0]
