from suitesparse_graphblas import check_status, ffi, lib

from .utils import _capture_c_output  # noqa: F401


def descriptor_wait(desc, waitmode=lib.GrB_COMPLETE):
    """Wait for a descriptor to complete pending operations.

    >>> descriptor_wait(lib.GrB_DESC_S)

    """
    check_status(desc, lib.GrB_Descriptor_wait(desc, waitmode))


def descriptor_print(desc, name="", level=lib.GxB_COMPLETE):
    """Print a descriptor to stdout.

    ``level`` controls verbosity: ``lib.GxB_SILENT``, ``lib.GxB_SUMMARY``,
    ``lib.GxB_SHORT``, ``lib.GxB_COMPLETE``, ``lib.GxB_SHORT_VERBOSE``,
    or ``lib.GxB_COMPLETE_VERBOSE``.

    >>> out = _capture_c_output(descriptor_print, lib.GrB_DESC_S, 'desc', lib.GxB_SHORT)
    >>> 'Descriptor' in out
    True

    """
    check_status(desc, lib.GxB_Descriptor_fprint(
        desc, name.encode() if isinstance(name, str) else name,
        level, ffi.NULL,
    ))


def descriptor_fprint(desc, f, name="", level=lib.GxB_COMPLETE):
    """Print a descriptor to a C FILE* stream.

    Pass ``ffi.NULL`` for ``f`` to print to stdout.
    ``level`` controls verbosity (see :func:`descriptor_print`).

    >>> out = _capture_c_output(descriptor_fprint, lib.GrB_DESC_S, ffi.NULL, 'desc', lib.GxB_SHORT)
    >>> 'Descriptor' in out
    True

    """
    check_status(desc, lib.GxB_Descriptor_fprint(
        desc, name.encode() if isinstance(name, str) else name,
        level, f,
    ))


# ---------------------------------------------------------------------------
# Get / Set
# ---------------------------------------------------------------------------


def descriptor_get_int32(desc, field):
    """Get a descriptor property as an int32.

    >>> isinstance(descriptor_get_int32(lib.GrB_DESC_S, lib.GrB_OUTP_FIELD), int)
    True

    """
    val = ffi.new("int32_t*")
    check_status(desc, lib.GrB_Descriptor_get_INT32(desc, val, field))
    return val[0]


def descriptor_set_int32(desc, field, value):
    """Set a descriptor property from an int32.

    >>> from suitesparse_graphblas.api.descriptor import descriptor_get_int32
    >>> descriptor_get_int32(lib.GrB_DESC_S, lib.GrB_OUTP_FIELD) == lib.GrB_REPLACE
    False

    """
    check_status(desc, lib.GrB_Descriptor_set_INT32(
        desc, ffi.cast("int32_t", value), field,
    ))


def descriptor_get_size(desc, field):
    """Get a descriptor property as a size_t.

    >>> isinstance(descriptor_get_size(lib.GrB_DESC_S, lib.GrB_NAME), int)
    True

    """
    val = ffi.new("size_t*")
    check_status(desc, lib.GrB_Descriptor_get_SIZE(desc, val, field))
    return val[0]


def descriptor_get_string(desc, field):
    """Get a descriptor property as a string.

    >>> isinstance(descriptor_get_string(lib.GrB_DESC_S, lib.GrB_NAME), str)
    True

    """
    val = ffi.new("char[256]")
    check_status(desc, lib.GrB_Descriptor_get_String(desc, val, field))
    return ffi.string(val).decode()


def descriptor_set_string(desc, field, value):
    """Set a descriptor property from a string.

    >>> descriptor_get_string(lib.GrB_DESC_S, lib.GrB_NAME)
    'GrB_DESC_S'

    """
    check_status(desc, lib.GrB_Descriptor_set_String(desc, value.encode(), field))
