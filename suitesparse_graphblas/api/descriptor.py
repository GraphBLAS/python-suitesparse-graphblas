from suitesparse_graphblas import check_status, ffi, lib

from .utils import _capture_c_output  # noqa: F401


def descriptor_free(desc):
    """Free a descriptor.

    >>> desc = descriptor_new()
    >>> descriptor_free(desc)

    """
    check_status(desc, lib.GrB_Descriptor_free(desc))


def descriptor_new(*, free=descriptor_free):
    """Create a new ``GrB_Descriptor`` and initialize it.

    The ``free`` argument is called when the object is garbage
    collected, the default is ``descriptor.descriptor_free()``.  If ``free``
    is None then there is no automatic garbage collection and it is up
    to the user to free the descriptor.

    >>> desc = descriptor_new()

    """
    desc = ffi.new("GrB_Descriptor*")
    check_status(desc, lib.GrB_Descriptor_new(desc))
    if free:
        return ffi.gc(desc, free)
    return desc


def descriptor_wait(desc, waitmode=lib.GrB_COMPLETE):
    """Wait for a descriptor to complete pending operations.

    >>> desc = descriptor_new()
    >>> descriptor_wait(desc)

    """
    check_status(desc, lib.GrB_Descriptor_wait(desc[0], waitmode))


def descriptor_print(desc, name="", level=lib.GxB_COMPLETE):
    """Print a descriptor to stdout.

    ``level`` controls verbosity: ``lib.GxB_SILENT``, ``lib.GxB_SUMMARY``,
    ``lib.GxB_SHORT``, ``lib.GxB_COMPLETE``, ``lib.GxB_SHORT_VERBOSE``,
    or ``lib.GxB_COMPLETE_VERBOSE``.

    >>> desc = descriptor_new()
    >>> out = _capture_c_output(descriptor_print, desc, 'desc', lib.GxB_SHORT)
    >>> 'Descriptor' in out
    True

    """
    check_status(desc, lib.GxB_Descriptor_fprint(
        desc[0], name.encode() if isinstance(name, str) else name,
        level, ffi.NULL,
    ))


def descriptor_fprint(desc, f, name="", level=lib.GxB_COMPLETE):
    """Print a descriptor to a C FILE* stream.

    Pass ``ffi.NULL`` for ``f`` to print to stdout.
    ``level`` controls verbosity (see :func:`descriptor_print`).

    >>> desc = descriptor_new()
    >>> out = _capture_c_output(descriptor_fprint, desc, ffi.NULL, 'desc', lib.GxB_SHORT)
    >>> 'Descriptor' in out
    True

    """
    check_status(desc, lib.GxB_Descriptor_fprint(
        desc[0], name.encode() if isinstance(name, str) else name,
        level, f,
    ))


# ---------------------------------------------------------------------------
# Get / Set
# ---------------------------------------------------------------------------


def descriptor_get_int32(desc, field):
    """Get a descriptor property as an int32.

    >>> desc = descriptor_new()
    >>> isinstance(descriptor_get_int32(desc, lib.GrB_OUTP_FIELD), int)
    True

    """
    val = ffi.new("int32_t*")
    check_status(desc, lib.GrB_Descriptor_get_INT32(desc[0], val, field))
    return val[0]


def descriptor_set_int32(desc, field, value):
    """Set a descriptor property from an int32.

    >>> desc = descriptor_new()
    >>> descriptor_set_int32(desc, lib.GrB_OUTP_FIELD, lib.GrB_REPLACE)

    """
    check_status(desc, lib.GrB_Descriptor_set_INT32(
        desc[0], ffi.cast("int32_t", value), field,
    ))


def descriptor_get_size(desc, field):
    """Get a descriptor property as a size_t.

    >>> desc = descriptor_new()
    >>> isinstance(descriptor_get_size(desc, lib.GrB_NAME), int)
    True

    """
    val = ffi.new("size_t*")
    check_status(desc, lib.GrB_Descriptor_get_SIZE(desc[0], val, field))
    return val[0]


def descriptor_get_string(desc, field):
    """Get a descriptor property as a string.

    >>> desc = descriptor_new()
    >>> isinstance(descriptor_get_string(desc, lib.GrB_NAME), str)
    True

    """
    val = ffi.new("char[256]")
    check_status(desc, lib.GrB_Descriptor_get_String(desc[0], val, field))
    return ffi.string(val).decode()


def descriptor_set_string(desc, field, value):
    """Set a descriptor property from a string.

    >>> desc = descriptor_new()
    >>> descriptor_set_string(desc, lib.GrB_NAME, "my_desc")
    >>> descriptor_get_string(desc, lib.GrB_NAME)
    'my_desc'

    """
    check_status(desc, lib.GrB_Descriptor_set_String(desc[0], value.encode(), field))
