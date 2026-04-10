from suitesparse_graphblas import _error_code_lookup, ffi, lib


def container_free(c):
    """Free a container.

    >>> c = container_new()
    >>> container_free(c)

    """
    info = lib.GxB_Container_free(c)
    if info != lib.GrB_SUCCESS:
        raise _error_code_lookup.get(info, RuntimeError)(
            f"GxB_Container_free failed with info={info}"
        )


def container_new(*, free=container_free):
    """Create a new ``GxB_Container`` and initialize it.

    The ``free`` argument is called when the object is garbage
    collected, the default is ``container.container_free()``.  If ``free``
    is None then there is no automatic garbage collection and it is up
    to the user to free the container.

    >>> c = container_new()

    """
    c = ffi.new("GxB_Container*")
    info = lib.GxB_Container_new(c)
    if info != lib.GrB_SUCCESS:
        raise _error_code_lookup.get(info, RuntimeError)(
            f"GxB_Container_new failed with info={info}"
        )
    if free:
        return ffi.gc(c, free)
    return c
