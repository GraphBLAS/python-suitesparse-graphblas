if __name__ == "__main__":
    import numpy as np
    import pytest

    import suitesparse_graphblas as ssgb
    from suitesparse_graphblas import burble, exceptions, lib, utils
    from suitesparse_graphblas.api import container, global_options

    assert ssgb.is_initialized() is False

    # Before initialize, GraphBLAS calls fail with GrB_PANIC
    for func in (lambda: burble.is_enabled, burble.enable, burble.disable):
        with pytest.raises(exceptions.Panic, match="has GraphBLAS been initialized"):
            func()
    with pytest.raises(exceptions.Panic, match="info=-101"):
        global_options.global_option_get_int32(lib.GxB_BURBLE)
    with pytest.raises(exceptions.Panic, match="info=-101"):
        container.container_new()

    # An unknown memory manager is an error, and leaves GraphBLAS uninitialized
    with pytest.raises(ValueError, match='"numpy" or "c"'):
        ssgb.initialize(memory_manager="malloc")
    assert ssgb.is_initialized() is False

    # Every pytest session initializes with the default, "numpy" (see conftest.py), so
    # use the C library's allocator here, spelled in another case
    ssgb.initialize(memory_manager="C")
    assert ssgb.is_initialized() is True
    # NumPy's memory can't be handed to GraphBLAS without a copy, since GraphBLAS frees it
    # with libc free
    assert utils.can_unclaim_buffer(np.empty(3)) is False
    with pytest.raises(RuntimeError, match="GraphBLAS is already initialized"):
        ssgb.initialize()
    assert burble.is_enabled is False
