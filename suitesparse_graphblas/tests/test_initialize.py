if __name__ == "__main__":
    import suitesparse_graphblas as ssgb
    import pytest

    assert ssgb.is_initialized() is False
    ssgb.initialize()
    assert ssgb.is_initialized() is True
    with pytest.raises(RuntimeError, match="GraphBLAS is already initialized"):
        ssgb.initialize()
