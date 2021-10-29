if __name__ == "__main__":
    import pytest

    import suitesparse_graphblas as ssgb

    assert ssgb.is_initialized() is False
    ssgb.initialize()
    assert ssgb.is_initialized() is True
    with pytest.raises(RuntimeError, match="GraphBLAS is already initialized"):
        ssgb.initialize()
