def test_run_doctests():
    import doctest
    from suitesparse_graphblas import (
        matrix,
        vector,
        scalar,
    )

    for mod in (
        matrix,
        vector,
        scalar,
    ):
        doctest.testmod(mod, optionflags=doctest.ELLIPSIS, raise_on_error=True)
