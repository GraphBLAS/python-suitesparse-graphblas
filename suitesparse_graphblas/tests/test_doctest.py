def test_run_doctests():
    import doctest

    from suitesparse_graphblas import iterator, matrix, scalar, vector

    for mod in (
        matrix,
        vector,
        scalar,
        iterator,
    ):
        doctest.testmod(mod, optionflags=doctest.ELLIPSIS, raise_on_error=True)
