def test_run_doctests():
    from suitesparse_graphblas import matrix
    from suitesparse_graphblas import vector
    from suitesparse_graphblas import scalar
    import sys, doctest

    for mod in (
        matrix,
        vector,
        scalar,
    ):
        doctest.testmod(mod, optionflags=doctest.ELLIPSIS, raise_on_error=True)
