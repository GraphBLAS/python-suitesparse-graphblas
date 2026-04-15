def test_run_doctests():
    import doctest

    from suitesparse_graphblas.api import (
        binaryop,
        container,
        context,
        descriptor,
        global_options,
        grb_type,
        indexbinaryop,
        indexunaryop,
        iterator,
        matrix,
        monoid,
        scalar,
        selectop,
        semiring,
        unaryop,
        vector,
    )

    for mod in (
        matrix,
        vector,
        scalar,
        iterator,
        context,
        global_options,
        grb_type,
        unaryop,
        binaryop,
        indexunaryop,
        indexbinaryop,
        monoid,
        semiring,
        descriptor,
        selectop,
        container,
    ):
        doctest.testmod(mod, optionflags=doctest.ELLIPSIS, raise_on_error=True)
