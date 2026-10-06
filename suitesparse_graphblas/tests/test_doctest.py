import doctest
import importlib
import pkgutil

from suitesparse_graphblas import api, utils


def test_run_doctests():
    # Every module of the functional API, found rather than listed so none is missed: the
    # wheel builds' test command does not collect doctests, so they run only through here
    attempted = 0
    names = sorted(info.name for info in pkgutil.walk_packages(api.__path__, f"{api.__name__}."))
    for name in names:
        mod = importlib.import_module(name)
        _, tried = doctest.testmod(mod, optionflags=doctest.ELLIPSIS, raise_on_error=True)
        attempted += tried
    assert attempted > 0

    # `testmod` doesn't find these: Cython functions, or wrapped by `contextmanager`
    runner = doctest.DebugRunner(optionflags=doctest.ELLIPSIS)  # raises on failure
    for test in doctest.DocTestFinder().find(utils.give_buffer, "give_buffer", globs={}):
        runner.run(test)
    assert runner.tries > 0
