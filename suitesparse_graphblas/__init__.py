from ._graphblas import ffi, lib  # noqa
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
