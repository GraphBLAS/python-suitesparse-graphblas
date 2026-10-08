# python-suitesparse-graphblas

[![Version](https://img.shields.io/pypi/v/suitesparse-graphblas.svg)](https://pypi.org/project/suitesparse-graphblas/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/GraphBLAS/python-suitesparse-graphblas/blob/main/LICENSE)
[![Build Status](https://github.com/GraphBLAS/python-suitesparse-graphblas/workflows/Test/badge.svg)](https://github.com/GraphBLAS/python-suitesparse-graphblas/actions)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Python CFFI Binding around
[SuiteSparse:GraphBLAS](https://github.com/DrTimothyAldenDavis/GraphBLAS)

This is a base package that exposes only the low level CFFI API
bindings and symbols.  This package is shared by the syntax bindings
[pygraphblas](https://github.com/Graphegon/pygraphblas) and
[python-graphblas](https://github.com/python-graphblas/python-graphblas).


## Installation from pre-built wheels
Pre-built wheels for common platforms are available from PyPI and conda. These bundle a compiled copy of SuiteSparse:GraphBLAS.

```bash
pip install suitesparse-graphblas
```

or

```bash
conda install -c conda-forge python-suitesparse-graphblas
```

### In the browser (Pyodide)
PyPI also has WebAssembly wheels for [Pyodide](https://pyodide.org) 314 (Python 3.14) and
0.29.4 or later (Python 3.13), the `pyemscripten` platform of
[PEP 783](https://peps.python.org/pep-0783/). So, in Pyodide or JupyterLite
(jupyterlite-pyodide-kernel 0.8 uses Pyodide 314):

```python
import micropip
await micropip.install("suitesparse-graphblas")
```

Older versions of Pyodide don't recognize these wheels: micropip says it can't find a pure
Python 3 wheel. GraphBLAS runs single-threaded in Pyodide, which has no threads (so no OpenMP).

## Installation from source
If you wish to link against your own copy of SuiteSparse:GraphBLAS you may build from source.

Specify the location of your SuiteSparse:GraphBLAS installation in the `GraphBLAS_ROOT` environment variable then use the standard pip build from source mechanism. This location must contain `include/GraphBLAS.h` and `lib/`.

```bash
export GraphBLAS_ROOT="/path/to/graphblas"
pip install suitesparse-graphblas-*.tar.gz
```
You may also have to appropriately set `LD_LIBRARY_PATH` to find `libgraphblas` at runtime.

For example, to use Homebrew's SuiteSparse:GraphBLAS on macOS, with the sdist from PyPI, and with all dependencies using wheels:
```bash
GraphBLAS_ROOT="$(brew --prefix suitesparse)" pip install --no-binary suitesparse-graphblas suitesparse-graphblas
```
