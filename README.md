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


## Installation via pre-built wheels
Pre-built wheels for common platforms are available via PyPI and conda. These bundle a compiled copy of SuiteSparse:GraphBLAS.

```bash
pip install suitesparse-graphblas
```

or

```bash
conda install -c conda-forge python-suitesparse-graphblas
```

## Installation from source
If you wish to link against your own copy of SuiteSparse:GraphBLAS you may build from source.

Specify the location of your SuiteSparse:GraphBLAS installation in the `GraphBLAS_ROOT` environment variable then use the standard pip build from source mechanism. This location must contain `include/GraphBLAS.h` and `lib/`.

```bash
export GraphBLAS_ROOT="/path/to/graphblas"
pip install suitesparse-graphblas-*.tar.gz
```

For example, to use Homebrew's SuiteSparse:GraphBLAS on macOS, with the sdist from PyPI, and with all dependencies using wheels:
```bash
GraphBLAS_ROOT="$(brew --prefix suitesparse)" pip install --no-binary suitesparse-graphblas suitesparse-graphblas
```
