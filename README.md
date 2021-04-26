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
[grblas](https://github.com/metagraph-dev/grblas).
