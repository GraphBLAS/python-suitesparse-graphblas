from setuptools import setup, find_packages

setup(
    name='suitesparse-graphblas',
    version='4.0.3',
    description='SuiteSparse:GraphBLAS Python bindings.',
    packages=find_packages(),
    author='Michel Pelletier, James Kitchen, Erik Welch',
    cffi_modules=["suitesparse/graphblas/build.py:ffibuilder"],
    install_requires=["cffi>=1.0.0"],
    setup_requires=["cffi>=1.0.0", "pytest-runner"],
    tests_require=["pytest"],
)

