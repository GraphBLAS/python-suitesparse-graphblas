# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this package is

`suitesparse-graphblas` is a low-level Python CFFI binding around the C library
[SuiteSparse:GraphBLAS](https://github.com/DrTimothyAldenDavis/GraphBLAS). It exposes the
raw `ffi` and `lib` symbols and, on top of them, a small **functional API** in
`suitesparse_graphblas.matrix`, `suitesparse_graphblas.vector`, and
`suitesparse_graphblas.scalar`. These three modules are the **main entry point for users
of this library directly** — they are module-level functions operating on opaque CFFI
handles, e.g. `A = matrix.new(lib.GrB_BOOL, 3, 3); matrix.set_bool(A, True, 2, 2)`. Higher-level
syntax wrappers ([python-graphblas](https://github.com/python-graphblas/python-graphblas)
and [pygraphblas](https://github.com/Graphegon/pygraphblas)) build on top of this same
package for users who want a more Pythonic, OO-style interface.

The currently targeted SuiteSparse:GraphBLAS version is pinned in `GB_VERSION.txt`.

## Building from source

Building requires a working SuiteSparse:GraphBLAS C library on the system. Point at it via
`GraphBLAS_ROOT` (must contain `include/GraphBLAS.h` and `lib/`):

```bash
export GraphBLAS_ROOT="/path/to/graphblas"   # or `$(brew --prefix suitesparse)` on macOS
pip install -e . --no-deps                    # editable dev install
```

If `GraphBLAS_ROOT` is unset, the build falls back to:
- `C:\GraphBLAS` on Windows
- `/usr/local` if `/usr/local/include/suitesparse` exists (the path used by `suitesparse.sh`)
- `sys.prefix` (works for conda-installed `graphblas`)

The CI build uses conda-forge `graphblas=$(cat GB_VERSION.txt)`. To build SuiteSparse from
source instead, run `bash suitesparse.sh refs/tags/$(cat GB_VERSION.txt).0`. That script also
honors `SUITESPARSE_FAST_BUILD` / `SUITESPARSE_FASTEST_BUILD` env vars to disable many type
specializations for much faster local builds.

## Tests, lint, and other commands

Testing requires the compiled CFFI extension and the SuiteSparse:GraphBLAS C library, so
tests should be run inside the Docker container. Build the image once (this compiles
GraphBLAS from source and takes several minutes), then run tests against it:

```bash
# Build the test image (uses the psg stage which includes pytest)
docker build --target psg \
    --build-arg SUITESPARSE=v$(cat GB_VERSION.txt) \
    --build-arg VERSION=99.0.0.0 \
    -t psg-test .

# Run the full test suite (unit tests + all doctests)
docker run --rm -w /tmp psg-test pytest --pyargs suitesparse_graphblas --doctest-modules -v

# Run only the unit tests (no doctests)
docker run --rm -w /tmp psg-test pytest --pyargs suitesparse_graphblas.tests -v

# Run a single test file
docker run --rm -w /tmp psg-test pytest --pyargs suitesparse_graphblas.tests.test_scalar -v

# Run by test name substring
docker run --rm -w /tmp psg-test pytest --pyargs suitesparse_graphblas --doctest-modules -k test_print_jit_config -v

# Run linters/formatters (locally, no C library needed)
pre-commit run --all-files
```

Rebuild the Docker image after making changes — the `ADD . /psg` layer picks up the
current working tree. The SuiteSparse compilation layer is cached so rebuilds are fast.

`conftest.py` calls `suitesparse_graphblas.initialize()` once per session — `GrB_init` may
only be called once per process, so `test_initialize.py` is run as a separate process in CI:

```bash
docker run --rm -w /tmp psg-test python3 -m suitesparse_graphblas.tests.test_initialize
```

Coverage runs in CI use `CYTHON_COVERAGE=true` so the Cython `utils.pyx` extension is
recompiled with line tracing.

## Architecture: how the binding is generated and assembled

There are two layers of generated code, and understanding them is essential before touching
anything related to types, defines, or the FFI surface.

### 1. Header generation — `suitesparse_graphblas/create_headers.py`

This script regenerates **`suitesparse_graphblas.h`**, **`suitesparse_graphblas_no_complex.h`**,
and **`source.c`** from an upstream `GraphBLAS.h`. It:

- Copies `GraphBLAS.h` from the install, runs the C preprocessor (using pycparser's
  `fake_libc_include`), and parses the result with pycparser.
- Emits a cleaned-up header that cffi can `cdef()`, plus a complex-free variant for
  platforms (notably MSVC) where `_Complex` types don't work.
- Manually tracks `DEFINES`, `CHAR_DEFINES`, `IGNORE_DEFINES`, and `DEPRECATED` sets.
  **When updating to a new SuiteSparse:GraphBLAS version, these lists are the things most
  likely to need editing.** New macros, new deprecations, or removed symbols all flow
  through here.
- CI runs this script and `git diff --exit-code` to fail the build if the committed headers
  drift from upstream. Re-running it locally and committing the result is the standard fix.

### 2. CFFI compilation — `build_graphblas_cffi.py`

`ffibuilder` calls `set_source()` with `source.c` and `cdef()` with `suitesparse_graphblas.h`
to produce the compiled extension `suitesparse_graphblas._graphblas` (which exposes `ffi`
and `lib`). On Windows it instead emits a `_graphblas.c` file and runs a textual patch
(`float _Complex` → `_Fcomplex`, `double _Complex` → `_Dcomplex`, `-DGxB_HAVE_COMPLEX_MSVC`)
because cffi cannot represent MSVC's complex types — see `get_extension()` for the patching
logic. `setup.py` chooses between the cffi-driven and Extension-driven paths based on
`build_graphblas_cffi.is_win`.

`setup.py` also cythonizes any `*.pyx` under `suitesparse_graphblas/` (currently
`utils.pyx`). When Cython is unavailable, it falls back to checked-in `*.c` files; the
build will refuse to proceed if any are missing.

### 3. Python layer (`suitesparse_graphblas/__init__.py` and friends)

The package re-exports `ffi`/`lib` from the compiled extension and adds only thin helpers:

- `initialize(blocking=False, memory_manager="numpy")` — must be called exactly once before
  any GraphBLAS calls. The `numpy` memory manager routes allocation through
  `PyDataMem_NEW`/`FREE` (defined in `utils.pyx::call_gxb_init`) so buffers can be claimed
  zero-copy by NumPy and tracked by `tracemalloc`.
- `check_status(obj, info)` — central error handler. Maps `GrB_Info` codes to exception
  classes in `exceptions.py` and pulls the human-readable message via the type-specific
  `*_error()` function, looked up by cdata cname in `_error_func_lookup`.
- `vararg(val)` — a workaround for variadic GraphBLAS calls on `osx-arm64` and `ppc64le`
  where ARM64 calling conventions force variadic args onto the stack. Prefer the
  non-variadic typed variants (e.g. `GxB_Matrix_Option_get_INT32`) when they exist.
- `libget(name)` — fallback that retries a `GrB_*` lookup as `GxB_*` when SuiteSparse moves
  a symbol between standard and extension namespaces.
- `burble` — context manager / global toggle for `GxB_BURBLE` diagnostic output.
- `matrix.py`, `vector.py`, `scalar.py` — **the functional API** of the package, and the
  primary user-facing surface for code that uses `suitesparse-graphblas` directly. Each
  module follows the same convention: `<module>.new(...)` returns an `ffi.gc`-managed
  cdata handle (`GrB_Matrix*`, `GrB_Vector*`, or `GxB_Scalar*`) and every other function
  takes that handle as its first argument and routes errors through `check_status`. The
  design is deliberately functional rather than class-based so the same handles can be
  passed through higher-level wrappers without object-identity friction. When adding
  features to this package, this is the layer where new user-facing helpers belong.
- `io/serialize.py`, `io/binary.py` — supporting I/O helpers (compressed serialize /
  deserialize, binary format read/write). `matrix.py` and `vector.py` already re-export
  `serialize` / `deserialize` from `io/serialize.py` so callers can reach them as
  `matrix.serialize(A)` etc.

### 4. `utils.pyx` and free-threading

The Cython module exists primarily to (a) call `GxB_init` with NumPy's allocators by
casting the cffi function pointer through `uintptr_t` into a real C function pointer, and
(b) wrap NumPy buffers around GraphBLAS-allocated memory with `claim_buffer` /
`unclaim_buffer`, transferring ownership of the underlying allocation to NumPy.

The file is marked `freethreading_compatible=True`. The package does nothing special for
free-threading itself — correctness depends on SuiteSparse:GraphBLAS being thread-safe,
which it is required to be.

## Style and tooling

- Black, isort, flake8 (config in `.flake8`, line length 100, double quotes), pyupgrade
  (`--py311-plus`), autoflake, shellcheck — all wired through `pre-commit`. Run
  `pre-commit run --all-files` before pushing.
- `pre-commit` also blocks direct commits to `main`.
- Python ≥ 3.11. NumPy ≥ 2.0 is required at build time (CFFI extension), ≥ 1.24 at runtime.
- Generated headers (`suitesparse_graphblas.h`, `suitesparse_graphblas_no_complex.h`,
  `source.c`) are checked in and **must be regenerated via `create_headers.py`** rather
  than hand-edited. CI enforces this.
