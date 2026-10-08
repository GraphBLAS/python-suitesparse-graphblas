from contextlib import ExitStack
from pathlib import Path

import numpy as np

from suitesparse_graphblas import __version__, check_status, ffi, lib
from suitesparse_graphblas.api import matrix
from suitesparse_graphblas.utils import empty, give_buffer

GRB_HEADER_LEN = 512
NULL = ffi.NULL

header_template = """\
SuiteSparse:GraphBLAS matrix
{suitesparse_version} ({user_agent})
nrows:   {nrows}
ncols:   {ncols}
nvec:    {nvec}
nvals:   {nvals}
format:  {format}
size:    {size}
type:    {type}
iso:     {iso}
{comments}
"""

sizeof = ffi.sizeof
ffinew = ffi.new
buff = ffi.buffer
frombuff = ffi.from_buffer
Isize = ffi.sizeof("GrB_Index")

_ss_typecodes = {
    lib.GrB_BOOL: 0,
    lib.GrB_INT8: 1,
    lib.GrB_INT16: 2,
    lib.GrB_INT32: 3,
    lib.GrB_INT64: 4,
    lib.GrB_UINT8: 5,
    lib.GrB_UINT16: 6,
    lib.GrB_UINT32: 7,
    lib.GrB_UINT64: 8,
    lib.GrB_FP32: 9,
    lib.GrB_FP64: 10,
    lib.GxB_FC32: 11,
    lib.GxB_FC64: 12,
}

_ss_typenames = {
    lib.GrB_BOOL: "GrB_BOOL",
    lib.GrB_INT8: "GrB_INT8",
    lib.GrB_INT16: "GrB_INT16",
    lib.GrB_INT32: "GrB_INT32",
    lib.GrB_INT64: "GrB_INT64",
    lib.GrB_UINT8: "GrB_UINT8",
    lib.GrB_UINT16: "GrB_UINT16",
    lib.GrB_UINT32: "GrB_UINT32",
    lib.GrB_UINT64: "GrB_UINT64",
    lib.GrB_FP32: "GrB_FP32",
    lib.GrB_FP64: "GrB_FP64",
    lib.GxB_FC32: "GxB_FC32",
    lib.GxB_FC64: "GxB_FC64",
}

_ss_codetypes = {v: k for k, v in _ss_typecodes.items()}


def binwrite(A, filename, comments=None, opener=Path.open):
    """Write Matrix ``A`` to ``filename`` in SuiteSparse:GraphBLAS's binary format.

    ``opener`` opens the file for writing (e.g. ``gzip.open`` to compress it).  To avoid a
    copy, ``A`` is unpacked while it is written and packed again afterwards (even if writing
    fails), so other threads must not use ``A`` meanwhile.
    """
    if isinstance(filename, str):
        filename = Path(filename)

    check_status(A, lib.GrB_Matrix_wait(A[0], lib.GrB_MATERIALIZE))

    ffinew = ffi.new

    Ap = ffinew("GrB_Index**")
    Ai = ffinew("GrB_Index**")
    Ah = ffinew("GrB_Index**")
    Ax = ffinew("void**")
    Ab = ffinew("int8_t**")

    Ap_size = ffinew("GrB_Index*")
    Ai_size = ffinew("GrB_Index*")
    Ah_size = ffinew("GrB_Index*")
    Ax_size = ffinew("GrB_Index*")
    Ab_size = ffinew("GrB_Index*")

    nvec = ffinew("GrB_Index*")
    nrows = ffinew("GrB_Index*")
    ncols = ffinew("GrB_Index*")
    nvals = ffinew("GrB_Index*")

    typesize = ffi.new("size_t*")
    is_iso = ffinew("bool*")
    is_jumbled = ffinew("bool*")

    impl = ffi.new("uint64_t*", lib.GxB_IMPLEMENTATION)
    format = ffinew("GxB_Format_Value*")
    hyper_switch = ffinew("double*")
    bitmap_switch = ffinew("double*")
    sparsity_control = ffinew("int32_t*")
    sparsity_status = ffinew("int32_t*")

    typecode = ffinew("int32_t*")
    matrix_type = ffi.new("GrB_Type*")

    nrows[0] = matrix.matrix_nrows(A)
    ncols[0] = matrix.matrix_ncols(A)
    nvals[0] = matrix.matrix_nvals(A)
    matrix_type[0] = matrix.matrix_type(A)

    check_status(A, lib.GxB_Type_size(typesize, matrix_type[0]))
    typecode[0] = _ss_typecodes[matrix_type[0]]

    format[0] = matrix.matrix_format(A)
    hyper_switch[0] = matrix.matrix_hyper_switch(A)
    bitmap_switch[0] = matrix.matrix_bitmap_switch(A)
    sparsity_status[0] = matrix.matrix_sparsity_status(A)
    sparsity_control[0] = matrix.matrix_sparsity_control(A)

    by_row = format[0] == lib.GxB_BY_ROW
    by_col = format[0] == lib.GxB_BY_COL

    is_hyper = sparsity_status[0] == lib.GxB_HYPERSPARSE
    is_sparse = sparsity_status[0] == lib.GxB_SPARSE
    is_bitmap = sparsity_status[0] == lib.GxB_BITMAP
    is_full = sparsity_status[0] == lib.GxB_FULL

    if by_col and is_hyper:
        check_status(
            A,
            lib.GxB_Matrix_unpack_HyperCSC(
                A[0],
                Ap,
                Ah,
                Ai,
                Ax,
                Ap_size,
                Ah_size,
                Ai_size,
                Ax_size,
                is_iso,
                nvec,
                is_jumbled,
                NULL,
            ),
        )
        fmt_string = "HCSC"

    elif by_row and is_hyper:
        check_status(
            A,
            lib.GxB_Matrix_unpack_HyperCSR(
                A[0],
                Ap,
                Ah,
                Ai,
                Ax,
                Ap_size,
                Ah_size,
                Ai_size,
                Ax_size,
                is_iso,
                nvec,
                is_jumbled,
                NULL,
            ),
        )
        fmt_string = "HCSR"

    elif by_col and is_sparse:
        check_status(
            A,
            lib.GxB_Matrix_unpack_CSC(
                A[0], Ap, Ai, Ax, Ap_size, Ai_size, Ax_size, is_iso, is_jumbled, NULL
            ),
        )
        nvec[0] = ncols[0]
        fmt_string = "CSC"

    elif by_row and is_sparse:
        check_status(
            A,
            lib.GxB_Matrix_unpack_CSR(
                A[0], Ap, Ai, Ax, Ap_size, Ai_size, Ax_size, is_iso, is_jumbled, NULL
            ),
        )
        nvec[0] = nrows[0]
        fmt_string = "CSR"

    elif by_col and is_bitmap:
        check_status(
            A, lib.GxB_Matrix_unpack_BitmapC(A[0], Ab, Ax, Ab_size, Ax_size, is_iso, nvals, NULL)
        )
        nvec[0] = ncols[0]
        fmt_string = "BITMAPC"

    elif by_row and is_bitmap:
        check_status(
            A, lib.GxB_Matrix_unpack_BitmapR(A[0], Ab, Ax, Ab_size, Ax_size, is_iso, nvals, NULL)
        )
        nvec[0] = nrows[0]
        fmt_string = "BITMAPR"

    elif by_col and is_full:
        check_status(A, lib.GxB_Matrix_unpack_FullC(A[0], Ax, Ax_size, is_iso, NULL))
        nvec[0] = ncols[0]
        fmt_string = "FULLC"

    elif by_row and is_full:
        check_status(A, lib.GxB_Matrix_unpack_FullR(A[0], Ax, Ax_size, is_iso, NULL))
        nvec[0] = nrows[0]
        fmt_string = "FULLR"

    else:  # pragma nocover
        raise TypeError(f"Unknown Matrix format {format[0]}")

    suitesparse_version = (
        f"v{lib.GxB_IMPLEMENTATION_MAJOR}."
        f"{lib.GxB_IMPLEMENTATION_MINOR}."
        f"{lib.GxB_IMPLEMENTATION_SUB}"
    )

    vars = dict(  # noqa: C408
        suitesparse_version=suitesparse_version,
        user_agent="pygraphblas-" + __version__,
        nrows=nrows[0],
        ncols=ncols[0],
        nvals=nvals[0],
        nvec=nvec[0],
        format=fmt_string,
        size=typesize[0],
        type=_ss_typenames[matrix_type[0]],
        iso=int(is_iso[0]),
        comments=comments,
    )
    header_content = header_template.format(**vars)
    header = f"{header_content: <{GRB_HEADER_LEN}}".encode("ascii")

    try:
        with opener(filename, "wb") as f:
            fwrite = f.write
            fwrite(header)
            fwrite(buff(impl, sizeof("uint64_t")))
            fwrite(buff(format, sizeof("GxB_Format_Value")))
            fwrite(buff(sparsity_status, sizeof("int32_t")))
            fwrite(buff(sparsity_control, sizeof("int32_t")))
            fwrite(buff(hyper_switch, sizeof("double")))
            fwrite(buff(bitmap_switch, sizeof("double")))
            fwrite(buff(nrows, Isize))
            fwrite(buff(ncols, Isize))
            fwrite(buff(nvec, Isize))
            fwrite(buff(nvals, Isize))
            fwrite(buff(typecode, sizeof("int32_t")))
            fwrite(buff(typesize, sizeof("size_t")))
            fwrite(buff(is_iso, sizeof("bool")))

            Tsize = typesize[0]
            iso = is_iso[0]

            if is_hyper:
                fwrite(buff(Ap[0], (nvec[0] + 1) * Isize))
                fwrite(buff(Ah[0], nvec[0] * Isize))
                fwrite(buff(Ai[0], nvals[0] * Isize))
                Axsize = Tsize if iso else nvals[0] * Tsize
            elif is_sparse:
                fwrite(buff(Ap[0], (nvec[0] + 1) * Isize))
                fwrite(buff(Ai[0], nvals[0] * Isize))
                Axsize = Tsize if iso else nvals[0] * Tsize
            elif is_bitmap:
                fwrite(buff(Ab[0], nrows[0] * ncols[0] * ffi.sizeof("int8_t")))
                Axsize = Tsize if iso else nrows[0] * ncols[0] * Tsize
            else:
                Axsize = Tsize if iso else nrows[0] * ncols[0] * Tsize

            fwrite(buff(Ax[0], Axsize))
    finally:
        # The matrix was unpacked to write its arrays, so give them back even if that failed
        if by_col and is_hyper:
            check_status(
                A,
                lib.GxB_Matrix_pack_HyperCSC(
                    A[0],
                    Ap,
                    Ah,
                    Ai,
                    Ax,
                    Ap_size[0],
                    Ah_size[0],
                    Ai_size[0],
                    Ax_size[0],
                    is_iso[0],
                    nvec[0],
                    is_jumbled[0],
                    NULL,
                ),
            )

        elif by_row and is_hyper:
            check_status(
                A,
                lib.GxB_Matrix_pack_HyperCSR(
                    A[0],
                    Ap,
                    Ah,
                    Ai,
                    Ax,
                    Ap_size[0],
                    Ah_size[0],
                    Ai_size[0],
                    Ax_size[0],
                    is_iso[0],
                    nvec[0],
                    is_jumbled[0],
                    NULL,
                ),
            )

        elif by_col and is_sparse:
            check_status(
                A,
                lib.GxB_Matrix_pack_CSC(
                    A[0],
                    Ap,
                    Ai,
                    Ax,
                    Ap_size[0],
                    Ai_size[0],
                    Ax_size[0],
                    is_iso[0],
                    is_jumbled[0],
                    NULL,
                ),
            )

        elif by_row and is_sparse:
            check_status(
                A,
                lib.GxB_Matrix_pack_CSR(
                    A[0],
                    Ap,
                    Ai,
                    Ax,
                    Ap_size[0],
                    Ai_size[0],
                    Ax_size[0],
                    is_iso[0],
                    is_jumbled[0],
                    NULL,
                ),
            )

        elif by_col and is_bitmap:
            check_status(
                A,
                lib.GxB_Matrix_pack_BitmapC(
                    A[0], Ab, Ax, Ab_size[0], Ax_size[0], is_iso[0], nvals[0], NULL
                ),
            )

        elif by_row and is_bitmap:
            check_status(
                A,
                lib.GxB_Matrix_pack_BitmapR(
                    A[0], Ab, Ax, Ab_size[0], Ax_size[0], is_iso[0], nvals[0], NULL
                ),
            )

        elif by_col and is_full:
            check_status(A, lib.GxB_Matrix_pack_FullC(A[0], Ax, Ax_size[0], is_iso[0], NULL))

        elif by_row and is_full:
            check_status(A, lib.GxB_Matrix_pack_FullR(A[0], Ax, Ax_size[0], is_iso[0], NULL))
        else:
            raise TypeError("This should never happen")


def binread(filename, opener=Path.open):
    """Read a Matrix from ``filename``, written by ``binwrite``.

    ``opener`` opens the file for reading (e.g. ``gzip.open``).
    """
    if isinstance(filename, str):
        filename = Path(filename)

    with opener(filename, "rb") as f:
        fread = f.read

        fread(GRB_HEADER_LEN)
        impl = frombuff("uint64_t*", fread(sizeof("uint64_t")))

        assert impl[0] == lib.GxB_IMPLEMENTATION

        format = frombuff("GxB_Format_Value*", fread(sizeof("GxB_Format_Value")))
        sparsity_status = frombuff("int32_t*", fread(sizeof("int32_t")))
        sparsity_control = frombuff("int32_t*", fread(sizeof("int32_t")))
        hyper_switch = frombuff("double*", fread(sizeof("double")))
        bitmap_switch = frombuff("double*", fread(sizeof("double")))
        nrows = frombuff("GrB_Index*", fread(Isize))
        ncols = frombuff("GrB_Index*", fread(Isize))
        nvec = frombuff("GrB_Index*", fread(Isize))
        nvals = frombuff("GrB_Index*", fread(Isize))
        typecode = frombuff("int32_t*", fread(sizeof("int32_t")))
        typesize = frombuff("size_t*", fread(sizeof("size_t")))
        is_iso = frombuff("bool*", fread(sizeof("bool")))

        by_row = format[0] == lib.GxB_BY_ROW
        by_col = format[0] == lib.GxB_BY_COL

        is_hyper = sparsity_status[0] == lib.GxB_HYPERSPARSE
        is_sparse = sparsity_status[0] == lib.GxB_SPARSE
        is_bitmap = sparsity_status[0] == lib.GxB_BITMAP
        is_full = sparsity_status[0] == lib.GxB_FULL

        atype = _ss_codetypes[typecode[0]]

        if not (by_row or by_col):
            raise TypeError(f"Unknown format {format[0]}")

        A = matrix.matrix_new(atype, nrows[0], ncols[0])
        # GraphBLAS will own the buffers we read into, and takes them to be from the data
        # arena that new matrices use
        arena = ffinew("int32_t*")
        check_status(A, lib.GrB_Matrix_get_INT32(A[0], arena, lib.GxB_ARENA_DATA))
        arena = arena[0]
        buffers = []  # (array, ctype), in the order the file and `GxB_Matrix_pack_*` have them

        def read(nbytes, ctype):
            array = empty(nbytes, np.uint8, arena=arena)
            if f.readinto(array) != nbytes:
                raise EOFError(f"{filename} is truncated")
            buffers.append((array, ctype))

        if is_hyper:
            pack = lib.GxB_Matrix_pack_HyperCSC if by_col else lib.GxB_Matrix_pack_HyperCSR
            read((nvec[0] + 1) * Isize, "GrB_Index*")  # Ap
            read(nvec[0] * Isize, "GrB_Index*")  # Ah
            read(nvals[0] * Isize, "GrB_Index*")  # Ai
            nx = nvals[0]
            args = [nvec[0], False]  # False: not jumbled
        elif is_sparse:
            pack = lib.GxB_Matrix_pack_CSC if by_col else lib.GxB_Matrix_pack_CSR
            read((nvec[0] + 1) * Isize, "GrB_Index*")  # Ap
            read(nvals[0] * Isize, "GrB_Index*")  # Ai
            nx = nvals[0]
            args = [False]  # not jumbled
        elif is_bitmap:
            pack = lib.GxB_Matrix_pack_BitmapC if by_col else lib.GxB_Matrix_pack_BitmapR
            read(nrows[0] * ncols[0], "int8_t*")  # Ab
            nx = nrows[0] * ncols[0]
            args = [nvals[0]]
        elif is_full:
            pack = lib.GxB_Matrix_pack_FullC if by_col else lib.GxB_Matrix_pack_FullR
            nx = nrows[0] * ncols[0]
            args = []
        else:
            raise TypeError(f"Unknown sparsity status {sparsity_status[0]}")
        # An iso matrix stores a single value, and GraphBLAS must be told its true size
        read(typesize[0] if is_iso[0] else nx * typesize[0], "void*")  # Ax

        with ExitStack() as stack:
            # The buffers become GraphBLAS's only if the pack succeeds
            given = [
                stack.enter_context(give_buffer(array, ctype, copy=False, arena=arena))
                for array, ctype in buffers
            ]
            ptrs = [ptr for ptr, _, _ in given]
            sizes = [nbytes for _, nbytes, _ in given]
            check_status(A, pack(A[0], *ptrs, *sizes, is_iso[0], *args, NULL))

        matrix.matrix_set_sparsity_control(A, sparsity_control[0])
        matrix.matrix_set_hyper_switch(A, hyper_switch[0])
        matrix.matrix_set_bitmap_switch(A, bitmap_switch[0])
        return A
