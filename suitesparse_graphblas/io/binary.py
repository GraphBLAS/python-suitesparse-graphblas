from ctypes.util import find_library
from pathlib import Path

from cffi import FFI

from suitesparse_graphblas import __version__, check_status, ffi, lib, matrix

stdffi = FFI()
stdffi.cdef(
    """
void *malloc(size_t size);
"""
)
stdlib = stdffi.dlopen(find_library("c"))

# When "packing" a matrix the owner of the memory buffer is transfered
# to SuiteSparse, which then becomes responsible for freeing it.  cffi
# wisely does not allow you to do this without declaring and calling
# malloc directly.  When SuiteSparse moves over to a more formal
# memory manager with the cuda work, this will likely change and have
# to be replaceable with a allocator common to numpy, cuda, and here.
# Maybe PyDataMem_NEW?


def readinto_new_buffer(f, typ, size, allocator=stdlib.malloc):
    buff = ffi.cast(typ, allocator(size))
    f.readinto(ffi.buffer(buff, size))
    return buff


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
    sparsity_status = ffinew("int32_t*")

    nrows[0] = matrix.nrows(A)
    ncols[0] = matrix.ncols(A)
    nvals[0] = matrix.nvals(A)
    matrix_type[0] = matrix.type(A)

    check_status(A, lib.GxB_Type_size(typesize, matrix_type[0]))
    typecode[0] = _ss_typecodes[matrix_type[0]]

    format[0] = matrix.format(A)
    hyper_switch[0] = matrix.hyper_switch(A)
    bitmap_switch[0] = matrix.bitmap_switch(A)
    sparsity_status[0] = matrix.sparsity_status(A)
    sparsity_control[0] = matrix.sparsity_control(A)

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

    vars = dict(
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
                A[0], Ap, Ai, Ax, Ap_size[0], Ai_size[0], Ax_size[0], is_iso[0], is_jumbled[0], NULL
            ),
        )

    elif by_row and is_sparse:
        check_status(
            A,
            lib.GxB_Matrix_pack_CSR(
                A[0], Ap, Ai, Ax, Ap_size[0], Ai_size[0], Ax_size[0], is_iso[0], is_jumbled[0], NULL
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
        raise TypeError("This should hever happen")


def binread(filename, opener=Path.open):
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
        is_jumbled = ffi.new("bool*", 0)

        by_row = format[0] == lib.GxB_BY_ROW
        by_col = format[0] == lib.GxB_BY_COL

        is_hyper = sparsity_status[0] == lib.GxB_HYPERSPARSE
        is_sparse = sparsity_status[0] == lib.GxB_SPARSE
        is_bitmap = sparsity_status[0] == lib.GxB_BITMAP
        is_full = sparsity_status[0] == lib.GxB_FULL

        atype = _ss_codetypes[typecode[0]]

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

        if is_hyper:
            Ap_size[0] = (nvec[0] + 1) * Isize
            Ah_size[0] = nvec[0] * Isize
            Ai_size[0] = nvals[0] * Isize
            Ax_size[0] = nvals[0] * typesize[0]

            Ap[0] = readinto_new_buffer(f, "GrB_Index*", Ap_size[0])
            Ah[0] = readinto_new_buffer(f, "GrB_Index*", Ah_size[0])
            Ai[0] = readinto_new_buffer(f, "GrB_Index*", Ai_size[0])
        elif is_sparse:
            Ap_size[0] = (nvec[0] + 1) * Isize
            Ai_size[0] = nvals[0] * Isize
            Ax_size[0] = nvals[0] * typesize[0]
            Ap[0] = readinto_new_buffer(f, "GrB_Index*", Ap_size[0])
            Ai[0] = readinto_new_buffer(f, "GrB_Index*", Ai_size[0])
        elif is_bitmap:
            Ab_size[0] = nrows[0] * ncols[0] * ffi.sizeof("int8_t")
            Ax_size[0] = nrows[0] * ncols[0] * typesize[0]
            Ab[0] = readinto_new_buffer(f, "int8_t*", Ab_size[0])
        elif is_full:
            Ax_size[0] = nrows[0] * ncols[0] * typesize[0]

        Ax[0] = readinto_new_buffer(f, "uint8_t*", typesize[0] if is_iso[0] else Ax_size[0])

        A = matrix.new(atype, nrows[0], ncols[0])

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
            raise TypeError("Unknown format {format[0]}")

        matrix.set_sparsity_control(A, sparsity_control[0])
        matrix.set_hyper_switch(A, hyper_switch[0])
        matrix.set_bitmap_switch(A, bitmap_switch[0])
        return A
