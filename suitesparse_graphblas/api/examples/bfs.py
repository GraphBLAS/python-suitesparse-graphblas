"""Breadth-First Search using the GraphBLAS functional API.

This is a Python translation of the vanilla (push-only) BFS algorithm from
[LAGraph](https://github.com/GraphBLAS/LAGraph)
(`LG_BreadthFirstSearch_vanilla_template.c`).

The algorithm uses vector-matrix multiply (`vxm`) with a structural
complement mask to expand the frontier to unvisited nodes at each level.
It can compute BFS levels (distances from source), parent node IDs in
the BFS tree, or both.

Example
-------
Build a small directed graph and run BFS from node 0::

    >>> from suitesparse_graphblas import lib
    >>> from suitesparse_graphblas.api import matrix, vector
    >>> from suitesparse_graphblas.api.examples.bfs import bfs
    >>> #
    >>> # Graph: 0 -> 1 -> 2 -> 3
    >>> A = matrix.matrix_new(lib.GrB_BOOL, 4, 4)
    >>> matrix.set_bool(A, True, 0, 1)
    >>> matrix.set_bool(A, True, 1, 2)
    >>> matrix.set_bool(A, True, 2, 3)
    >>> level, parent = bfs(A, 0)
    >>> [vector.get_int64(level, i) for i in range(4)]
    [0, 1, 2, 3]
    >>> [vector.get_int64(parent, i) for i in range(4)]
    [0, 0, 1, 2]
"""

from suitesparse_graphblas import check_status, ffi, lib
from suitesparse_graphblas.api import matrix, scalar, vector


def bfs(A, src, compute_level=True, compute_parent=True):
    """Breadth-first search on a graph represented as an adjacency matrix.

    Parameters
    ----------
    A : GrB_Matrix*
        Square adjacency matrix (any type). Edge from ``i`` to ``j``
        exists when ``A(i, j)`` is stored.
    src : int
        Source node index (0-based).
    compute_level : bool
        If True, return a level vector where ``level(i)`` is the
        shortest-path distance from *src* to node *i*.
    compute_parent : bool
        If True, return a parent vector where ``parent(i)`` is the
        parent of node *i* in the BFS tree (``parent(src) == src``).

    Returns
    -------
    level : GrB_Vector* or None
        Level vector (if *compute_level* is True).
    parent : GrB_Vector* or None
        Parent vector (if *compute_parent* is True).
    """
    if not compute_level and not compute_parent:
        raise ValueError("at least one of compute_level or compute_parent must be True")

    n = matrix.matrix_nrows(A)

    if compute_parent:
        # Parent mode: frontier is INT64, holds parent IDs.
        # Semiring: MIN_FIRST selects the minimum parent index.
        parent_vec = vector.vector_new(lib.GrB_INT64, n)
        semiring = lib.GrB_MIN_FIRST_SEMIRING_INT64
        frontier = vector.vector_new(lib.GrB_INT64, n)
        vector.set_int64(frontier, src, src)
    else:
        # Level-only mode: frontier is BOOL.
        # Semiring: ANY_PAIR just checks reachability.
        parent_vec = None
        semiring = lib.GxB_ANY_PAIR_BOOL
        frontier = vector.vector_new(lib.GrB_BOOL, n)
        vector.set_bool(frontier, True, src)

    level_vec = vector.vector_new(lib.GrB_INT64, n) if compute_level else None

    # The mask is the set of already-visited nodes (parent or level vector).
    # GrB_DESC_RSC complements the mask so vxm only writes to unvisited nodes.
    mask = parent_vec if compute_parent else level_vec

    # Scalar used to assign the current level number.
    if compute_level:
        level_scalar = scalar.scalar_new(lib.GrB_INT64)

    current_level = 0
    while True:
        # Assign current level to all frontier nodes: level<s(frontier)> = current_level
        if compute_level:
            scalar.set_int64(level_scalar, current_level)
            vector.vector_assign_scalar(
                level_vec, level_scalar, lib.GrB_ALL, n,
                mask=frontier, desc=lib.GrB_DESC_S,
            )

        if compute_parent:
            # Record parent IDs: parent<s(frontier)> = frontier
            vector.vector_assign(
                parent_vec, frontier, lib.GrB_ALL, n,
                mask=frontier, desc=lib.GrB_DESC_S,
            )
            # Convert frontier values to their own indices (ROWINDEX).
            # After this, frontier(i) == i for every stored entry,
            # so the next vxm propagates node i as the parent ID.
            check_status(
                frontier,
                lib.GrB_Vector_apply_IndexOp_INT64(
                    frontier[0], ffi.NULL, ffi.NULL,
                    lib.GrB_ROWINDEX_INT64, frontier[0], 0, ffi.NULL,
                ),
            )

        current_level += 1

        # Expand frontier: frontier<!mask> = frontier * A
        vector.vector_vxm(
            frontier, semiring, frontier, A,
            mask=mask, desc=lib.GrB_DESC_RSC,
        )

        # Stop when the frontier is empty.
        if vector.vector_nvals(frontier) == 0:
            break

    return level_vec, parent_vec


if __name__ == "__main__":
    import suitesparse_graphblas as gb

    gb.initialize()

    # Build a small undirected graph (7 nodes):
    #
    #   0 --- 1 --- 2
    #   |     |
    #   3     4 --- 5
    #         |
    #         6
    #
    n = 7
    A = matrix.matrix_new(lib.GrB_BOOL, n, n)
    edges = [(0, 1), (0, 3), (1, 2), (1, 4), (4, 5), (4, 6)]
    for i, j in edges:
        matrix.set_bool(A, True, i, j)
        matrix.set_bool(A, True, j, i)  # undirected

    src = 0
    level, parent = bfs(A, src)

    print(f"BFS from node {src}:")
    print(f"  {'node':>4}  {'level':>5}  {'parent':>6}")
    for i in range(n):
        lv = vector.get_int64(level, i)
        pa = vector.get_int64(parent, i)
        print(f"  {i:>4}  {lv:>5}  {pa:>6}")
