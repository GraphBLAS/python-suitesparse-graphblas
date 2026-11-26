import random
import runpy

import pytest

import suitesparse_graphblas
from suitesparse_graphblas import lib, matrix, vector
from suitesparse_graphblas.api.examples import bfs as bfs_module
from suitesparse_graphblas.api.examples.bfs import bfs


def _reference_bfs(n, edges, src):
    """BFS levels, and parents chosen as bfs() does: the lowest-numbered one."""
    out = [[] for _ in range(n)]
    for i, j in edges:
        out[i].append(j)
    level = {src: 0}
    parent = {src: src}
    frontier = [src]
    while frontier:
        reached = {}
        for i in frontier:
            for j in out[i]:
                if j not in level:
                    reached[j] = min(reached.get(j, i), i)
        for j, p in reached.items():
            level[j] = level[p] + 1
            parent[j] = p
        frontier = list(reached)
    return level, parent


def _to_dict(v):
    indices, values = vector.vector_extract_tuples_int64(v)
    return {int(i): int(x) for i, x in zip(indices, values)}


def _random_graph(n, nedges, T=lib.GrB_BOOL):
    # Directed, with cycles, self-loops, and nodes that can't be reached from 0
    rng = random.Random(12345)
    edges = {(rng.randrange(n), rng.randrange(n)) for _ in range(nedges)}
    A = matrix.matrix_new(T, n, n)
    for i, j in edges:
        if T == lib.GrB_BOOL:
            matrix.set_bool(A, True, i, j)
        else:
            matrix.set_fp64(A, 0.5, i, j)
    return A, edges


@pytest.mark.parametrize("T", [lib.GrB_BOOL, lib.GrB_FP64])
@pytest.mark.parametrize("src", [0, 45])
def test_bfs_matches_reference(T, src):
    n = 60
    A, edges = _random_graph(n, 90, T)
    expected_level, expected_parent = _reference_bfs(n, edges, src)
    assert 1 < len(expected_level) < n  # the graph is neither trivial nor all reachable

    level, parent = bfs(A, src)
    assert _to_dict(level) == expected_level
    assert _to_dict(parent) == expected_parent

    level, parent = bfs(A, src, compute_parent=False)
    assert parent is None
    assert _to_dict(level) == expected_level

    level, parent = bfs(A, src, compute_level=False)
    assert level is None
    assert _to_dict(parent) == expected_parent


def test_bfs_lowest_parent():
    # 3 can be reached from 1 or 2; MIN_FIRST picks 1
    A = matrix.matrix_new(lib.GrB_BOOL, 4, 4)
    for i, j in [(0, 2), (0, 1), (2, 3), (1, 3)]:
        matrix.set_bool(A, True, i, j)
    level, parent = bfs(A, 0)
    assert _to_dict(level) == {0: 0, 1: 1, 2: 1, 3: 2}
    assert _to_dict(parent) == {0: 0, 1: 0, 2: 0, 3: 1}


def test_bfs_isolated_source():
    A = matrix.matrix_new(lib.GrB_BOOL, 3, 3)
    matrix.set_bool(A, True, 1, 2)
    level, parent = bfs(A, 0)
    assert _to_dict(level) == {0: 0}
    assert _to_dict(parent) == {0: 0}
    assert vector.vector_size(level) == vector.vector_size(parent) == 3


def test_bfs_needs_an_output():
    A = matrix.matrix_new(lib.GrB_BOOL, 3, 3)
    with pytest.raises(ValueError, match="at least one"):
        bfs(A, 0, compute_level=False, compute_parent=False)


@pytest.mark.thread_unsafe(reason="patches suitesparse_graphblas.initialize and captures stdout")
def test_bfs_script(monkeypatch, capsys):
    # The script initializes GraphBLAS, which this session already did
    monkeypatch.setattr(suitesparse_graphblas, "initialize", lambda: None)
    runpy.run_path(bfs_module.__file__, run_name="__main__")
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "BFS from node 0:"
    assert lines[1].split() == ["node", "level", "parent"]
    rows = [tuple(int(x) for x in line.split()) for line in lines[2:]]
    #   0 --- 1 --- 2
    #   |     |
    #   3     4 --- 5
    #         |
    #         6
    assert rows == [
        (0, 0, 0),
        (1, 1, 0),
        (2, 2, 1),
        (3, 1, 0),
        (4, 2, 1),
        (5, 3, 4),
        (6, 3, 4),
    ]
