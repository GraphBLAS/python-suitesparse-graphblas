import pytest

from suitesparse_graphblas import burble


@pytest.mark.thread_unsafe(reason="burble is a global option")
def test_is_enabled():
    assert burble.is_enabled is False
    burble.enable()
    try:
        assert burble.is_enabled is True
        assert repr(burble) == "<burble is_enabled=True>"
    finally:
        burble.disable()
    assert burble.is_enabled is False
    assert repr(burble) == "<burble is_enabled=False>"
