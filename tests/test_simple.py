import pytest
from indexed_heapq.indexed_heapq import IndexedHeapQueue


def test_ipq_initialization():
    ipq = IndexedHeapQueue()
    assert len(ipq) == 0


def test_ipq_data_types():
    ipq = IndexedHeapQueue({"a": 0, 0.5: 0.5, 10: -10})
    assert ipq["a"] == 0
    assert ipq[0.5] == 0.5
    assert ipq[10] == -10
    assert ipq.peek() == (10, -10)


def test_ipq_empty_peek_raises():
    ipq = IndexedHeapQueue()
    with pytest.raises(IndexError):
        ipq.peek()


def test_ipq_empty_pop_raises():
    ipq = IndexedHeapQueue()
    with pytest.raises(IndexError):
        ipq.pop()


def test_ipq_empty_get_raises():
    ipq = IndexedHeapQueue()
    with pytest.raises(KeyError):
        ipq[0]


def test_ipq_empty_del_raises():
    ipq = IndexedHeapQueue()
    with pytest.raises(KeyError):
        del ipq[0]


def test_ipq_empty_contains_returns_false():
    ipq = IndexedHeapQueue[int, int]()
    assert 0 not in ipq
