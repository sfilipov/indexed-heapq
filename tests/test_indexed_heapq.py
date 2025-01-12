from typing import Union

from collections import defaultdict
import random

import pytest
from hypothesis import assume, given, note, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule, precondition, invariant

from indexed_heapq import IndexedHeapQueue
from .naive import NaiveIndexedPriorityQueue


Key = Union[str, int]

st_key = st.one_of(st.integers(), st.text())
st_priority = st.integers(min_value=-10, max_value=10)


def sampled_from_set(set):
    return st.sampled_from(sorted(set, key=lambda x: (type(x) == str, x)))


def test_ipq_initialization():
    ipq = IndexedHeapQueue()
    assert len(ipq) == 0


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
        ipq.get(0)


def test_ipq_empty_remove_raises():
    ipq = IndexedHeapQueue()
    with pytest.raises(KeyError):
        ipq.remove(0)


def test_ipq_double_insert_raises():
    ipq = IndexedHeapQueue()
    ipq.insert(0, 0)
    with pytest.raises(ValueError):
        ipq.insert(0, 0)


def test_ipq_empty_update_raises():
    ipq = IndexedHeapQueue()
    with pytest.raises(KeyError):
        ipq.update(0, 0)


def test_ipq_empty_contains_returns_false():
    ipq = IndexedHeapQueue[int, int]()
    assert 0 not in ipq


@given(st.dictionaries(st_key, st_priority))
def test_ipq_many_inserts(d: dict[Key, int]):
    ipq = IndexedHeapQueue[Key, int]()
    for key, priority in d.items():
        ipq.insert(key, priority)
    assert len(ipq) == len(d)


@given(st.dictionaries(st_key, st_priority))
def test_ipq_many_gets(d: dict[Key, int]):
    ipq = IndexedHeapQueue[Key, int]()
    for key, priority in d.items():
        ipq.insert(key, priority)
    assert len(ipq) == len(d)
    for key, priority in d.items():
        assert key in ipq
        assert ipq.get(key) == priority
    assert len(ipq) == len(d)


@given(st.dictionaries(st_key, st_priority))
def test_ipq_many_pops(d: dict[Key, int]):
    ipq = IndexedHeapQueue[Key, int]()
    for key, priority in d.items():
        ipq.insert(key, priority)

    assert_ipq_contains_exactly(ipq, d)


@given(st.dictionaries(st_key, st.tuples(st_priority, st_priority)))
def test_ipq_many_inserts_and_updates(d: dict[Key, tuple[int, int]]):
    ipq = IndexedHeapQueue[Key, int]()
    for key, (priority, _) in d.items():
        ipq.insert(key, priority)
    for key, (_, priority) in d.items():
        ipq.update(key, priority)

    d_upd = {key: priority for key, (_, priority) in d.items()}
    assert_ipq_contains_exactly(ipq, d_upd)


@given(st.dictionaries(st_key, st.tuples(st_priority, st_priority)))
def test_ipq_many_upserts(d: dict[Key, tuple[int, int]]):
    ipq = IndexedHeapQueue[Key, int]()
    for key, (priority, _) in d.items():
        ipq.upsert(key, priority)
    for key, (_, priority) in d.items():
        ipq.upsert(key, priority)

    d_upd = {key: priority for key, (_, priority) in d.items()}
    assert_ipq_contains_exactly(ipq, d_upd)


@given(st.dictionaries(st_key, st_priority))
def test_ipq_many_removes(d: dict[Key, int]):
    ipq = IndexedHeapQueue[Key, int]()
    for key, priority in d.items():
        ipq.insert(key, priority)

    for key in d.keys():
        assert key in ipq
        ipq.remove(key)
        assert key not in ipq

    assert len(ipq) == 0


def test_ipq_stateful():
    IPQComparison.TestCase().runTest()


@given(st.dictionaries(st_key, st.lists(st_priority)), st.randoms())
def test_ipq_many_fuzz(d: dict[Key, list[int]], rnd: random.Random):
    actions = ["upsert", "delete"]
    ipq = NaiveIndexedPriorityQueue[Key, int]()
    d = {key: priorities for key, priorities in d.items() if priorities}
    next_action = {key: "insert" for key in d}
    d_cur = dict[Key, int]()
    while d or ipq:
        action = rnd.choice(actions)
        if not ipq:
            action = "upsert"
        elif not d:
            action = "delete"

        if action == "upsert":
            key = rnd.choice(list(d.keys()))
            priority = d[key].pop()
            if not d[key]:
                del d[key]
            if next_action[key] == "insert":
                note(f"insert {key=} {priority=}")
                assert key not in ipq
                ipq.insert(key, priority)
            else:
                note(f"update {key=} {priority=}")
                assert key in ipq
                ipq.update(key, priority)

            assert ipq.get(key) == priority
            next_action[key] = "update"
            d_cur[key] = priority

        else:
            if rnd.choice((True, False)):
                key, priority = ipq.peek()
                note(f"pop {key=} {priority=}")
                assert key in ipq
                assert ipq.get(key) == priority
                assert (key, priority) == ipq.pop()
                assert priority == min(d_cur.values())
                assert priority == d_cur[key]
                assert key not in ipq
            else:
                key, priority = rnd.choice(list(d_cur.items()))
                note(f"remove {key=} {priority=}")
                assert key in ipq
                assert ipq.get(key) == priority
                ipq.remove(key)
                assert key not in ipq
            del d_cur[key]
            next_action[key] = "insert"


def assert_ipq_contains_exactly(ipq: IndexedHeapQueue[Key, int], d: dict[Key, int]):
    assert len(ipq) == len(d)

    # we will check later that the IPQ returns items in the correct prority order
    # but we order only by priority and the keys might be in arbitrary order (for the same priority)
    # so we need to group the keys by priority as we might get any key for the same priority
    priority_to_keys = defaultdict[int, set[Key]](set)
    for key, priority in d.items():
        priority_to_keys[priority].add(key)

    lst = sorted(d.values())
    for priority in lst:
        # item we peek should have the corrent key for the priority
        item = ipq.peek()
        assert item[1] == priority
        assert item[0] in priority_to_keys[priority]

        # pop should match the peek
        assert ipq.pop() == item

        # remove the key from the set of keys for the priority to avoid IPQ bugs that return the same key twice
        priority_to_keys[priority].remove(item[0])

    assert sum(len(keys) for keys in priority_to_keys.values()) == 0
    assert len(ipq) == 0


class IPQComparison(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.ipq = IndexedHeapQueue[Key, int]()
        self.naive = NaiveIndexedPriorityQueue[Key, int]()
        self.new_keys = set()
        self.inserted_keys = set()
        self.priorities = set()
        self.size = 0

    @rule(k=st_key)
    def add_key(self, k):
        assume(k not in self.new_keys and k not in self.inserted_keys)
        self.new_keys.add(k)

    @rule(p=st_priority)
    def add_priority(self, p):
        assume(p not in self.priorities)
        self.priorities.add(p)

    @precondition(lambda self: len(self.new_keys) > 0 and len(self.priorities) > 0)
    @rule(
        k=st.runner().flatmap(lambda self: sampled_from_set(self.new_keys)),
        p=st.runner().flatmap(lambda self: sampled_from_set(self.priorities)),
    )
    def insert(self, k, p):
        self.ipq.insert(k, p)
        self.naive.insert(k, p)
        assert self.ipq.get(k) == self.naive.get(k)
        self.new_keys.remove(k)
        self.inserted_keys.add(k)
        self.size += 1

    @precondition(lambda self: len(self.inserted_keys) > 0 and len(self.priorities) > 0)
    @rule(
        k=st.runner().flatmap(lambda self: sampled_from_set(self.inserted_keys)),
        p=st.runner().flatmap(lambda self: sampled_from_set(self.priorities)),
    )
    def update(self, k, p):
        assert self.ipq.get(k) == self.naive.get(k)
        self.ipq.update(k, p)
        self.naive.update(k, p)
        assert self.ipq.get(k) == self.naive.get(k)

    @precondition(lambda self: len(self.inserted_keys) > 0)
    @rule()
    def pop(self):
        a = self.ipq.peek()
        assert a == self.ipq.pop()
        self.naive.remove(a[0])
        self.inserted_keys.remove(a[0])
        self.new_keys.add(a[0])
        self.size -= 1

    @precondition(lambda self: len(self.inserted_keys) > 0)
    @rule(
        k=st.runner().flatmap(lambda self: sampled_from_set(self.inserted_keys)),
    )
    def remove(self, k):
        assert self.ipq.get(k) == self.naive.get(k)
        self.ipq.remove(k)
        self.naive.remove(k)
        assert k not in self.ipq
        self.inserted_keys.remove(k)
        self.new_keys.add(k)
        self.size -= 1

    @invariant()
    def match_len(self):
        assert len(self.ipq) == self.size

    @invariant()
    def match_peek(self):
        if len(self.ipq) > 0:
            assert self.ipq.peek()[1] == self.naive.peek()[1]