from collections import defaultdict

from hypothesis import given, note, settings, target, strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    rule,
    precondition,
    invariant,
    initialize,
)

from indexed_heapq import IndexedHeapQueue
from .naive import NaiveIndexedPriorityQueue


@st.composite
def dict_of_items(
    draw: st.DrawFn,
    min_keys=0,
    max_keys=100,
    min_priorities=0,
    max_priorities=100,
    ps_per_key=1,
) -> dict[int, list[int]]:
    num_priorities = draw(
        st.integers(min_value=min_priorities, max_value=max_priorities)
    )
    return draw(
        st.dictionaries(
            keys=st.integers(min_value=0, max_value=max_keys),
            values=st.lists(
                st.integers(min_value=0, max_value=num_priorities),
                min_size=ps_per_key,
                max_size=ps_per_key,
            ),
            min_size=min_keys,
            max_size=max_keys,
        )
    )


@st.composite
def dict_of_single(
    draw: st.DrawFn, min_keys=0, max_keys=100, min_priorities=0, max_priorities=100
) -> dict[int, int]:
    d = draw(
        dict_of_items(
            min_keys=min_keys,
            max_keys=max_keys,
            min_priorities=min_priorities,
            max_priorities=max_priorities,
        )
    )
    return {k: ps[0] for k, ps in d.items()}


def sampled_from_set(set):
    return st.sampled_from(sorted(set, key=lambda x: (type(x) is str, x)))


@given(dict_of_single())
def test_ipq_initialize_from_map(d: dict[int, int]):
    ipq = IndexedHeapQueue(d)
    for key, ps in d.items():
        assert key in ipq
        assert ipq[key] == ps
    assert_heap_property(ipq)
    if d:
        min_priority = min(d.values())
        assert ipq.peek()[1] == min_priority
        assert list(d) == list(ipq)
        assert list(d.keys()) == list(ipq.keys())
        assert list(d.values()) == list(ipq.values())
        assert list(d.items()) == list(ipq.items())


@given(dict_of_single())
def test_ipq_many_gets(d: dict[int, int]):
    ipq = IndexedHeapQueue[int, int]()
    for key, priority in d.items():
        ipq[key] = priority
        assert key in ipq
        assert ipq[key] == priority
    assert list(d) == list(ipq)
    assert list(d.keys()) == list(ipq.keys())
    assert list(d.values()) == list(ipq.values())
    assert list(d.items()) == list(ipq.items())
    assert_heap_property(ipq)


@given(dict_of_single())
def test_ipq_many_insert_pops(d: dict[int, int]):
    ipq = IndexedHeapQueue[int, int]()
    for key, priority in d.items():
        ipq[key] = priority

    assert_ipq_contains_exactly(ipq, d)


@given(dict_of_items(ps_per_key=2))
def test_ipq_many_update_pops(d: dict[int, list[int]]):
    ipq = IndexedHeapQueue[int, int]()
    for key, (insert, update) in d.items():
        ipq[key] = insert
        ipq[key] = update

    d_upd = {key: priority for key, (_, priority) in d.items()}
    assert_ipq_contains_exactly(ipq, d_upd)


@given(dict_of_items(ps_per_key=2))
def test_ipq_many_init_update_pops(d: dict[int, list[int]]):
    d_init = {key: insert for key, (insert, _) in d.items()}
    ipq = IndexedHeapQueue(d_init)
    for key, (_, update) in d.items():
        ipq[key] = update

    d_upd = {key: update for key, (_, update) in d.items()}
    assert_ipq_contains_exactly(ipq, d_upd)


@given(dict_of_single())
def test_ipq_many_dels(d: dict[int, int]):
    ipq = IndexedHeapQueue[int, int]()
    for key, priority in d.items():
        ipq[key] = priority

    for key in d.keys():
        assert key in ipq
        del ipq[key]
        assert key not in ipq
        assert_heap_property(ipq)


def test_ipq_stateful():
    IPQComparison.TestCase.settings = settings(stateful_step_count=10)
    IPQComparison.TestCase().runTest()


class IPQComparison(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.new_keys = set()
        self.inserted_keys = set()
        self.priorities = set()
        self.size = 0

    @initialize(d=dict_of_single(min_keys=1, min_priorities=1))
    def init_with_dict(self, d: dict[int, int]):
        self.ipq = IndexedHeapQueue(d)
        self.naive = NaiveIndexedPriorityQueue[int, int]()
        for k, v in d.items():
            self.naive[k] = v
            self.inserted_keys.add(k)

        note(f"{d=}")

        self.priorities.update(d.values())
        self.size = len(d)

        target(len(d))

    @precondition(lambda self: len(self.new_keys) > 0 and len(self.priorities) > 0)
    @rule(
        k=st.runner().flatmap(lambda self: sampled_from_set(self.new_keys)),
        p=st.runner().flatmap(lambda self: sampled_from_set(self.priorities)),
    )
    def insert(self, k, p):
        self.ipq[k] = p
        self.naive[k] = p
        assert self.ipq[k] == self.naive[k]
        self.new_keys.remove(k)
        self.inserted_keys.add(k)
        self.size += 1

    @precondition(lambda self: len(self.inserted_keys) > 0 and len(self.priorities) > 0)
    @rule(
        k=st.runner().flatmap(lambda self: sampled_from_set(self.inserted_keys)),
        p=st.runner().flatmap(lambda self: sampled_from_set(self.priorities)),
    )
    def update(self, k, p):
        assert self.ipq[k] == self.naive[k]
        self.ipq[k] = p
        self.naive[k] = p
        assert self.ipq[k] == self.naive[k]

    @precondition(lambda self: len(self.inserted_keys) > 0)
    @rule()
    def pop(self):
        a = self.ipq.peek()
        assert a == self.ipq.pop()
        del self.naive[a[0]]
        self.inserted_keys.remove(a[0])
        self.new_keys.add(a[0])
        self.size -= 1

    @precondition(lambda self: len(self.inserted_keys) > 0)
    @rule(
        k=st.runner().flatmap(lambda self: sampled_from_set(self.inserted_keys)),
    )
    def delete(self, k):
        assert self.ipq[k] == self.naive[k]
        del self.ipq[k]
        del self.naive[k]
        assert k not in self.ipq
        self.inserted_keys.remove(k)
        self.new_keys.add(k)
        self.size -= 1

    @invariant()
    def assert_len(self):
        assert len(self.ipq) == self.size
        assert len(self.ipq.pq) == self.size
        assert len(self.ipq.pq_index) == self.size

    @invariant()
    def assert_peek(self):
        if self.size > 0:
            assert self.ipq.peek()[1] == self.naive.peek()[1]

    @invariant()
    def assert_gets(self):
        for key in self.inserted_keys:
            assert self.ipq[key] == self.naive[key]

    @invariant()
    def assert_heap_prop(self):
        assert_heap_property(self.ipq)


def assert_ipq_contains_exactly(ipq: IndexedHeapQueue[int, int], d: dict[int, int]):
    assert len(ipq) == len(d)

    # we will check later that the IPQ returns items in the correct prority order
    # but we order only by priority and the keys might be in arbitrary order (for the same priority)
    # so we need to group the keys by priority as we might get any key for the same priority
    priority_to_keys = defaultdict[int, set[int]](set)
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

        # maintain heap property after every pop
        assert_heap_property(ipq)

    assert sum(len(keys) for keys in priority_to_keys.values()) == 0
    assert len(ipq) == 0


def assert_heap_property(ipq: IndexedHeapQueue):
    n = len(ipq.pq)
    last_parent = (n - 2) // 2
    for i in range(last_parent):
        assert not ipq.pq[i] > ipq.pq[2 * i + 1]
        if 2 * i + 2 < n:
            assert not ipq.pq[i] > ipq.pq[2 * i + 1]
