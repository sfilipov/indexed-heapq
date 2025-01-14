from typing import TypeVar, Generic, Hashable
import heapq

from indexed_heapq.indexed_heapq import SupportsLessThan, Comparator

K = TypeVar("K", bound=Hashable)
P = TypeVar("P", bound=SupportsLessThan)


class NaiveIndexedPriorityQueue(Generic[K, P]):
    """
    Naive (slow) simple indexed priority queue used for testing purposes only.
    Used to check correctness of invariants between operations.
    Most operations take O(n) time rather than O(logn).
    """

    def __init__(self):
        self.pq: list[Comparator[K, P]] = []

    def __len__(self) -> int:
        return len(self.pq)

    def __contains__(self, key: K) -> bool:
        match = [item for item in self.pq if item.key == key]
        return len(match) > 0

    def __getitem__(self, key: K) -> P:
        match = [item for item in self.pq if item.key == key]
        if not match:
            raise KeyError(f"Key {key} not in queue")

        return match[0].priority

    def __setitem__(self, key: K, priority: P):
        match = [item for item in self.pq if item.key == key]
        if match:
            item = match[0]
            item.priority = priority
            heapq.heapify(self.pq)
        else:
            item = Comparator(key, priority)
            heapq.heappush(self.pq, item)

    def __delitem__(self, key: K):
        match = [i for i, item in enumerate(self.pq) if item.key == key]
        if not match:
            raise KeyError(f"Key {key} not in queue")

        self.pq.pop(match[0])
        heapq.heapify(self.pq)

    def peek(self) -> tuple[K, P]:
        item = self.pq[0]
        return item.key, item.priority

    def pop(self) -> tuple[K, P]:
        item = heapq.heappop(self.pq)
        return item.key, item.priority
