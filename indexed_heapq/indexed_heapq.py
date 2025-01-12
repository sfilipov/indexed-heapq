from typing import Hashable, Iterable, Protocol, TypeVar, Generic, Optional
import heapq


T = TypeVar("T", contravariant=True)


class SupportsLessThan(Protocol[T]):
    def __lt__(self: T, other: T, /) -> bool: ...


K = TypeVar("K", bound=Hashable)
P = TypeVar("P", bound=SupportsLessThan)
Pco = TypeVar("Pco", bound=SupportsLessThan, covariant=True)


class SupportsKeysAndGetItem(Protocol[K, Pco]):
    def keys(self) -> Iterable[K]: ...
    def __getitem__(self, key: K, /) -> Pco: ...


class Comparator(Generic[K, P]):
    def __init__(self, key: K, priority: P):
        self.key = key
        self.priority = priority

    def __lt__(self, other: "Comparator[K, P]", /) -> bool:
        return self.priority < other.priority

    def __repr__(self):
        return f"Comparator({repr(self.key)}, {repr(self.priority)})"


class IndexedHeapQueue(Generic[K, P]):
    """
    Indexed implementation of a min heap.
    The queue stores a set of items, each with an associated priority.
    """

    def __init__(self, map: Optional[SupportsKeysAndGetItem[K, P]] = None, /):
        self.pq: list[Comparator[K, P]] = []
        self.pq_index: dict[K, int] = {}

        if not map:
            return
        for key in map.keys():
            self.pq.append(Comparator(key, map[key]))

        heapq.heapify(self.pq)
        self.pq_index = {item.key: i for i, item in enumerate(self.pq)}

    def __len__(self) -> int:
        return len(self.pq)

    def __contains__(self, key: K) -> bool:
        return key in self.pq_index

    def pop(self) -> tuple[K, P]:
        """
        Remove and return the item with the minimum priority.
        """

        if not self.pq:
            raise IndexError("pop from empty priority queue")
        min_item = self.pq[0]
        last_item = self.pq.pop()
        if self.pq:
            self.pq[0] = last_item
            self.pq_index[last_item.key] = 0
            self._sink_down(0)
        del self.pq_index[min_item.key]
        return min_item.key, min_item.priority

    def peek(self) -> tuple[K, P]:
        """
        Return the item with the minimum priority without removing it.
        """
        if not self.pq:
            raise IndexError("peek from empty priority queue")
        return self.pq[0].key, self.pq[0].priority

    def get(self, key: K) -> P:
        """
        Return the priority of the item with the given key.
        """
        if key not in self.pq_index:
            raise KeyError(f"key {key} not in priority queue")
        return self.pq[self.pq_index[key]].priority

    def insert(self, key: K, priority: P) -> None:
        """
        Insert an item with the given key and priority.
        """
        if key in self.pq_index:
            raise ValueError(f"key {key} already in priority queue")
        item = Comparator(key, priority)
        self.pq.append(item)
        self.pq_index[key] = len(self.pq) - 1
        self._sift_up(len(self.pq) - 1)

    def remove(self, key: K) -> None:
        """
        Remove the item with the given key.
        """
        if key not in self.pq_index:
            raise KeyError(f"key {key} not in priority queue")
        index = self.pq_index[key]
        last_item = self.pq.pop()
        if index < len(self.pq):
            self.pq[index] = last_item
            self.pq_index[last_item.key] = index
            self._sift_up(index)
            self._sink_down(index)
        del self.pq_index[key]

    def update(self, key: K, priority: P) -> None:
        """
        Update the priority of the item with the given key.
        """
        if key not in self.pq_index:
            raise KeyError(f"key {key} not in priority queue")

        index = self.pq_index[key]
        item = self.pq[index]
        old_priority = item.priority
        item.priority = priority
        if priority < old_priority:
            self._sift_up(index)
        else:
            self._sink_down(index)

    def upsert(self, key: K, priority: P) -> None:
        """
        Update the priority of the item with the given key, or insert it if it does not exist.
        """
        if key in self.pq_index:
            self.update(key, priority)
        else:
            self.insert(key, priority)

    def _sift_up(self, index: int):
        """
        Move the item at the given index up the heap until it is in the correct position.
        """
        while index > 0:
            parent_index = (index - 1) // 2
            if self.pq[index] < self.pq[parent_index]:
                self.pq[index], self.pq[parent_index] = (
                    self.pq[parent_index],
                    self.pq[index],
                )
                self.pq_index[self.pq[index].key] = index
                self.pq_index[self.pq[parent_index].key] = parent_index
                index = parent_index
            else:
                break

    def _sink_down(self, index: int):
        """
        Move the item at the given index down the heap until it is in the correct position.
        """
        while True:
            left_idx = 2 * index + 1
            right_idx = 2 * index + 2
            smallest = index
            if left_idx < len(self.pq) and self.pq[left_idx] < self.pq[smallest]:
                smallest = left_idx
            if right_idx < len(self.pq) and self.pq[right_idx] < self.pq[smallest]:
                smallest = right_idx
            if smallest != index:
                self.pq[index], self.pq[smallest] = self.pq[smallest], self.pq[index]
                self.pq_index[self.pq[index].key] = index
                self.pq_index[self.pq[smallest].key] = smallest
                index = smallest
            else:
                break
