from typing import Hashable, Protocol, NamedTuple


class SupportsLessThan[T](Protocol):
    def __lt__(self: T, other: T, /) -> bool: ...


class Item[K: Hashable, P: SupportsLessThan](NamedTuple):
    priority: P
    key: K


class IndexedHeapQueue[K: Hashable, P: SupportsLessThan]:
    """
    Indexed implementation of a min heap.
    The queue stores a set of items, each with an associated priority.
    """

    def __init__(self):
        self.pq: list[Item[K, P]] = []
        self.pq_index: dict[K, int] = {}

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
        item = Item(priority, key)
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

        new_item = Item(priority, key)
        index = self.pq_index[key]
        old_item = self.pq[index]
        self.pq[index] = new_item
        self.pq_index[key] = index
        if new_item.priority < old_item.priority:
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
            if self.pq[index].priority < self.pq[parent_index].priority:
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
            if (
                left_idx < len(self.pq)
                and self.pq[left_idx].priority < self.pq[smallest].priority
            ):
                smallest = left_idx
            if (
                right_idx < len(self.pq)
                and self.pq[right_idx].priority < self.pq[smallest].priority
            ):
                smallest = right_idx
            if smallest != index:
                self.pq[index], self.pq[smallest] = self.pq[smallest], self.pq[index]
                self.pq_index[self.pq[index].key] = index
                self.pq_index[self.pq[smallest].key] = smallest
                index = smallest
            else:
                break
