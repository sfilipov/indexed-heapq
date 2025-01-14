from typing import Hashable, Iterable, Mapping, Protocol, TypeVar, Generic, Optional
from collections.abc import KeysView, ValuesView, ItemsView
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


class IHQKeysView(KeysView):
    def __repr__(self):
        return f"KeysView({list(self)})"


class IHQValuesView(ValuesView):
    def __repr__(self):
        return f"ValuesView({list(self)})"


class IHQItemsView(ItemsView):
    def __repr__(self):
        return f"ItemsView({list(self)})"


class IndexedHeapQueue(Generic[K, P], Mapping[K, P]):
    """
    Indexed implementation of a priority queue using a min heap data structure.
    Maintains a mapping between keys and their priorities while providing efficient
    access to the minimum priority element.

    The queue stores a set of items, each with an associated priority. Items can be
    accessed, modified, or removed by their keys in O(log n) time, while still
    maintaining the min-heap property for efficient minimum priority access.

    Type Parameters:
        K: The type of keys used to identify items. Must be Hashable.
        P: The type of priorities. Must support less-than comparison.
    """

    def __init__(self, map: Optional[SupportsKeysAndGetItem[K, P]] = None, /):
        """
        Initialize a new IndexedHeapQueue, optionally from an existing mapping.

        Args:
            map: An optional mapping-like object. If provided, all key-priority pairs
                from the map will be added to the queue while preserving the
                key iteration order of the original map.

        Time Complexity: O(n) where n is the number of items in the input map.
        Space Complexity: O(n)
        """
        self.pq: list[Comparator[K, P]] = []
        self.pq_index: dict[K, int] = {}

        if not map:
            return
        for key in map.keys():
            self.pq.append(Comparator(key, map[key]))
            # preserve key iteration order of original map
            self.pq_index[key] = -1

        heapq.heapify(self.pq)
        for i, item in enumerate(self.pq):
            self.pq_index[item.key] = i

    def __len__(self) -> int:
        return len(self.pq_index)

    def __contains__(self, key: object, /) -> bool:
        return key in self.pq_index

    def __iter__(self):
        return iter(self.pq_index)

    def keys(self):
        return IHQKeysView(self)

    def values(self):
        return IHQValuesView(self)

    def items(self):
        return IHQItemsView(self)

    def peek(self) -> tuple[K, P]:
        """
        Return the item with the minimum priority without removing it.

        Returns:
            A tuple (key, priority) containing the key and priority of the item
            with minimum priority in the queue.

        Raises:
            KeyError: If the queue is empty.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Example:
            >>> ihq = IndexedHeapQueue({'a': 2, 'b': 1, 'c': 3})
            >>> ihq.peek()
            ('b', 1)
            >>> len(ihq)  # peek doesn't remove the item
            3
        """
        if not self.pq:
            raise KeyError("peek(): queue is empty")
        return self.pq[0].key, self.pq[0].priority

    def __getitem__(self, key: K) -> P:
        """
        Return the priority of the item with the given key.

        Args:
            key: The key whose priority should be returned.

        Returns:
            The priority associated with the given key.

        Raises:
            KeyError: If the key is not present in the queue.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Example:
            >>> ihq = IndexedHeapQueue({'a': 1, 'b': 2})
            >>> ihq['a']
            1
            >>> ihq['c']  # raises KeyError
        """
        if key not in self.pq_index:
            raise KeyError(key)
        return self.pq[self.pq_index[key]].priority

    def __setitem__(self, key: K, priority: P) -> None:
        """
        Set or update the priority for the given key.

        If the key already exists, updates its priority and reorders the heap
        to maintain the min-heap property. If the key doesn't exist, adds a
        new key-priority pair to the queue.

        Args:
            key: The key whose priority should be set or updated.
            priority: The new priority value to associate with the key.

        Time Complexity: O(log n) where n is the number of items in the queue
        Space Complexity: O(1)

        Example:
            >>> ihq = IndexedHeapQueue({'a': 3})
            >>> ihq['b'] = 2  # Add new key-priority pair
            >>> ihq.peek()
            ('b', 2)
            >>> ihq['a'] = 1  # Update existing key's priority
            >>> ihq.peek()
            ('a', 1)
        """

        if key not in self.pq_index:
            item = Comparator(key, priority)
            self.pq.append(item)
            self.pq_index[key] = len(self.pq) - 1
            self._sift_up(len(self.pq) - 1)
        else:
            index = self.pq_index[key]
            item = self.pq[index]
            old_priority = item.priority
            item.priority = priority
            if priority < old_priority:
                self._sift_up(index)
            else:
                self._sink_down(index)

    def __delitem__(self, key: K) -> None:
        """
        Remove the item with the given key from the queue.

        Args:
            key: The key of the item to remove.

        Raises:
            KeyError: If the key is not present in the queue.

        Time Complexity: O(log n) where n is the number of items in the queue
        Space Complexity: O(1)

        Example:
            >>> ihq = IndexedHeapQueue({'a': 1, 'b': 2})
            >>> 'a' in ihq
            True
            >>> del ihq['a']
            >>> 'a' in ihq
            False
            >>> del ihq['c']  # raises KeyError
        """
        if key not in self.pq_index:
            raise KeyError(key)
        index = self.pq_index[key]
        last_item = self.pq.pop()
        if index < len(self.pq):
            self.pq[index] = last_item
            self.pq_index[last_item.key] = index
            self._sift_up(index)
            self._sink_down(index)
        del self.pq_index[key]

    def pop(self) -> tuple[K, P]:
        """
        Remove and return the item with the minimum priority.

        Returns:
            A tuple (key, priority) containing the key and priority of the removed
            item with minimum priority in the queue.

        Raises:
            KeyError: If the queue is empty.

        Time Complexity: O(log n) where n is the number of items in the queue
        Space Complexity: O(1)

        Example:
            >>> ihq = IndexedHeapQueue({'a': 1, 'b': 2, 'c': 3})
            >>> ihq.pop()
            ('a', 1)
            >>> len(ihq)
            2
        """

        if not self.pq:
            raise KeyError("pop(): queue is empty")
        min_item = self.pq[0]
        last_item = self.pq.pop()
        if self.pq:
            self.pq[0] = last_item
            self.pq_index[last_item.key] = 0
            self._sink_down(0)
        del self.pq_index[min_item.key]
        return min_item.key, min_item.priority

    def _sift_up(self, index: int):
        """
        Move the item at the given index up the heap until it is in the correct position.

        Internal method used to maintain the min-heap property after inserting a new
        item or decreasing a priority.

        Args:
            index: The index of the item to sift up.

        Time Complexity: O(log n) where n is the number of items in the queue
        Space Complexity: O(1)
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

        Internal method used to maintain the min-heap property after removing the
        minimum item or increasing a priority.

        Args:
            index: The index of the item to sink down.

        Time Complexity: O(log n) where n is the number of items in the queue
        Space Complexity: O(1)
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
