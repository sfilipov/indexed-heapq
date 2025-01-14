# Python Indexed Heap Queue

A Python implementation of an indexed priority queue using a min heap data structure. This data structure maintains a mapping between keys and their priorities while providing efficient O(log n) access to the minimum priority element.

## Key Features

- Fast O(log n) operations for insertion, deletion, and priority updates
- Direct key-based access to priorities in O(1) time
- Type-safe implementation with full generic type support
- Dictionary-like interface implementing the Python Mapping protocol
- Memory efficient with O(n) space complexity

## Requirements

- Python 3.9+
- No external dependencies

## Installation

Install using pip:

```bash
pip install indexed-heapq
```

## Basic Usage

```python
from indexed_heapq import IndexedHeapQueue

# Create an empty queue
queue = IndexedHeapQueue()

# Add some items
queue['task1'] = 3
queue['task2'] = 1
queue['task3'] = 2

# Get the highest priority item (lowest number)
next_task, priority = queue.peek()  # ('task2', 1)

# Remove and return the highest priority item
next_task, priority = queue.pop()   # ('task2', 1)

# Update a priority
queue['task1'] = 0  # Move task1 to the front

# Access a priority directly
priority = queue['task3']  # 2

# Remove an item
del queue['task3']

# Check if an item exists
if 'task1' in queue:
    print("Task1 is in the queue")
```

## Performance

| Operation     | Time Complexity |
|--------------|-----------------|
| getitem      | O(1)           |
| peek()       | O(1)           |
| pop()        | O(log n)       |
| insert       | O(log n)       |
| update       | O(log n)       |
| delete       | O(log n)       |

## Type Hints

The class is fully generic and type-safe:

```python
from indexed_heap_queue import IndexedHeapQueue

# With explicit type hints
queue: IndexedHeapQueue[str, float] = IndexedHeapQueue()
queue['task1'] = 1.5

# Or let type inference work
queue = IndexedHeapQueue({ 'task1': 1.5 })
queue['task2'] = 1
```

## Common Use Cases

- Task schedulers with priority management
- Graph algorithms (Eager Dijkstra's shortest path, Eager Prim's MST)
- Event scheduling systems
- Any application requiring a priority queue with key-based access

## License

This project is licensed under the MIT License - see the LICENSE file for details.
