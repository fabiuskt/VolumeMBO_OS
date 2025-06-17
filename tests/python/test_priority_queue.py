import numpy as np
import pytest

from volumembo.heap import CustomHeap


def make_min_comparator():
    return lambda a, b: -1 if a < b else (1 if a > b else 0)


def make_max_comparator():
    return lambda a, b: -1 if a > b else (1 if a < b else 0)


@pytest.mark.parametrize(
    "comparator_factory, reverse",
    [
        (make_min_comparator, False),
        (make_max_comparator, True),
    ],
)
def test_custom_heap_push_pop_remove(comparator_factory, reverse):
    values = [5, 3, 8, 1, -5, 6, 2]
    comparator = comparator_factory()
    pq = CustomHeap(comparator)

    assert not pq
    assert len(pq) == 0

    # Fill the heap with values
    for x in values:
        pq.heappush(x)

    assert bool(pq)
    assert len(pq) == len(values)

    # Remove one element
    to_remove = values[len(values) // 2]
    pq.heapremove(to_remove)
    assert len(pq) == len(values) - 1

    sorted_expected = sorted(values)
    sorted_expected.remove(to_remove)
    sorted_expected = sorted(sorted_expected, reverse=reverse)

    for i, expected in enumerate(sorted_expected):
        top = pq.heappeek()
        assert top == expected
        popped = pq.heappop()
        assert popped == expected
        assert len(pq) == len(sorted_expected) - i - 1

    assert not pq
    assert len(pq) == 0
