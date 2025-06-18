from typing import Callable, Iterator


class CustomHeap:
    def __init__(self, comparator: Callable[[int, int], int]) -> None:
        self.heap: list[int] = []
        self.comparator: Callable[[int, int], int] = comparator
        self.index_map: dict[int, int] = {}  # maps pid -> index in heap

    def heappeek(self) -> int | None:
        return self.heap[0] if self.heap else None

    def heappop(self) -> int:
        lastelt = self.heap.pop()
        if not self.heap:
            del self.index_map[lastelt]
            return lastelt
        returnitem = self.heap[0]
        self.heap[0] = lastelt
        self.index_map[lastelt] = 0
        del self.index_map[returnitem]
        self._siftup(0)
        return returnitem

    def heappush(self, pid: int) -> None:
        self.heap.append(pid)
        pos = len(self.heap) - 1
        self.index_map[pid] = pos
        self._siftdown(0, pos)

    def heapremove(self, pid: int) -> None:
        pos = self.index_map.pop(pid, None)
        if pos is None:
            return  # pid not in heap
        last = self.heap.pop()
        if pos == len(self.heap):
            return  # removed the last item
        self.heap[pos] = last
        self.index_map[last] = pos
        # Re-heapify from position
        self._siftup(pos)
        self._siftdown(0, pos)

    def __bool__(self) -> bool:
        return bool(self.heap)

    def __iter__(self) -> Iterator[int]:
        return iter(self.heap)

    def __len__(self) -> int:
        return len(self.heap)

    def _siftdown(self, startpos: int, pos: int) -> None:
        newitem = self.heap[pos]
        # Follow the path to the root, moving parents down until finding a place newitem fits
        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parent = self.heap[parentpos]
            if self.comparator(newitem, parent) < 0:
                self.heap[pos] = parent
                self.index_map[parent] = pos
                pos = parentpos
                continue
            break
        self.heap[pos] = newitem
        self.index_map[newitem] = pos

    def _siftup(self, pos: int) -> None:
        endpos = len(self.heap)
        startpos = pos
        newitem = self.heap[pos]
        # Bubble up the smaller child until hitting a leaf
        childpos = 2 * pos + 1
        while childpos < endpos:
            # Set childpos to index of smaller child
            rightpos = childpos + 1
            if (
                rightpos < endpos
                and self.comparator(self.heap[rightpos], self.heap[childpos]) < 0
            ):
                childpos = rightpos
            # Move the smaller child up.
            child = self.heap[childpos]
            self.heap[pos] = child
            self.index_map[child] = pos
            pos = childpos
            childpos = 2 * pos + 1
        # The leaf at pos is empty now. Put newitem there, and bubble it up to its final resting place (by sifting its parents down).
        self.heap[pos] = newitem
        self.index_map[newitem] = pos
        self._siftdown(startpos, pos)
