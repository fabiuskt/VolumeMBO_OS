#pragma once

#include <cassert>
#include <concepts>
#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

namespace volumembo {

/**
 * @brief A modifiable priority queue that allows removal of arbitrary elements.
 *
 * This priority queue supports standard heap operations like push and pop,
 * while also allowing efficient removal of arbitrary elements by ID.
 * Internally, it maintains an index map to track the position of each element
 * in the heap array for fast updates.
 *
 * @tparam PIDType The type of the point or element IDs.
 * @tparam Comparator The comparator used to define heap ordering
 *         (default: std::greater<PIDType> for min-heap behavior).
 */
template<typename PIDType, typename Comparator = std::greater<>>
class ModifiablePriorityQueue
{
public:
  // Default constructor when Compare is default-constructible
  ModifiablePriorityQueue()
    requires std::default_initializable<Comparator>
    : compare()
  {
  }

  // Constructor for custom comparator
  explicit ModifiablePriorityQueue(Comparator cmp)
    : compare(std::move(cmp))
  {
  }

  //! Check if the queue is empty
  bool empty() const { return heap.empty(); }

  /**
   * @brief Peek at the top element of the queue, without removing it
   *
   * @return The ID of the top element
   */
  PIDType peek() const
  {
    assert(!heap.empty());
    return heap[0];
  }

  /**
   * @brief Pop the top element from the queue, remove it
   *
   * @return The ID of the popped element
   */
  PIDType pop()
  {
    assert(!heap.empty());

    PIDType top = heap[0];
    PIDType last_id = heap.back();

    index_in_heap.erase(top);

    if (heap.size() == 1) {
      heap.pop_back();
      return top;
    }

    // Move last element to top and restore heap
    heap[0] = last_id;
    index_in_heap[last_id] = 0;
    heap.pop_back();

    sift_down(0);

    return top;
  }

  /**
   * @brief Push an element with the given ID into the queue
   *
   * @param id The ID of the element to push
   */
  void push(PIDType id)
  {
    heap.push_back(id);
    index_in_heap[id] = heap.size() - 1;
    sift_up(heap.size() - 1);
  }

  /**
   * @brief Remove an element with the given ID from the queue
   *
   * @param id The ID of the element to remove
   */
  void remove(PIDType id)
  {
    auto it = index_in_heap.find(id);
    if (it == index_in_heap.end())
      return;

    std::size_t i = it->second;
    PIDType last_id = heap.back();

    if (i != heap.size() - 1) {
      std::swap(heap[i], heap.back());
      index_in_heap[last_id] = i;
    }

    heap.pop_back();
    index_in_heap.erase(id);

    if (i < heap.size()) {
      if (!sift_down(i))
        sift_up(i);
    }
  }

private:
  //! The underlying heap structure
  std::vector<PIDType> heap;

  //! A map to keep track of the index of each element in the heap
  std::unordered_map<PIDType, std::size_t> index_in_heap;

  //! The comparator functor to maintain the heap property
  Comparator compare;

  /**
   * @brief Sift up the element at index i to maintain the heap property
   *
   * @param i The index of the element to sift up
   */
  void sift_up(std::size_t i)
  {
    PIDType item = heap[i];
    std::size_t current = i;

    // Find the correct position for item
    while (current > 0) {
      std::size_t parent = (current - 1) / 2;
      if (compare(heap[parent], item)) {
        heap[current] = heap[parent];
        index_in_heap[heap[current]] = current;
        current = parent;
      } else {
        break;
      }
    }

    heap[current] = item;
    index_in_heap[item] = current;
  }

  /**
   * @brief Sift down the element at index i to maintain the heap property
   *
   * @param i The index of the element to sift down
   *
   * @return true if the element was moved, false otherwise
   */
  bool sift_down(std::size_t i)
  {
    std::size_t n = heap.size();
    std::size_t start = i;
    PIDType item = heap[i];

    while (true) {
      std::size_t left = 2 * i + 1;
      std::size_t right = 2 * i + 2;

      // Choose the better child according to comparator
      std::size_t best = i;

      if (left < n && compare(item, heap[left])) {
        best = left;
      }
      if (right < n && compare(best == i ? item : heap[best], heap[right])) {
        best = right;
      }

      if (best != i) {
        heap[i] = heap[best];
        index_in_heap[heap[i]] = i;
        i = best;
      } else {
        break;
      }
    }

    heap[i] = item;
    index_in_heap[item] = i;

    return i != start;
  }
};

} // namespace volumembo
