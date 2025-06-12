#pragma once

#include <cassert>
#include <concepts>
#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

namespace volumembo {

template<typename PIDType, typename Comparator = std::greater<PIDType>>
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

  //! Peek at the top element without removing it
  PIDType peek() const
  {
    assert(!heap.empty());
    return heap[0];
  }

  //! Pop the top element and remove it from the queue
  PIDType pop()
  {
    assert(!heap.empty());
    PIDType top = heap[0];
    remove(top); // find another way to do this to circumvent search in indices!
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

    size_t i = it->second;
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
  std::unordered_map<PIDType, size_t> index_in_heap;

  //! The comparator functor to maintain the heap property
  Comparator compare;

  /**
   * @brief Sift up the element at index i to maintain the heap property
   *
   * @param i The index of the element to sift up
   */
  void sift_up(size_t i)
  {
    while (i > 0) {
      size_t parent = (i - 1) / 2;
      if (compare(heap[parent], heap[i])) {
        std::swap(heap[i], heap[parent]);
        index_in_heap[heap[i]] = i;
        index_in_heap[heap[parent]] = parent;
        i = parent;
      } else {
        break;
      }
    }
  }

  /**
   * @brief Sift down the element at index i to maintain the heap property
   *
   * @param i The index of the element to sift down
   *
   * @return true if the element was moved, false otherwise
   */
  bool sift_down(size_t i)
  {
    size_t n = heap.size();
    size_t start = i;

    while (true) {
      size_t left = 2 * i + 1;
      size_t right = 2 * i + 2;
      size_t largest = i;

      if (left < n && compare(heap[largest], heap[left]))
        largest = left;
      if (right < n && compare(heap[largest], heap[right]))
        largest = right;

      if (largest != i) {
        std::swap(heap[i], heap[largest]);
        index_in_heap[heap[i]] = i;
        index_in_heap[heap[largest]] = largest;
        i = largest;
      } else {
        break;
      }
    }

    return i != start;
  }
};

} // namespace volumembo
