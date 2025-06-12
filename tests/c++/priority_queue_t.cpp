#include "catch2/catch_test_macros.hpp"
#include "volumembo/priority_queue.hpp"

TEST_CASE("ModifiablePriorityQueue as min-heap")
{

  std::vector<int> list = { 5, 3, 8, 1, -5, 6, 2 };

  SECTION("Min-Heap")
  {
    volumembo::ModifiablePriorityQueue<int> pq;

    REQUIRE(pq.empty());

    for (int x : list)
      pq.push(x);

    REQUIRE_FALSE(pq.empty());

    // Remove a value from the heap
    int index_to_remove = list.size() / 2;
    pq.remove(index_to_remove);

    // Remove the value from the reference list
    std::vector<int> sorted = list;
    sorted.erase(std::remove(sorted.begin(), sorted.end(), index_to_remove),
                 sorted.end());
    std::sort(sorted.begin(), sorted.end());

    // Check that pop returns elements in sorted order
    for (int expected : sorted) {
      REQUIRE_FALSE(pq.empty());
      REQUIRE(pq.peek() == expected);
      pq.pop();
    }

    REQUIRE(pq.empty());
  }
  SECTION("Max-Heap")
  {
    volumembo::ModifiablePriorityQueue<int, std::less<>> pq;

    REQUIRE(pq.empty());

    for (int x : list)
      pq.push(x);

    REQUIRE_FALSE(pq.empty());

    // Remove a value from the heap
    int index_to_remove = list.size() / 2;
    pq.remove(index_to_remove);

    // Remove the value from the reference list
    std::vector<int> sorted = list;
    sorted.erase(std::remove(sorted.begin(), sorted.end(), index_to_remove),
                 sorted.end());
    std::sort(sorted.begin(), sorted.end(), std::greater<>());

    // Check that pop returns elements in sorted order
    for (int expected : sorted) {
      REQUIRE_FALSE(pq.empty());
      REQUIRE(pq.peek() == expected);
      pq.pop();
    }

    REQUIRE(pq.empty());
  }
}
