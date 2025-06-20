#pragma once

#include <cstddef>
#include <span>

namespace volumembo {

/**
 * @brief A 2D span-like class for handling 2D data in a row-major format.
 *
 * This class provides a way to access 2D data stored in a flat array using
 * row and column indices. It is designed to work with C++20's std::span
 * for zero-copy access to data.
 *
 * @tparam T The type of the elements in the 2D span.
 */
template<typename T>
class Span2D
{
public:
  /**
   * @brief Construct a Span2D from a std::span.
   *
   * @param data The 1D span containing the data
   * @param rows The number of rows
   * @param cols The number of columns
   */
  Span2D(std::span<T> data, std::size_t rows, std::size_t cols)
    : data_(data)
    , rows_(rows)
    , cols_(cols)
  {
  }

  /**
   * @brief Construct a Span2D from a raw pointer.
   *
   * @param data_ptr Pointer to the first element of the data
   * @param rows The number of rows
   * @param cols The number of columns
   */
  Span2D(const T* data_ptr, std::size_t rows, std::size_t cols)
    : data_(data_ptr, rows * cols)
    , rows_(rows)
    , cols_(cols)
  {
  }

  /**
   * @brief Access an element at the specified row and column.
   *
   * @param i The row index
   * @param j The column index
   *
   * @return The element at the specified position
   */
  T operator()(std::size_t i, std::size_t j) const
  {
    return data_[i * cols_ + j];
  }

  //! @brief Get the number of rows
  std::size_t rows() const { return rows_; }
  //! @brief Get the number of columns
  std::size_t cols() const { return cols_; }

private:
  std::span<T> data_;
  std::size_t rows_, cols_;
};

} // namespace volumembo
