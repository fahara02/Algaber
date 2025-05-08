#ifndef CUSTOM_ITERATORS_HPP
#define CUSTOM_ITERATORS_HPP

#include <iterator>
#include <numeric>

#include "types.hpp"

namespace algaber {

template <Arithmetic T, typename StoragePolicy>
class MatrixView;
// ==================== Iterator System ====================
// Tag enum for different traversal patterns
enum class IteratorPattern {
  RowMajor,  // Row by row (default)
  ColMajor,  // Column by column
  ZigZag,    // Alternate direction at end of rows
  Spiral,    // From outside to inside
  BlockWise  // Block by block for cache efficiency
};

// Base iterator interface
template <Arithmetic T, typename StoragePolicy>
class MatrixIteratorBase {
 public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using reference = T&;

  virtual ~MatrixIteratorBase() = default;

  virtual reference operator*() const = 0;
  virtual pointer operator->() const = 0;
  virtual MatrixIteratorBase& operator++() = 0;
  virtual MatrixIteratorBase& operator--() = 0;
  virtual MatrixIteratorBase& operator+=(difference_type n) = 0;
  virtual MatrixIteratorBase& operator-=(difference_type n) = 0;
  virtual bool operator==(const MatrixIteratorBase& other) const = 0;
  virtual bool operator!=(const MatrixIteratorBase& other) const = 0;
  virtual reference operator[](difference_type n) const = 0;
};

// Standard row-major iterator
template <Arithmetic T, typename StoragePolicy>
class RowMajorIterator {
  using MatrixType = Matrix<T, StoragePolicy>;
  const MatrixType* matrix_;
  size_t row_;
  size_t col_;

 public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using reference = T&;

  RowMajorIterator(const MatrixType* matrix, size_t row, size_t col)
      : matrix_(matrix), row_(row), col_(col) {}

  reference operator*() const {
    return const_cast<MatrixType*>(matrix_)->operator()(row_, col_);
  }

  pointer operator->() const {
    return &(const_cast<MatrixType*>(matrix_)->operator()(row_, col_));
  }

  RowMajorIterator& operator++() {
    if (++col_ >= matrix_->cols()) {
      col_ = 0;
      ++row_;
    }
    return *this;
  }

  RowMajorIterator operator++(int) {
    auto tmp = *this;
    ++*this;
    return tmp;
  }

  RowMajorIterator& operator--() {
    if (col_ == 0) {
      col_ = matrix_->cols() - 1;
      --row_;
    } else {
      --col_;
    }
    return *this;
  }

  RowMajorIterator operator--(int) {
    auto tmp = *this;
    --*this;
    return tmp;
  }

  RowMajorIterator& operator+=(difference_type n) {
    difference_type pos = row_ * matrix_->cols() + col_ + n;
    row_ = pos / matrix_->cols();
    col_ = pos % matrix_->cols();
    return *this;
  }

  RowMajorIterator operator+(difference_type n) const {
    auto tmp = *this;
    return tmp += n;
  }

  RowMajorIterator& operator-=(difference_type n) { return *this += -n; }

  RowMajorIterator operator-(difference_type n) const {
    auto tmp = *this;
    return tmp -= n;
  }

  difference_type operator-(const RowMajorIterator& other) const {
    return (row_ * matrix_->cols() + col_) -
           (other.row_ * matrix_->cols() + other.col_);
  }

  reference operator[](difference_type n) const { return *(*this + n); }

  bool operator==(const RowMajorIterator& other) const {
    return matrix_ == other.matrix_ && row_ == other.row_ && col_ == other.col_;
  }

  bool operator!=(const RowMajorIterator& other) const {
    return !(*this == other);
  }

  bool operator<(const RowMajorIterator& other) const {
    return row_ < other.row_ || (row_ == other.row_ && col_ < other.col_);
  }

  bool operator<=(const RowMajorIterator& other) const {
    return *this < other || *this == other;
  }

  bool operator>(const RowMajorIterator& other) const {
    return !(*this <= other);
  }

  bool operator>=(const RowMajorIterator& other) const {
    return !(*this < other);
  }
};
// Column-major iterator
template <Arithmetic T, typename StoragePolicy>
class ColMajorIterator {
  using MatrixType = Matrix<T, StoragePolicy>;
  const MatrixType* matrix_;
  size_t row_;
  size_t col_;

 public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using reference = T&;

  ColMajorIterator(const MatrixType* matrix, size_t row, size_t col)
      : matrix_(matrix), row_(row), col_(col) {}

  reference operator*() const {
    return const_cast<MatrixType*>(matrix_)->operator()(row_, col_);
  }

  pointer operator->() const {
    return &(const_cast<MatrixType*>(matrix_)->operator()(row_, col_));
  }

  ColMajorIterator& operator++() {
    if (++row_ >= matrix_->rows()) {
      row_ = 0;
      ++col_;
    }
    return *this;
  }

  ColMajorIterator operator++(int) {
    auto tmp = *this;
    ++*this;
    return tmp;
  }

  ColMajorIterator& operator--() {
    if (row_ == 0) {
      row_ = matrix_->rows() - 1;
      --col_;
    } else {
      --row_;
    }
    return *this;
  }

  ColMajorIterator operator--(int) {
    auto tmp = *this;
    --*this;
    return tmp;
  }

  ColMajorIterator& operator+=(difference_type n) {
    difference_type pos = col_ * matrix_->rows() + row_ + n;
    col_ = pos / matrix_->rows();
    row_ = pos % matrix_->rows();
    return *this;
  }

  ColMajorIterator operator+(difference_type n) const {
    auto tmp = *this;
    return tmp += n;
  }

  ColMajorIterator& operator-=(difference_type n) { return *this += -n; }

  ColMajorIterator operator-(difference_type n) const {
    auto tmp = *this;
    return tmp -= n;
  }

  difference_type operator-(const ColMajorIterator& other) const {
    return (col_ * matrix_->rows() + row_) -
           (other.col_ * matrix_->rows() + other.row_);
  }

  reference operator[](difference_type n) const { return *(*this + n); }

  bool operator==(const ColMajorIterator& other) const {
    return matrix_ == other.matrix_ && row_ == other.row_ && col_ == other.col_;
  }

  bool operator!=(const ColMajorIterator& other) const {
    return !(*this == other);
  }

  bool operator<(const ColMajorIterator& other) const {
    return col_ < other.col_ || (col_ == other.col_ && row_ < other.row_);
  }

  bool operator<=(const ColMajorIterator& other) const {
    return *this < other || *this == other;
  }

  bool operator>(const ColMajorIterator& other) const {
    return !(*this <= other);
  }

  bool operator>=(const ColMajorIterator& other) const {
    return !(*this < other);
  }
};

// ZigZag iterator - traverses rows alternating left-to-right and right-to-left
template <Arithmetic T, typename StoragePolicy>
class ZigZagIterator {
  using MatrixType = Matrix<T, StoragePolicy>;
  const MatrixType* matrix_;
  size_t row_;
  size_t col_;
  bool left_to_right_;

 public:
  using iterator_category = std::bidirectional_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using reference = T&;

  ZigZagIterator(const MatrixType* matrix, size_t row, size_t col)
      : matrix_(matrix), row_(row), col_(col), left_to_right_(row % 2 == 0) {}

  reference operator*() const {
    return const_cast<MatrixType*>(matrix_)->operator()(row_, col_);
  }

  pointer operator->() const {
    return &(const_cast<MatrixType*>(matrix_)->operator()(row_, col_));
  }

  ZigZagIterator& operator++() {
    if (left_to_right_) {
      if (++col_ >= matrix_->cols()) {
        col_ = matrix_->cols() - 1;
        ++row_;
        left_to_right_ = !left_to_right_;
      }
    } else {
      if (col_ == 0) {
        ++row_;
        left_to_right_ = !left_to_right_;
      } else {
        --col_;
      }
    }
    return *this;
  }

  ZigZagIterator operator++(int) {
    auto tmp = *this;
    ++*this;
    return tmp;
  }

  ZigZagIterator& operator--() {
    if (left_to_right_) {
      if (col_ == 0) {
        --row_;
        left_to_right_ = !left_to_right_;
      } else {
        --col_;
      }
    } else {
      if (++col_ >= matrix_->cols()) {
        col_ = 0;
        --row_;
        left_to_right_ = !left_to_right_;
      }
    }
    return *this;
  }

  ZigZagIterator operator--(int) {
    auto tmp = *this;
    --*this;
    return tmp;
  }

  bool operator==(const ZigZagIterator& other) const {
    return matrix_ == other.matrix_ && row_ == other.row_ && col_ == other.col_;
  }

  bool operator!=(const ZigZagIterator& other) const {
    return !(*this == other);
  }
};

// Block-wise iterator for cache-friendly traversal
template <Arithmetic T, typename StoragePolicy>
class BlockIterator {
  using MatrixType = Matrix<T, StoragePolicy>;
  const MatrixType* matrix_;
  size_t row_;
  size_t col_;
  size_t block_size_;
  size_t block_row_;
  size_t block_col_;
  size_t inner_row_;
  size_t inner_col_;

 public:
  using iterator_category = std::bidirectional_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using reference = T&;

  BlockIterator(const MatrixType* matrix, size_t row, size_t col,
                size_t block_size = 16)
      : matrix_(matrix), row_(row), col_(col), block_size_(block_size) {
    block_row_ = row_ / block_size_;
    block_col_ = col_ / block_size_;
    inner_row_ = row_ % block_size_;
    inner_col_ = col_ % block_size_;
  }

  reference operator*() const {
    return const_cast<MatrixType*>(matrix_)->operator()(row_, col_);
  }

  pointer operator->() const {
    return &(const_cast<MatrixType*>(matrix_)->operator()(row_, col_));
  }

  BlockIterator& operator++() {
    // Move to next position within the block
    if (++inner_col_ >= block_size_) {
      inner_col_ = 0;
      if (++inner_row_ >= block_size_) {
        inner_row_ = 0;
        // Move to next block
        if (++block_col_ >= (matrix_->cols() + block_size_ - 1) / block_size_) {
          block_col_ = 0;
          ++block_row_;
        }
      }
    }

    // Calculate actual row and column
    row_ = block_row_ * block_size_ + inner_row_;
    col_ = block_col_ * block_size_ + inner_col_;

    // Handle boundary
    if (row_ >= matrix_->rows()) {
      row_ = matrix_->rows();
      col_ = 0;
    } else if (col_ >= matrix_->cols()) {
      // Move to next row within the block
      ++inner_row_;
      inner_col_ = 0;
      row_ = block_row_ * block_size_ + inner_row_;
      col_ = block_col_ * block_size_ + inner_col_;

      // Check boundaries again
      if (row_ >= matrix_->rows()) {
        row_ = matrix_->rows();
        col_ = 0;
      }
    }

    return *this;
  }

  BlockIterator operator++(int) {
    auto tmp = *this;
    ++*this;
    return tmp;
  }

  BlockIterator& operator--() {
    // Move to previous position within the block
    if (inner_col_ == 0) {
      inner_col_ = block_size_ - 1;
      if (inner_row_ == 0) {
        inner_row_ = block_size_ - 1;
        // Move to previous block
        if (block_col_ == 0) {
          block_col_ = (matrix_->cols() + block_size_ - 1) / block_size_ - 1;
          --block_row_;
        } else {
          --block_col_;
        }
      } else {
        --inner_row_;
      }
    } else {
      --inner_col_;
    }

    // Calculate actual row and column
    row_ = block_row_ * block_size_ + inner_row_;
    col_ = block_col_ * block_size_ + inner_col_;

    // Handle boundary
    if (col_ >= matrix_->cols()) {
      col_ = matrix_->cols() - 1;
    }

    return *this;
  }

  BlockIterator operator--(int) {
    auto tmp = *this;
    --*this;
    return tmp;
  }

  bool operator==(const BlockIterator& other) const {
    return matrix_ == other.matrix_ && row_ == other.row_ && col_ == other.col_;
  }

  bool operator!=(const BlockIterator& other) const {
    return !(*this == other);
  }
};

}  // namespace algaber
#endif