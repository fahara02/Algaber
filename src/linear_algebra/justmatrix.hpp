#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <omp.h>

#include <vector>
#include "StoragePolicy.hpp"
#include "custom_iterators.hpp"
#include "library_config.hpp"
#include "matrix_error.hpp"

namespace algaber {

enum class VectorType { RowVector, ColumnVector };
enum class Orientation { Row, Column };
// Enumeration for LU pivoting strategies
enum class PivotingStrategy {
  NONE,     // No pivoting (unstable)
  PARTIAL,  // Partial pivoting (row exchanges only)
  FULL      // Full pivoting (row and column exchanges)
};

// Forward declarations
template <Arithmetic T, typename StoragePolicy>
class Matrix;

class PermutationMatrix {
 private:
  std::vector<size_t> perm_;

 public:
  explicit PermutationMatrix(size_t n) : perm_(n) {
    for (size_t i = 0; i < n; ++i)
      perm_[i] = i;
  }

  size_t& operator[](size_t i) { return perm_[i]; }
  const size_t& operator[](size_t i) const { return perm_[i]; }

  size_t size() const { return perm_.size(); }

  // Declaration only - implementation after Matrix class
  Matrix<double, InMemoryStorage<double>> to_matrix() const;
  Matrix<double, InMemoryStorage<double>> inverse() const;
  Matrix<double, InMemoryStorage<double>> transpose() const;
};
// ==================== Matrix Class ====================
template <Arithmetic T, typename StoragePolicy = InMemoryStorage<T>>
class Matrix {
 private:
  std::shared_ptr<StorageInterface<T>> storage_;
  size_t view_row_start_ = 0;
  size_t view_col_start_ = 0;
  size_t view_rows_;
  size_t view_cols_;

  // Private constructor for views
  Matrix(std::shared_ptr<StorageInterface<T>> storage, size_t vr_start,
         size_t vc_start, size_t vr_size, size_t vc_size)
      : storage_(storage),
        view_row_start_(vr_start),
        view_col_start_(vc_start),
        view_rows_(vr_size),
        view_cols_(vc_size) {}

 public:
  // Type aliases
  using value_type = T;
  using iterator = RowMajorIterator<T, StoragePolicy>;
  using const_iterator = RowMajorIterator<const T, StoragePolicy>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using col_iterator = ColMajorIterator<T, StoragePolicy>;
  using zigzag_iterator = ZigZagIterator<T, StoragePolicy>;
  using block_iterator = BlockIterator<T, StoragePolicy>;

  // Declare iterators as friends to allow access to private members
  template <Arithmetic U, typename SP>
  friend class RowMajorIterator;

  template <Arithmetic U, typename SP>
  friend class ColMajorIterator;

  template <Arithmetic U, typename SP>
  friend class ZigZagIterator;

  template <Arithmetic U, typename SP>
  friend class BlockIterator;

  // ===== Constructors =====

  // Default constructor - empty matrix
  Matrix()
      : storage_(std::make_shared<StoragePolicy>(0, 0)),
        view_rows_(0),
        view_cols_(0) {}

  // Size constructor with optional initial value
  Matrix(size_t rows, size_t cols, T init_val = T{})
      : storage_(std::make_shared<StoragePolicy>(rows, cols, init_val)),
        view_rows_(rows),
        view_cols_(cols) {}

  // Generator constructor
  Matrix(size_t rows, size_t cols, std::function<T(size_t, size_t)> generator)
      : storage_(std::make_shared<StoragePolicy>(rows, cols)),
        view_rows_(rows),
        view_cols_(cols) {
    for (size_t i = 0; i < rows; ++i)
      for (size_t j = 0; j < cols; ++j)
        operator()(i, j) = generator(i, j);
  }

  // Initializer list constructor
  Matrix(std::initializer_list<std::initializer_list<T>> init) {
    view_rows_ = init.size();
    view_cols_ = init.begin()->size();
    storage_ = std::make_shared<StoragePolicy>(view_rows_, view_cols_);

    size_t i = 0;
    for (const auto& row : init) {
      size_t j = 0;
      for (const auto& val : row) {
        operator()(i, j) = val;
        ++j;
      }
      ++i;
    }
  }
  //constructor from a initializer list  and given adequate row and column
  Matrix(size_t rows, size_t cols, std::initializer_list<T> initList)
      : storage_(std::make_shared<StoragePolicy>(rows, cols)),
        view_rows_(rows),
        view_cols_(cols) {
    if (initList.size() != rows * cols) {
      throw std::invalid_argument(
          "Initializer list size does not match matrix dimensions");
    }
    auto it = initList.begin();
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        operator()(i, j) = *it++;
      }
    }
  }

  //constructor to create matrix from std::vector
  Matrix(size_t rows, size_t cols, const std::vector<T>& data)
      : storage_(std::make_shared<StoragePolicy>(rows, cols)),
        view_rows_(rows),
        view_cols_(cols) {
    if (data.size() != rows * cols) {
      throw std::invalid_argument("Data size does not match matrix dimensions");
    }
    auto it = data.begin();
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        operator()(i, j) = *it++;
      }
    }
  }

  //constructor for converting one type to other
  template <typename U>
  Matrix(const Matrix<U>& other)
      : storage_(std::make_shared<StoragePolicy>(other.rows(), other.cols())),
        view_rows_(other.rows()),
        view_cols_(other.cols()) {
    for (size_t i = 0; i < view_rows_; ++i) {
      for (size_t j = 0; j < view_cols_; ++j) {
        operator()(i, j) = static_cast<T>(other(i, j));
      }
    }
  }

  // Create from raw data pointer with test both of them again  ownership transfer
  Matrix(T* data, size_t rows, size_t cols, size_t stride = 0)
      : storage_(std::make_shared<StoragePolicy>(data, rows, cols,
                                                 stride ? stride : cols)),
        view_rows_(rows),
        view_cols_(cols) {}

  // Create from raw data pointer without ownership transfer
  static Matrix from_data(T* data, size_t rows, size_t cols,
                          size_t stride = 0) {
    // Use ViewStorage which doesn't take ownership
    return Matrix(std::shared_ptr<StorageInterface<T>>(new ViewStorage<T>(
                      data, rows, cols, stride ? stride : cols)),
                  0, 0, rows, cols);
  }

  // Copy constructor (deep copy)
  Matrix(const Matrix& other)
      : storage_(other.storage_->clone()),
        view_row_start_(other.view_row_start_),
        view_col_start_(other.view_col_start_),
        view_rows_(other.view_rows_),
        view_cols_(other.view_cols_) {}

  // Move constructor
  Matrix(Matrix&& other) noexcept = default;

  Matrix clone() const {
    // Create new matrix with independent storage
    Matrix copy(view_rows_, view_cols_);

    // Deep copy elements from current view to new matrix
    for (size_t i = 0; i < view_rows_; ++i) {
      for (size_t j = 0; j < view_cols_; ++j) {
        copy(i, j) = operator()(i, j);  // Copy element values
      }
    }

    return copy;
  }
  Matrix& operator=(const Matrix& other) {
    if (this != &other) {
      if (view_rows_ != other.view_rows_ || view_cols_ != other.view_cols_)
        throw std::invalid_argument(
            "Matrix dimensions mismatch for assignment");

      for (size_t i = 0; i < view_rows_; ++i) {
        for (size_t j = 0; j < view_cols_; ++j) {
          operator()(i, j) = other(i, j);
        }
      }
    }
    return *this;
  }

  // Move assignment
  Matrix& operator=(Matrix&& other) noexcept = default;
  // ===== Conversion Functions =====

  // Convert to std::vector (flattened)
  std::vector<T> to_vector() const {
    std::vector<T> result;
    result.reserve(view_rows_ * view_cols_);
    for (size_t i = 0; i < view_rows_; ++i) {
      for (size_t j = 0; j < view_cols_; ++j) {
        result.push_back(operator()(i, j));
      }
    }
    return result;
  }

  // Convert to 2D vector
  std::vector<std::vector<T>> to_vector_2d() const {
    std::vector<std::vector<T>> result(view_rows_, std::vector<T>(view_cols_));
    for (size_t i = 0; i < view_rows_; ++i) {
      for (size_t j = 0; j < view_cols_; ++j) {
        result[i][j] = operator()(i, j);
      }
    }
    return result;
  }

  // ===== Size Information =====
  size_t rows() const { return view_rows_; }
  size_t cols() const { return view_cols_; }

  // ===== Utility Functions =====

  size_t size() const noexcept { return view_rows_ * view_cols_; }

  //Booleans
  bool empty() const noexcept { return view_rows_ == 0 || view_cols_ == 0; }
  // Check if matrix is square
  bool is_square() const { return view_rows_ == view_cols_; }

  // Check if matrix is symmetric
  bool is_symmetric() const {
    if (!is_square())
      return false;

    for (size_t i = 0; i < view_rows_; ++i)
      for (size_t j = i + 1; j < view_cols_; ++j)
        if (operator()(i, j) != operator()(j, i))
          return false;

    return true;
  }

  bool is_column() const { return view_cols_ == 1; }

  bool is_row() const { return view_rows_ == 1; }

  bool contains(T value) {
    return std::find(storage_->begin(), storage_->end(), value) !=
           storage_->end();
  }

  // ===== Element Access =====

  T& operator()(size_t row, size_t col) {
    assert(row < view_rows_ && col < view_cols_ &&
           "Matrix index out of bounds");
    return storage_->get(view_row_start_ + row, view_col_start_ + col);
  }

  const T& operator()(size_t row, size_t col) const {
    assert(row < view_rows_ && col < view_cols_ &&
           "Matrix index out of bounds");
    return storage_->get(view_row_start_ + row, view_col_start_ + col);
  }

  T& at(size_t row, size_t col) {
    if (row >= view_rows_ || col >= view_cols_)
      throw std::out_of_range("Matrix index out of range");
    return operator()(row, col);
  }

  const T& at(size_t row, size_t col) const {
    if (row >= view_rows_ || col >= view_cols_)
      throw std::out_of_range("Matrix index out of range");
    return operator()(row, col);
  }

  // Access entire row as std::span (C++20)
  std::span<T> row_span(size_t row) {
    assert(row < view_rows_ && "Row index out of bounds");
    return std::span<T>(&storage_->get(view_row_start_ + row, view_col_start_),
                        view_cols_);
  }

  std::span<const T> row_span(size_t row) const {
    assert(row < view_rows_ && "Row index out of bounds");
    return std::span<const T>(
        &storage_->get(view_row_start_ + row, view_col_start_), view_cols_);
  }
  // Get entire row as new matrix
  Matrix get_row(size_t row) const { return view(row, 0, 1, view_cols_); }

  // Get entire column as new matrix
  Matrix get_column(size_t col) const { return view(0, col, view_rows_, 1); }

  // ===== Block Manipulation Utilities =====
  Matrix get_block(size_t row_start, size_t col_start, size_t rows,
                   size_t cols) const {
    if (row_start + rows > view_rows_ || col_start + cols > view_cols_)
      throw std::invalid_argument("Block dimensions exceed matrix bounds");
    return view(row_start, col_start, rows, cols);
  }

  void set_block(size_t row_start, size_t col_start, const Matrix& block) {
    if (row_start + block.rows() > view_rows_ ||
        col_start + block.cols() > view_cols_)
      throw std::invalid_argument("Block exceeds matrix dimensions");
    auto target = view(row_start, col_start, block.rows(), block.cols());
    target = block;  // Copies data into the target view
  }

  // Set entire row from container
  template <typename Container>
  void set_row(size_t row, const Container& values) {
    assert(row < view_rows_ && "Row index out of bounds");
    assert(std::size(values) >= view_cols_ && "Container size too small");

    auto it = std::begin(values);
    for (size_t j = 0; j < view_cols_; ++j, ++it) {
      operator()(row, j) = *it;
    }
  }

  // Set entire column from container
  template <typename Container>
  void set_column(size_t col, const Container& values) {
    assert(col < view_cols_ && "Column index out of bounds");
    assert(std::size(values) >= view_rows_ && "Container size too small");

    auto it = std::begin(values);
    for (size_t i = 0; i < view_rows_; ++i, ++it) {
      operator()(i, col) = *it;
    }
  }

  // Row-major iterators (default)
  iterator begin() { return iterator(this, 0, 0); }
  const_iterator begin() const { return const_iterator(this, 0, 0); }
  const_iterator cbegin() const { return const_iterator(this, 0, 0); }
  iterator end() { return iterator(this, view_rows_, 0); }
  const_iterator end() const { return const_iterator(this, view_rows_, 0); }
  const_iterator cend() const { return const_iterator(this, view_rows_, 0); }

  // Reverse iterators
  reverse_iterator rbegin() { return reverse_iterator(end()); }
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }
  const_reverse_iterator crbegin() const {
    return const_reverse_iterator(cend());
  }
  reverse_iterator rend() { return reverse_iterator(begin()); }
  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }
  const_reverse_iterator crend() const {
    return const_reverse_iterator(cbegin());
  }

  // Column-major iterators
  col_iterator col_begin() { return col_iterator(this, 0, 0); }
  col_iterator col_end() { return col_iterator(this, 0, view_cols_); }

  // ZigZag iterators
  zigzag_iterator zigzag_begin() { return zigzag_iterator(this, 0, 0); }
  zigzag_iterator zigzag_end() { return zigzag_iterator(this, view_rows_, 0); }

  // Block iterators
  block_iterator block_begin(size_t block_size = 16) {
    return block_iterator(this, 0, 0, block_size);
  }
  block_iterator block_end(size_t block_size = 16) {
    return block_iterator(this, view_rows_, 0, block_size);
  }

  // ===== Views =====

  // Create a view of a submatrix
  Matrix view(size_t row_start, size_t col_start, size_t rows,
              size_t cols) const {
    if (row_start + rows > view_rows_ || col_start + cols > view_cols_)
      throw std::invalid_argument("View exceeds matrix dimensions");

    return Matrix(storage_, view_row_start_ + row_start,
                  view_col_start_ + col_start, rows, cols);
  }

  // Create a view of a single row
  Matrix row(size_t row_idx) const { return view(row_idx, 0, 1, view_cols_); }

  // Create a view of a single column
  Matrix column(size_t col_idx) const {
    return view(0, col_idx, view_rows_, 1);
  }

  // Create a view of a range of rows
  Matrix rows(size_t start_row, size_t num_rows) const {
    return view(start_row, 0, num_rows, view_cols_);
  }

  // Create a view of a range of columns
  Matrix columns(size_t start_col, size_t num_cols) const {
    return view(0, start_col, view_rows_, num_cols);
  }

  // Create a diagonal view
  Matrix diagonal() const {
    size_t min_dim = std::min(view_rows_, view_cols_);
    Matrix result(1, min_dim);
    for (size_t i = 0; i < min_dim; ++i) {
      result(0, i) = operator()(i, i);
    }
    return result;
  }

  // Create a view of the upper triangular part
  Matrix upper_triangular(bool include_diagonal = true) const {
    if (!is_square())
      throw std::invalid_argument("Upper triangular requires square matrix");

    Matrix result = zeros(view_rows_, view_cols_);
    for (size_t i = 0; i < view_rows_; ++i) {
      for (size_t j = (include_diagonal ? i : i + 1); j < view_cols_; ++j) {
        result(i, j) = operator()(i, j);
      }
    }
    return result;
  }

  // Create a view of the lower triangular part
  Matrix lower_triangular(bool include_diagonal = true) const {
    if (!is_square())
      throw std::invalid_argument("Lower triangular requires square matrix");

    Matrix result = zeros(view_rows_, view_cols_);
    for (size_t i = 0; i < view_rows_; ++i) {
      for (size_t j = 0; j <= (include_diagonal ? i : i - 1); ++j) {
        result(i, j) = operator()(i, j);
      }
    }
    return result;
  }

  // ===== Basic Matrix Operations =====

  Matrix transpose() const {
    Matrix result(view_cols_, view_rows_);
    const size_t block_size = BLOCK_SIZE;  // Tune block size as needed

    if constexpr (ALGABER_OPENMP_ENABLED) {
#pragma omp parallel for num_threads(ThreadCount) collapse(2)
      for (size_t i_outer = 0; i_outer < view_rows_; i_outer += block_size) {
        for (size_t j_outer = 0; j_outer < view_cols_; j_outer += block_size) {
          size_t i_end = std::min(i_outer + block_size, view_rows_);
          size_t j_end = std::min(j_outer + block_size, view_cols_);
          // Process block in row-major order for better cache utilization
          for (size_t i = i_outer; i < i_end; ++i) {
            for (size_t j = j_outer; j < j_end; ++j) {
              result(j, i) = operator()(i, j);
            }
          }
        }
      }
    } else {
      for (size_t i_outer = 0; i_outer < view_rows_; i_outer += block_size) {
        size_t i_end = std::min(i_outer + block_size, view_rows_);
        for (size_t j_outer = 0; j_outer < view_cols_; j_outer += block_size) {
          size_t j_end = std::min(j_outer + block_size, view_cols_);
          for (size_t i = i_outer; i < i_end; ++i) {
            for (size_t j = j_outer; j < j_end; ++j) {
              result(j, i) = operator()(i, j);
            }
          }
        }
      }
    }
    return result;
  }

  Matrix operator+(const Matrix& rhs) const {
    if (view_rows_ != rhs.view_rows_ || view_cols_ != rhs.view_cols_)
      throw std::invalid_argument("Matrix dimensions mismatch");

    const size_t block_size = BLOCK_SIZE;
    Matrix result(view_rows_, view_cols_);

    if constexpr (ALGABER_OPENMP_ENABLED) {
#pragma omp parallel for num_threads(ThreadCount) collapse(2)
      for (size_t i_outer = 0; i_outer < view_rows_; i_outer += block_size) {
        for (size_t j_outer = 0; j_outer < view_cols_; j_outer += block_size) {
          size_t i_end = std::min(i_outer + block_size, view_rows_);
          size_t j_end = std::min(j_outer + block_size, view_cols_);
          for (size_t i = i_outer; i < i_end; ++i) {
            for (size_t j = j_outer; j < j_end; ++j) {
              result(i, j) = operator()(i, j) + rhs(i, j);
            }
          }
        }
      }
    } else {
      for (size_t i_outer = 0; i_outer < view_rows_; i_outer += block_size) {
        size_t i_end = std::min(i_outer + block_size, view_rows_);
        for (size_t j_outer = 0; j_outer < view_cols_; j_outer += block_size) {
          size_t j_end = std::min(j_outer + block_size, view_cols_);
          for (size_t i = i_outer; i < i_end; ++i) {
            for (size_t j = j_outer; j < j_end; ++j) {
              result(i, j) = operator()(i, j) + rhs(i, j);
            }
          }
        }
      }
    }
    return result;
  }

  Matrix operator-(const Matrix& rhs) const {
    if (view_rows_ != rhs.view_rows_ || view_cols_ != rhs.view_cols_)
      throw std::invalid_argument("Matrix dimensions mismatch");

    const size_t block_size = BLOCK_SIZE;
    Matrix result(view_rows_, view_cols_);

    if constexpr (ALGABER_OPENMP_ENABLED) {
#pragma omp parallel for num_threads(ThreadCount) collapse(2)
      for (size_t i_outer = 0; i_outer < view_rows_; i_outer += block_size) {
        for (size_t j_outer = 0; j_outer < view_cols_; j_outer += block_size) {
          size_t i_end = std::min(i_outer + block_size, view_rows_);
          size_t j_end = std::min(j_outer + block_size, view_cols_);
          for (size_t i = i_outer; i < i_end; ++i) {
            for (size_t j = j_outer; j < j_end; ++j) {
              result(i, j) = operator()(i, j) - rhs(i, j);
            }
          }
        }
      }
    } else {
      for (size_t i_outer = 0; i_outer < view_rows_; i_outer += block_size) {
        size_t i_end = std::min(i_outer + block_size, view_rows_);
        for (size_t j_outer = 0; j_outer < view_cols_; j_outer += block_size) {
          size_t j_end = std::min(j_outer + block_size, view_cols_);
          for (size_t i = i_outer; i < i_end; ++i) {
            for (size_t j = j_outer; j < j_end; ++j) {
              result(i, j) = operator()(i, j) - rhs(i, j);
            }
          }
        }
      }
    }
    return result;
  }

  Matrix operator*(const Matrix& rhs) const {
    if (view_cols_ != rhs.view_rows_) {
      throw std::invalid_argument(
          "Matrix dimensions incompatible for multiplication");
    }

    const size_t block_size = BLOCK_SIZE;

    Matrix result(view_rows_, rhs.view_cols_, T{0});

    if constexpr (ALGABER_OPENMP_ENABLED) {
#pragma omp parallel for num_threads(ThreadCount)
      for (size_t i_outer = 0; i_outer < view_rows_; i_outer += block_size) {
        size_t i_end = std::min(i_outer + block_size, view_rows_);
        for (size_t k_outer = 0; k_outer < view_cols_; k_outer += block_size) {
          size_t k_end = std::min(k_outer + block_size, view_cols_);
          for (size_t j_outer = 0; j_outer < rhs.view_cols_;
               j_outer += block_size) {
            size_t j_end = std::min(j_outer + block_size, rhs.view_cols_);
            for (size_t i = i_outer; i < i_end; ++i) {
              for (size_t k = k_outer; k < k_end; ++k) {
                T a = operator()(i, k);
                for (size_t j = j_outer; j < j_end; ++j) {
                  result(i, j) += a * rhs(k, j);
                }
              }
            }
          }
        }
      }
    } else {
      for (size_t i_outer = 0; i_outer < view_rows_; i_outer += block_size) {
        size_t i_end = std::min(i_outer + block_size, view_rows_);
        for (size_t k_outer = 0; k_outer < view_cols_; k_outer += block_size) {
          size_t k_end = std::min(k_outer + block_size, view_cols_);
          for (size_t j_outer = 0; j_outer < rhs.view_cols_;
               j_outer += block_size) {
            size_t j_end = std::min(j_outer + block_size, rhs.view_cols_);
            for (size_t i = i_outer; i < i_end; ++i) {
              for (size_t k = k_outer; k < k_end; ++k) {
                T a = operator()(i, k);
                for (size_t j = j_outer; j < j_end; ++j) {
                  result(i, j) += a * rhs(k, j);
                }
              }
            }
          }
        }
      }
    }

    return result;
  }

  // ===== Block-Based Scalar Multiplication =====
  Matrix operator*(const T& scalar) const {
    const size_t block_size = BLOCK_SIZE;
    Matrix result(view_rows_, view_cols_);

    if constexpr (ALGABER_OPENMP_ENABLED) {
#pragma omp parallel for num_threads(ThreadCount) collapse(2)
      for (size_t i_outer = 0; i_outer < view_rows_; i_outer += block_size) {
        for (size_t j_outer = 0; j_outer < view_cols_; j_outer += block_size) {
          size_t i_end = std::min(i_outer + block_size, view_rows_);
          size_t j_end = std::min(j_outer + block_size, view_cols_);
          for (size_t i = i_outer; i < i_end; ++i) {
            for (size_t j = j_outer; j < j_end; ++j) {
              result(i, j) = operator()(i, j) * scalar;
            }
          }
        }
      }
    } else {
      for (size_t i_outer = 0; i_outer < view_rows_; i_outer += block_size) {
        size_t i_end = std::min(i_outer + block_size, view_rows_);
        for (size_t j_outer = 0; j_outer < view_cols_; j_outer += block_size) {
          size_t j_end = std::min(j_outer + block_size, view_cols_);
          for (size_t i = i_outer; i < i_end; ++i) {
            for (size_t j = j_outer; j < j_end; ++j) {
              result(i, j) = operator()(i, j) * scalar;
            }
          }
        }
      }
    }

    return result;
  }

  // ===== Block-Based Scalar Division =====
  Matrix operator/(const T& scalar) const {
    if (scalar == T{0})
      throw std::invalid_argument("Division by zero");

    const size_t block_size = BLOCK_SIZE;
    Matrix result(view_rows_, view_cols_);

    if constexpr (ALGABER_OPENMP_ENABLED) {
#pragma omp parallel for num_threads(ThreadCount) collapse(2)
      for (size_t i_outer = 0; i_outer < view_rows_; i_outer += block_size) {
        for (size_t j_outer = 0; j_outer < view_cols_; j_outer += block_size) {
          size_t i_end = std::min(i_outer + block_size, view_rows_);
          size_t j_end = std::min(j_outer + block_size, view_cols_);
          for (size_t i = i_outer; i < i_end; ++i) {
            for (size_t j = j_outer; j < j_end; ++j) {
              result(i, j) = operator()(i, j) / scalar;
            }
          }
        }
      }
    } else {
      for (size_t i_outer = 0; i_outer < view_rows_; i_outer += block_size) {
        size_t i_end = std::min(i_outer + block_size, view_rows_);
        for (size_t j_outer = 0; j_outer < view_cols_; j_outer += block_size) {
          size_t j_end = std::min(j_outer + block_size, view_cols_);
          for (size_t i = i_outer; i < i_end; ++i) {
            for (size_t j = j_outer; j < j_end; ++j) {
              result(i, j) = operator()(i, j) / scalar;
            }
          }
        }
      }
    }

    return result;
  }

  Matrix slice(size_t start, size_t end, Orientation sliceType) const {
    if (sliceType == Orientation::Row) {
      if (start >= view_rows_ || end > view_rows_ || start > end) {
        throw std::out_of_range("Invalid row slice indices");
      }
      size_t sliceRows = end - start;
      Matrix result(sliceRows, view_cols_);
      for (size_t i = start; i < end; ++i) {
        for (size_t j = 0; j < view_cols_; ++j) {
          result(i - start, j) = (*this)(i, j);
        }
      }
      return result;
    } else if (sliceType == Orientation::Column) {
      if (start >= view_cols_ || end > view_cols_ || start > end) {
        throw std::out_of_range("Invalid column slice indices");
      }
      size_t sliceCols = end - start;
      Matrix result(view_rows_, sliceCols);
      for (size_t i = 0; i < view_rows_; ++i) {
        for (size_t j = start; j < end; ++j) {
          result(i, j - start) = (*this)(i, j);
        }
      }
      return result;
    } else {
      throw std::invalid_argument("Invalid slice type");
    }
  }

  Matrix unit_vector(size_t n, size_t i) {
    Matrix<T, StoragePolicy> vec(n, 1, T(0));
    vec(i, 0) = T(1);
    return vec;
  }

  //first

  Matrix slice(size_t row_start, size_t row_end, size_t col_start,
               size_t col_end) const {
    if (row_end > view_rows_ || col_end > view_cols_ || row_start >= row_end ||
        col_start >= col_end) {
      throw std::invalid_argument("Invalid slice dimensions");
    }
    return view(row_start, col_start, row_end - row_start, col_end - col_start);
  }
  Matrix operator[](
      std::pair<std::pair<size_t, size_t>, Orientation> indices) const {
    size_t start = indices.first.first;
    size_t end = indices.first.second;
    Orientation sliceType = indices.second;

    if (sliceType == Orientation::Row) {
      return slice(start, end, Orientation::Row);
    } else if (sliceType == Orientation::Column) {
      return slice(start, end, Orientation::Column);
    } else {
      throw std::invalid_argument("Invalid slice type");
    }
  }

  Matrix operator[](std::pair<size_t, Orientation> index) const {
    size_t idx = index.first;
    Orientation sliceType = index.second;

    if (sliceType == Orientation::Row) {
      return slice(idx, idx + 1, Orientation::Row);
    } else if (sliceType == Orientation::Column) {
      return slice(idx, idx + 1, Orientation::Column);
    } else {
      throw std::invalid_argument("Invalid slice type");
    }
  }
  // In-place addition
  Matrix& operator+=(const Matrix& rhs) {
    if (view_rows_ != rhs.view_rows_ || view_cols_ != rhs.view_cols_)
      throw std::invalid_argument("Matrix dimensions mismatch");

    for (size_t i = 0; i < view_rows_; ++i)
      for (size_t j = 0; j < view_cols_; ++j)
        operator()(i, j) += rhs(i, j);

    return *this;
  }

  // In-place subtraction
  Matrix& operator-=(const Matrix& rhs) {
    if (view_rows_ != rhs.view_rows_ || view_cols_ != rhs.view_cols_)
      throw std::invalid_argument("Matrix dimensions mismatch");

    for (size_t i = 0; i < view_rows_; ++i)
      for (size_t j = 0; j < view_cols_; ++j)
        operator()(i, j) -= rhs(i, j);

    return *this;
  }

  // In-place scalar multiplication
  Matrix& operator*=(const T& scalar) {
    for (size_t i = 0; i < view_rows_; ++i)
      for (size_t j = 0; j < view_cols_; ++j)
        operator()(i, j) *= scalar;

    return *this;
  }
  // In-place scalar division
  Matrix& operator/=(const T& scalar) {
    if (scalar == T{0})
      throw std::invalid_argument("Division by zero");

    for (size_t i = 0; i < view_rows_; ++i)
      for (size_t j = 0; j < view_cols_; ++j)
        operator()(i, j) /= scalar;

    return *this;
  }

  return *std::max_element(begin(), end());
}

  // Trace (sum of diagonal elements)
  T trace() const {
  if (!is_square())
    throw std::invalid_argument("Trace requires square matrix");

  T result = T{0};
  for (size_t i = 0; i < view_rows_; ++i) {
    result += operator()(i, i);
  }
  return result;
}

T frobenius_norm() const {
  T sum_squares = T{};
  for (size_t i = 0; i < rows(); ++i) {
    for (size_t j = 0; j < cols(); ++j) {
      sum_squares += operator()(i, j) * operator()(i, j);
    }
  }
  return std::sqrt(sum_squares);
}

// Alias for frobenius_norm for compatibility
T norm() const {
  return frobenius_norm();
}
//Typical getters-1
void validateIndexes(size_t row, size_t col) const {
  if (row >= view_rows_ || col >= view_cols_)
    throw std::out_of_range("Index out of bounds");
}

// Getter function
T get_element(size_t row, size_t col) const {
  try {
    validateIndexes(row, col);
    return operator()(row, col);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;

    throw;
  }
}

Matrix getRow(size_t index) {
  if (index >= view_rows_)
    throw std::invalid_argument("Row index out of bounds");

  Matrix result(static_cast<size_t>(1), static_cast<size_t>(view_cols_));
  for (size_t i = 0; i < view_cols_; i++)
    result(0, i) = operator()(index, i);

  return result;
}

Matrix getCol(size_t index) {
  if (index >= view_cols_)
    throw std::invalid_argument("Column index out of bounds");

  Matrix result(static_cast<size_t>(view_rows_), static_cast<size_t>(1));
  for (size_t i = 0; i < view_rows_; i++)
    result(i, 0) = operator()(i, index);

  return result;
}

// Setter function
void set_element(size_t row, size_t col, const T& value) {
  try {
    validateIndexes(row, col);
    operator()(row, col) = value;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    throw;
  }
}

Matrix<T>& setRow(Matrix<T> rowVector, size_t rowPos) {

  if (rowVector.rows() != 1)
    throw std::invalid_argument("Too many rows provided");

  if (rowVector.cols() != view_cols_)
    throw std::invalid_argument("Incorrect number of columns");
  if (rowPos >= view_rows_)
    throw std::out_of_range("Row index out of bounds");

  for (size_t col = 0; col < view_cols_; col++)
    operator()(rowPos, col) = rowVector(0, col);

  return *this;
}

Matrix& setColumn(Matrix columnVector, size_t colPos) {
  if (columnVector.rows() != view_rows_)
    throw std::invalid_argument("Too many rows provided");

  if (columnVector.cols() > 1)
    throw std::invalid_argument("Too many columns provided");
  if (colPos >= view_cols_)
    throw std::out_of_range("Column index out of bounds");
  for (size_t row = 0; row < rows(); row++)
    operator()(row, colPos) = columnVector(row, 0);
  return *this;
}

Matrix& swapRows(size_t r1, size_t r2) {
  if (std::max(r1, r2) >= rows())
    throw std::out_of_range("Row index out of bounds");

  for (size_t col = 0; col < cols(); col++) {
    std::swap(operator()(r1, col), operator()(r2, col));
  }

  return *this;
}

Matrix& swapColumns(size_t c1, size_t c2) {
  if (std::max(c1, c2) >= cols())
    throw std::out_of_range("Column index out of bounds");

  for (size_t row = 0; row < rows(); row++) {
    std::swap(operator()(row, c1), operator()(row, c2));
  }

  return *this;
}

//Application Based functions (LINEAR ALGEBRA)
// ==================== Linear Algebra Functions ====================

void normalize_pivot_row(Matrix& matrix, const T& pivotElement,
                         size_t processingRow, size_t processingColumn) const {
  if (pivotElement == T{0})
    return;
  // Normalize the current row
  size_t block_size = BLOCK_SIZE;
  for (size_t j_outer = processingColumn; j_outer < matrix.cols();
       j_outer += block_size) {
    size_t j_end = std::min(j_outer + block_size, matrix.cols());
    for (size_t j = j_outer; j < j_end; ++j) {
      matrix(processingRow, j) /= pivotElement;
    }
  }
}

void eliminate_rows(Matrix& matrix, const T& pivotElement, size_t processingRow,
                    size_t processingColumn, bool reduced_form) const {
  if (pivotElement == T{0})
    return;

  // Eliminate in other rows
  for (size_t i = 0; i < matrix.rows(); ++i) {
    if (i == processingRow)
      continue;

    // Only eliminate below the pivot row when not in reduced form
    if (!reduced_form && i < processingRow)
      continue;

    T factor = matrix(i, processingColumn);
    // for (size_t j = processingColumn; j < matrix.cols(); ++j) {
    //   matrix(i, j) -= factor * matrix(processingRow, j);
    // }
    size_t block_size = BLOCK_SIZE;
    for (size_t j_outer = processingColumn; j_outer < matrix.cols();
         j_outer += block_size) {
      size_t j_end = std::min(j_outer + block_size, matrix.cols());
      for (size_t j = j_outer; j < j_end; ++j) {
        matrix(i, j) -= factor * matrix(processingRow, j);
      }
    }
  }
}

template <typename U>
Matrix<U> augmentedMatrix(const std::vector<U>& vector) const {

  if (vector.size() != this->view_rows_)
    throw bad_size();

  Matrix<U> augmented(this->view_rows_, this->view_cols_ + 1);

  // Copy the original matrix into the left half of the augmented matrix
  for (size_t i = 0; i < this->view_rows_; ++i) {
    for (size_t j = 0; j < this->view_cols_; ++j) {
      augmented(i, j) = this->operator()(i, j);
    }
  }

  // Copy the vector into the rightmost column of the augmented matrix
  for (size_t i = 0; i < this->view_rows_; ++i) {
    augmented(i, this->view_cols_) = vector[i];
  }

  return augmented;
}

// Overload to augment with another matrix
template <typename U>
Matrix<U> augmentedMatrix(const Matrix<U>& other) const {
  if (other.rows() != this->view_rows_)
    throw bad_size();

  Matrix<U> augmented(this->view_rows_, this->view_cols_ + other.cols());

  // Copy the original matrix into the left part of the augmented matrix
  for (size_t i = 0; i < this->view_rows_; ++i) {
    for (size_t j = 0; j < this->view_cols_; ++j) {
      augmented(i, j) = this->operator()(i, j);
    }
  }

  // Copy the other matrix into the right part of the augmented matrix
  for (size_t i = 0; i < this->view_rows_; ++i) {
    for (size_t j = 0; j < other.cols(); ++j) {
      augmented(i, this->view_cols_ + j) = other(i, j);
    }
  }

  return augmented;
}

std::pair<Matrix, int> echelonizeWithSign(bool reduced_row_echelon) const {
  Matrix result = *this;
  int sign = 1;  // For determinant calculation

  size_t lead = 0;
  size_t rowCount = result.rows();
  size_t colCount = result.cols();

  for (size_t r = 0; r < rowCount; ++r) {
    if (lead >= colCount)
      break;

    // Find pivot
    size_t i = r;
    while (i < rowCount && result(i, lead) == T{0}) {
      ++i;
    }

    if (i == rowCount) {
      // No pivot found in this column
      ++lead;
      --r;
      continue;
    }

    // Swap rows if needed
    if (i != r) {
      for (size_t j = 0; j < colCount; ++j) {
        std::swap(result(r, j), result(i, j));
      }
      sign *= -1;  // Track change in sign for determinant
    }

    // Normalize pivot row
    T pivot = result(r, lead);
    if (reduced_row_echelon) {
      normalize_pivot_row(result, pivot, r, lead);
    }

    // Eliminate other rows
    eliminate_rows(result, pivot, r, lead, reduced_row_echelon);

    ++lead;
  }

  return {result, sign};
}
Matrix rowEchelon() const {

  auto [echelonMatrix, sign] = echelonizeWithSign(false);
  return echelonMatrix;  // Call echelon with default argument
}

Matrix rref() {
  auto [echelonMatrix, sign] = echelonizeWithSign(true);
  return echelonMatrix;  // Call echelon with make_pivot_one set to true
}

// Partial pivoting LU decomposition (row pivoting only)
Decomposition_LU to_LU_partial_pivoted() const {
  if (!is_square())
    throw std::invalid_argument("LU decomposition requires square matrix");

  const size_t n = view_rows_;
  Decomposition_LU lu(n);
  lu.U = *this;

  for (size_t k = 0; k < n; ++k) {
    // Partial pivoting - find max in current column
    size_t max_row = k;
    auto max_val = std::abs(lu.U(k, k));
    for (size_t i = k + 1; i < n; ++i) {
      if (std::abs(lu.U(i, k)) > max_val) {
        max_val = std::abs(lu.U(i, k));
        max_row = i;
      }
    }

    // Swap rows if needed
    if (max_row != k) {
      lu.U.swapRows(k, max_row);
      std::swap(lu.P[k], lu.P[max_row]);
      lu.sign *= -1;
    }

    // Check for singularity
    if (std::abs(lu.U(k, k)) < 1e-12) {
      throw std::runtime_error("Matrix is singular to working precision");
    }

    // Compute multipliers and eliminate
    for (size_t i = k + 1; i < n; ++i) {
      lu.U(i, k) /= lu.U(k, k);
      for (size_t j = k + 1; j < n; ++j) {
        lu.U(i, j) -= lu.U(i, k) * lu.U(k, j);
      }
    }
  }

  // Extract L and U
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (i > j) {
        lu.L(i, j) = lu.U(i, j);
        lu.U(i, j) = 0;
      } else if (i == j) {
        lu.L(i, j) = 1;
      } else {
        lu.L(i, j) = 0;
      }
    }
  }

  return lu;
}

// Full pivoting LU decomposition (row and column pivoting)
Decomposition_LU to_LU_full_pivoted() const {
  if (!is_square())
    throw std::invalid_argument("LU decomposition requires square matrix");

  const size_t n = view_rows_;
  Decomposition_LU lu(n);
  lu.U = *this;
  lu.full_pivoting = true;

  for (size_t k = 0; k < n; ++k) {
    // Full pivoting - find max in entire remaining submatrix
    size_t max_row = k;
    size_t max_col = k;
    auto max_val = std::abs(lu.U(k, k));

    for (size_t i = k; i < n; ++i) {
      for (size_t j = k; j < n; ++j) {
        if (std::abs(lu.U(i, j)) > max_val) {
          max_val = std::abs(lu.U(i, j));
          max_row = i;
          max_col = j;
        }
      }
    }

    // Swap rows if needed
    if (max_row != k) {
      lu.U.swapRows(k, max_row);
      std::swap(lu.P[k], lu.P[max_row]);
      lu.sign *= -1;
    }

    // Swap columns if needed
    if (max_col != k) {
      lu.U.swapColumns(k, max_col);
      std::swap(lu.Q[k], lu.Q[max_col]);
      lu.sign *= -1;
    }

    // Check for singularity
    if (std::abs(lu.U(k, k)) < 1e-12) {
      throw std::runtime_error("Matrix is singular to working precision");
    }

    // Compute multipliers and eliminate
    for (size_t i = k + 1; i < n; ++i) {
      lu.U(i, k) /= lu.U(k, k);
      for (size_t j = k + 1; j < n; ++j) {
        lu.U(i, j) -= lu.U(i, k) * lu.U(k, j);
      }
    }
  }

  // Extract L and U
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (i > j) {
        lu.L(i, j) = lu.U(i, j);
        lu.U(i, j) = 0;
      } else if (i == j) {
        lu.L(i, j) = 1;
      } else {
        lu.L(i, j) = 0;
      }
    }
  }

  return lu;
}

// Main LU decomposition method with strategy selection
Decomposition_LU to_LU(
    PivotingStrategy strategy = PivotingStrategy::FULL) const {
  switch (strategy) {
    case PivotingStrategy::NONE:
      // Basic LU without pivoting (unstable for singular/near-singular matrices)
      throw std::runtime_error(
          "LU without pivoting is unstable and not supported. Use partial or "
          "full pivoting.");
    case PivotingStrategy::PARTIAL:
      return to_LU_partial_pivoted();
    case PivotingStrategy::FULL:
      return to_LU_full_pivoted();
    default:
      throw std::invalid_argument("Unknown pivoting strategy");
  }
}

Decomposition_LDU to_LDU() const {
  if (!is_square())
    throw std::invalid_argument("LDU decomposition requires square matrix");

  // First get LU decomposition
  auto lu = to_LU();

  size_t n = view_rows_;
  Matrix D = Matrix::zeros(n, n);
  Matrix U = Matrix::identity(n);

  // Extract diagonal from U to D and normalize U
  for (size_t i = 0; i < n; ++i) {
    if (lu.U(i, i) == T{0}) {
      throw std::runtime_error("LDU decomposition failed: zero diagonal in U");
    }

    D(i, i) = lu.U(i, i);

    for (size_t j = i; j < n; ++j) {
      U(i, j) = lu.U(i, j) / D(i, i);
    }
  }

  return {lu.L, D, U};
}

std::vector<std::pair<size_t, Matrix>> getPivotColumns() {
  auto echelon_result = echelonizeWithSign(false).first;
  std::vector<std::pair<size_t, Matrix>> pivotColumns;

  for (size_t i = 0; i < echelon_result.rows(); ++i) {
    // Find the pivot column in this row
    for (size_t j = 0; j < echelon_result.cols(); ++j) {
      if (echelon_result(i, j) != T{0}) {
        // Found a pivot column
        pivotColumns.push_back({j, this->get_column(j)});
        break;
      }
    }
  }

  return pivotColumns;
}

T det() const {
  if (!is_square())
    throw std::invalid_argument("Determinant requires square matrix");

  if (view_rows_ == 1) {
    return operator()(0, 0);
  }

  if (view_rows_ == 2) {
    return operator()(0, 0) * operator()(1, 1) - operator()(0, 1) * operator()(
                                                                        1, 0);
  }

  // For larger matrices, use row echelon form
  auto [echelon, sign] = echelonizeWithSign(false);

  T determinant = sign;
  for (size_t i = 0; i < view_rows_; ++i) {
    determinant *= echelon(i, i);
  }

  return determinant;
}

T rank() const {
  return static_cast<T>(
      echelonizeWithSign(false).first.getPivotColumns().size());
}
T norm_squared() const {
  T sum{};
  for (const auto& val : *this)
    sum += val * val;
  return sum;
}

Matrix sub_matrix(const size_t row, const size_t column) const {
  if (row >= view_rows_ || column >= view_cols_) {
    throw std::invalid_argument("Row or column index out of bounds");
  }

  Matrix result(view_rows_ - 1, view_cols_ - 1);

  for (size_t i = 0, r = 0; i < view_rows_; ++i) {
    if (i == row)
      continue;

    for (size_t j = 0, c = 0; j < view_cols_; ++j) {
      if (j == column)
        continue;

      result(r, c) = operator()(i, j);
      ++c;
    }
    ++r;
  }

  return result;
}

Matrix<long double> inverse() const {
  if (!is_square())
    throw std::invalid_argument("Inverse requires square matrix");

  T determinant = det();
  if (determinant == T{0}) {
    throw std::invalid_argument(
        "Matrix is not invertible (determinant is zero)");
  }

  size_t n = view_rows_;

  if (n <= 3) {
    // Use adjugate method for small matrices
    Matrix<long double> result(n, n);

    if (n == 1) {
      result(0, 0) = 1.0L / static_cast<long double>(operator()(0, 0));
      return result;
    }

    // Calculate cofactor matrix
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        // Calculate minor
        Matrix minor = sub_matrix(i, j);
        T minor_det = minor.det();

        // Calculate cofactor (include sign)
        T cofactor = ((i + j) % 2 == 0) ? minor_det : -minor_det;

        // Adjugate is transpose of cofactor matrix
        result(j, i) = static_cast<long double>(cofactor) /
                       static_cast<long double>(determinant);
      }
    }

    return result;
  } else {

    size_t col = view_cols_;
    Matrix<long double> doubleMatrix(*this);
    Matrix<long double> eye = identity(view_rows_);
    Matrix<long double> augmented = doubleMatrix.augmentedMatrix(eye);
    Matrix<long double> inverseMatrix =
        augmented.rref()[{{col, 2 * col}, algaber::Orientation::Column}];
    return inverseMatrix;
  }
}

void findPivotAndFreeColumns(std::vector<size_t>& pivotColumns,
                             std::vector<size_t>& freeColumns) {
  Matrix rrefMatrix = *this;
  size_t numRows = rrefMatrix->rows();
  size_t numCols = rrefMatrix->cols() - 1;  // Exclude the last column (b)

  std::vector<bool> columnFound(numCols, false);

  for (size_t rowIdx = 0; rowIdx < numRows; ++rowIdx) {
    bool foundPivot = false;
    for (size_t colIdx = 0; colIdx < numCols; ++colIdx) {
      if (rrefMatrix(rowIdx, colIdx) != 0) {
        pivotColumns.push_back(colIdx);
        columnFound[colIdx] = true;
        foundPivot = true;
        break;
      }
    }
    if (!foundPivot) {
      // If no pivot is found in this row, all columns in this row are free columns
      for (size_t colIdx = 0; colIdx < numCols; ++colIdx) {
        if (!columnFound[colIdx]) {
          freeColumns.push_back(colIdx);
        }
      }
      // No need to check further rows
      break;
    }
  }
}

void calculateXP(const std::vector<size_t>& pivotColumns, size_t numUnknowns,
                 std::vector<T>& xp) {
  Matrix rrefMatrix = *this;
  for (size_t i = 0; i < pivotColumns.size(); ++i) {
    size_t pivotCol = pivotColumns[i];
    for (size_t rowIdx = 0; rowIdx < rrefMatrix->rows(); ++rowIdx) {
      if (rrefMatrix(rowIdx, pivotCol) != 0) {
        xp[pivotCol] = rrefMatrix(rowIdx, numUnknowns);
        break;
      }
    }
  }
}

std::vector<std::vector<T>> calculateXs(const std::vector<size_t>& pivotColumns,
                                        const std::vector<size_t>& freeColumns,
                                        size_t numCols) {
  Matrix rrefMatrix = *this;
  std::vector<std::vector<T>> xsExpressions;
  size_t numXs = freeColumns.size();

  for (size_t xsIdx = 0; xsIdx < numXs; ++xsIdx) {
    std::vector<T> xs(numCols, 0);
    size_t freeCol = freeColumns[xsIdx];
    xs[freeCol] = 1;

    for (size_t i = 0; i < pivotColumns.size(); ++i) {
      size_t pivotCol = pivotColumns[i];
      T coefficient = 0;  // Initialize coefficient to 0
      bool foundNonZero = false;

      for (size_t rowIdx = 0; rowIdx < rrefMatrix->rows(); ++rowIdx) {
        if (rrefMatrix(rowIdx, pivotCol) != 0) {
          if (!foundNonZero) {
            coefficient =
                -rrefMatrix(rowIdx, freeCol) / rrefMatrix(rowIdx, pivotCol);
            foundNonZero = true;
          } else {
            coefficient *= rrefMatrix(rowIdx, pivotCol);
            coefficient -= rrefMatrix(rowIdx, freeCol);
          }
        }
      }

      xs[pivotCol] = coefficient;
    }

    xsExpressions.push_back(xs);
  }

  return xsExpressions;
}
std::string printLinearCombination(
    const std::vector<std::vector<double>>& xsExpressions) {
  std::ostringstream linearCombStream;
  for (size_t i = 0; i < xsExpressions.size(); ++i) {
    linearCombStream << "C" << i + 1 << " * (";
    const auto& xs = xsExpressions[i];
    for (size_t j = 0; j < xs.size(); ++j) {
      linearCombStream << "X" << j + 1 << " = " << xs[j];
      if (j < xs.size() - 1) {
        linearCombStream << ", ";
      }
    }
    linearCombStream << ")";
    if (i < xsExpressions.size() - 1) {
      linearCombStream << " + ";
    }
  }
  return linearCombStream.str();
}

Solution solver_AX_b(const std::vector<T>& b) {
  Matrix A = *this;
  size_t numUnknowns = A->cols();
  Matrix augmented = A->augmentedMatrix(b);
  Matrix echelonMatrix = augmented.rowEchelon();

  SolutionType solutionType = echelonMatrix.findSolutionType(numUnknowns);

  switch (solutionType) {
    case SolutionType::EXACT_SOLUTION: {
      std::vector<T> xp = echelonMatrix.backSubstitution();
      Matrix<T> XP(xp, VectorType::ColumnVector);  // Convert xp to XP
      std::vector<Matrix<T>> xs;
      // Empty xs for exact solution
      return {solutionType, XP, xs, ""};
    }
    case SolutionType::NO_SOLUTION:
      return {solutionType, {}, {}, "No solution exists."};
    case SolutionType::INFINITE_SOLUTIONS: {
      Matrix<T> rrefMatrix = augmented.rref();
      std::vector<size_t> pivotColumns;
      std::vector<size_t> freeColumns;

      rrefMatrix.findPivotAndFreeColumns(pivotColumns, freeColumns);
      std::vector<T> xp(numUnknowns, 0);
      rrefMatrix.calculateXP(pivotColumns, numUnknowns, xp);
      Matrix<T> XP(xp, VectorType::ColumnVector);  // Convert xp to XP
      std::vector<std::vector<T>> xs =
          rrefMatrix.calculateXs(pivotColumns, freeColumns, numUnknowns);
      std::vector<Matrix<T>> XS;
      for (const auto& x : xs) {
        XS.emplace_back(x, VectorType::ColumnVector);  // Convert each x to X
      }
      std::string linearComb = printLinearCombination(xs);
      return {solutionType, XP, XS, linearComb};
    }
  }

  // Default return statement in case no solution type matches
  return {SolutionType::NO_SOLUTION, {}, {}, "Unexpected error occurred."};
}
Matrix solve_triangular(const Matrix& rhs, bool lower, bool unit_diag) const {
  // Check if the current matrix is square
  if (rows() != cols()) {
    throw std::invalid_argument("Triangular matrix must be square");
  }
  // Check if rhs has the same number of rows as the matrix
  if (rhs.rows() != rows()) {
    throw std::invalid_argument("Right-hand side rows must match matrix rows");
  }

  // Create the solution matrix with the same dimensions as rhs
  Matrix X(rhs.rows(), rhs.cols());
  size_t block_size = BLOCK_SIZE;
  for (size_t col_outer = 0; col_outer < rhs.cols(); col_outer += block_size) {
    size_t col_end = std::min(col_outer + block_size, rhs.cols());
    // For each column in rhs, solve the triangular system
    for (size_t col = 0; col < rhs.cols(); ++col) {
      // Copy the current column from rhs to X
      for (size_t i = 0; i < rows(); ++i) {
        X(i, col) = rhs(i, col);
      }

      if (lower) {
        // Forward substitution for lower triangular matrix
        for (size_t i = 0; i < rows(); ++i) {
          T sum = X(i, col);
          for (size_t j = 0; j < i; ++j) {
            sum -= operator()(i, j) * X(j, col);
          }
          if (!unit_diag) {
            sum /= operator()(i, i);
          }
          X(i, col) = sum;
        }
      } else {
        // Backward substitution for upper triangular matrix
        for (size_t i = rows(); i-- > 0;) {
          T sum = X(i, col);
          for (size_t j = i + 1; j < cols(); ++j) {
            sum -= operator()(i, j) * X(j, col);
          }
          if (!unit_diag) {
            sum /= operator()(i, i);
          }
          X(i, col) = sum;
        }
      }
    }
  }

  return X;
}

// Linear system solver

Matrix solve(const Matrix& b) const {
  if (!is_square())
    throw std::invalid_argument("Coefficient matrix must be square");

  if (view_rows_ != b.rows())
    throw std::invalid_argument("Incompatible dimensions for system solving");

  size_t n = view_rows_;
  size_t m = b.cols();

  // Create augmented matrix [A|b]
  Matrix<T, StoragePolicy> aug(size_t(n), size_t(n + m));
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      aug(i, j) = operator()(i, j);
    }
    for (size_t j = 0; j < m; j++) {
      aug(i, n + j) = b(i, j);
    }
  }

  // Perform Gaussian elimination with partial pivoting
  for (size_t i = 0; i < n; i++) {
    // Find pivot
    size_t pivot_row = i;
    T pivot_val = std::abs(aug(i, i));

    for (size_t j = i + 1; j < n; j++) {
      if (std::abs(aug(j, i)) > pivot_val) {
        pivot_val = std::abs(aug(j, i));
        pivot_row = j;
      }
    }

    if (pivot_val < 1e-10)
      throw std::runtime_error("Matrix is singular or nearly singular");

    // Swap rows if needed
    if (pivot_row != i) {
      for (size_t j = 0; j < n + m; j++) {
        std::swap(aug(i, j), aug(pivot_row, j));
      }
    }

    // Eliminate below
    for (size_t j = i + 1; j < n; j++) {
      T factor = aug(j, i) / aug(i, i);
      for (size_t k = i; k < n + m; k++) {
        aug(j, k) -= factor * aug(i, k);
      }
    }
  }

  // Back substitution
  Matrix<T, StoragePolicy> x(size_t(n), size_t(m));
  for (size_t k = 0; k < m; k++) {
    for (size_t i = n; i-- > 0;) {
      T sum = aug(i, n + k);
      for (size_t j = i + 1; j < n; j++) {
        sum -= aug(i, j) * x(j, k);
      }
      x(i, k) = sum / aug(i, i);
    }
  }

  return x;
}

// Algorithm 3: Householder reflector based HT reduction

std::tuple<Matrix, Matrix, Matrix, Matrix> householder_ht_reduction(
    const Matrix<T, StoragePolicy>& A_in,
    const Matrix<T, StoragePolicy>& B_in) {
  if (!A_in.is_square() || !B_in.is_square() || A_in.rows() != B_in.rows())
    throw std::invalid_argument(
        "Matrices A and B must be square and of the same size");

  size_t n = A_in.rows();

  // Create copies to work with
  Matrix A = A_in.clone();
  Matrix B = B_in.clone();

  // Initialize Q and Z as identity matrices
  Matrix Q = Matrix::identity(n);
  Matrix Z = Matrix::identity(n);

  // Step 2: Calculate the first Householder reflector H1 for B
  Matrix B_col = B.get_column(0);

  auto [v1, beta1] = B_col.house_v();

  // Step 3: Apply H1 to A and B from the left and to Q from the right
  A = A.apply_householder_left(v1, beta1);
  B = B.apply_householder_left(v1, beta1);
  Q = Q.apply_householder_right(v1, beta1);

  // Steps 4-10: Main loop for j = 1 to n-2
  for (size_t j = 0; j < n - 2; j++) {
    // Step 5: Calculate Householder reflector H2 for A
    Matrix A_col = A.view(j + 1, j, n - (j + 1), 1);

    auto [v2, beta2] = A_col.house_v();

    // Step 6: Apply H2 to A and B from the left and to Q from the right
    Matrix A_block = A.view(j + 1, j, n - (j + 1), n - j);
    A_block = A_block.apply_householder_left(v2, beta2);
    A.set_block(j + 1, j, A_block);

    Matrix B_block = B.view(j + 1, j, n - (j + 1), n - j);
    B_block = B_block.apply_householder_left(v2, beta2);
    B.set_block(j + 1, j, B_block);

    Matrix Q_block = Q.view(0, j + 1, n, n - (j + 1));
    Q_block = Q_block.apply_householder_right(v2, beta2);
    Q.set_block(0, j + 1, Q_block);

    // Step 7: Solve the linear system B_{j+1:n, j+1:n} * x = e1
    Matrix B_submatrix = B.view(j + 1, j + 1, n - (j + 1), n - (j + 1));
    Matrix e1 = Matrix::unit_vector(n - (j + 1), 0);
    Matrix x = B_submatrix.solve(e1);

    // Step 8: Calculate Householder reflector H3 for x

    auto [v3, beta3] = x.house_v();

    // Step 9: Apply H3 to A, B, and Z from the right
    Matrix A_right_block = A.view(0, j + 1, n, n - (j + 1));
    A_right_block = A_right_block.apply_householder_right(v3, beta3);
    A.set_block(0, j + 1, A_right_block);

    Matrix B_right_block = B.view(0, j + 1, n, n - (j + 1));
    B_right_block = B_right_block.apply_householder_right(v3, beta3);
    B.set_block(0, j + 1, B_right_block);

    Matrix Z_right_block = Z.view(0, j + 1, n, n - (j + 1));
    Z_right_block = Z_right_block.apply_householder_right(v3, beta3);
    Z.set_block(0, j + 1, Z_right_block);
  }

  return {A, B, Q, Z};
}

Matrix apply_householder_left(const Matrix<T, StoragePolicy>& v, T beta) {
  Matrix A = *this;
  // w = beta * A^T * v
  Matrix<T, StoragePolicy> w = beta * (A.transpose() * v);
  // A = A - v * w^T
  A = A - v * w.transpose();
  return A;
}

// Apply Householder reflector from the right: A = A(I - beta*v*v^T)

Matrix apply_householder_right(const Matrix<T, StoragePolicy>& v, T beta) {
  Matrix A = *this;
  // w = beta * A * v
  Matrix<T, StoragePolicy> w = beta * (A * v);
  // A = A - w * v^T
  A = A - w * v.transpose();
  return A;
}

std::tuple<Matrix<T>, T> house_v() const {
  size_t m = this->rows();
  Matrix<T> u(size_t(m), size_t(1));
  T tau = T(0);

  if (m == 0) {
    return {u, tau};
  }

  T x0 = (*this)(0, 0);
  Matrix<T> x_sub = this->view(1, 0, m - 1, 1);
  T sigma = x_sub.norm();

  if (sigma == T(0)) {
    u(0, 0) = T(1);
    return {u, tau};
  }

  T mu = std::sqrt(x0 * x0 + sigma * sigma);
  T beta;

  if (x0 <= T(0)) {
    beta = x0 - mu;
  } else {
    beta = -sigma * sigma / (x0 + mu);
  }

  // Set the first component of u
  u(0, 0) = T(1);

  // Set remaining components and normalize
  T scale = T(1) / beta;
  for (size_t i = 1; i < m; ++i) {
    u(i, 0) = (*this)(i, 0) * scale;
  }

  tau = T(2) * beta * beta / (sigma * sigma + beta * beta);

  return {u, tau};
}
};

// ==================== Free Functions ====================

// Output stream operator for matrices
template <Arithmetic T, typename StoragePolicy>
std::ostream& operator<<(std::ostream& os, const Matrix<T, StoragePolicy>& m) {
  os << "[";
  for (size_t i = 0; i < m.rows(); ++i) {
    os << (i == 0 ? "[" : " [");
    for (size_t j = 0; j < m.cols(); ++j) {
      os << m(i, j);
      if (j < m.cols() - 1)
        os << ", ";
    }
    os << "]";
    if (i < m.rows() - 1)
      os << ",\n";
  }
  os << "]";
  return os;
}

// Scalar multiplication (scalar on left side)
template <Arithmetic T, typename StoragePolicy>
Matrix<T, StoragePolicy> operator*(const T& scalar,
                                   const Matrix<T, StoragePolicy>& m) {
  return m * scalar;
}

// Implementation of PermutationMatrix::to_matrix() after Matrix class is fully defined
Matrix<double, InMemoryStorage<double>> PermutationMatrix::to_matrix() const {
  Matrix<double, InMemoryStorage<double>> P(perm_.size(), perm_.size());
  for (size_t i = 0; i < perm_.size(); ++i) {
    P(i, perm_[i]) = 1.0;
  }
  return P;
}
Matrix<double, InMemoryStorage<double>> PermutationMatrix::inverse() const {
  Matrix<double, InMemoryStorage<double>> inv(perm_.size(), perm_.size());
  for (size_t i = 0; i < perm_.size(); ++i) {
    inv(perm_[i], i) = 1.0;
  }
  return inv;
}
Matrix<double, InMemoryStorage<double>> PermutationMatrix::transpose() const {
  return inverse();
}

}  // namespace algaber

#endif  // MATRIX_HPP