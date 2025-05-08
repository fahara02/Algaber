#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <omp.h>
#include <algorithm>
#include <cassert>
#include <concepts>
#include <execution>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <ranges>
#include <span>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>
#include "StoragePolicy.hpp"
#include "custom_iterators.hpp"
#include "library_config.hpp"
#include "matrix_error.hpp"

// TODO: Core Linear Algebra Features
// 1. Matrix Decompositions:
//    ✅ LU decomposition (to_LU implemented with full pivoting)
//    ✅ LDU decomposition (to_LDU implemented)
//    ✅ QR decomposition (Householder)
//    ✅ Cholesky decomposition (for symmetric positive definite)
//    ⚠️ Eigenvalue decompositions
//       ✅ Matrix balancing implemented (BalancedForm)
//       ✅ Hessenberg form implemented (hessenberg_decomposition)
//       ✅ Hessenberg-Triangular reduction implemented
//       ❌ QR algorithm for eigenvalues
//    ❌ SVD (Singular Value Decomposition)

// 2. Solvers:
//    ✅ General linear system solver (using LU)
//       ⚠️ Partial: Implemented but could be optimized with better pivoting strategies
//    ❌ Least squares solver (using QR/SVD)
//    ❌ Sparse matrix solvers (Conjugate Gradient, GMRES)
//    ✅ solve_triangular implemented
//    ✅ Block Jacobi implemented

// 3. Matrix Operations:
//    ✅ Kronecker product
//    ✅ Hadamard product (elementwise operations are implemented based on tests)
//    ❌ Matrix exponential
//    ❌ Matrix functions (log, sqrt, etc.)
//    ✅ Basic operations (+, -, *, /) implemented
//    ✅ Transpose implemented
//    ✅ Determinant implemented
//       ⚠️ Uses recursive method for small matrices, REF for large
//    ✅ Inverse implemented
//       ⚠️ Uses adjugate for small matrices, REF for large - could use pivoted LU
//    ✅ Matrix form checks (is_hessenberg, etc.)

// TODO: Performance Optimizations
// 1. Block matrix operations ✅
//    ⚠️ Partial: Basic block operations implemented (see BlockIterator)
// 2. SIMD/vectorization support ❌
//    ✅ OpenMP support
// 3. Parallelization ✅ (OpenMP)
// 4. Memory-efficient storage options:
//    - Sparse matrices (CSR, CSC formats) ❌
//    - Banded matrices ❌
//    - Symmetric/Hermitian storage ❌

// TODO: Numerics
// 1. Condition number estimation ❌
// 2. Numerical stability improvements ❌
// 3. Iterative refinement ❌
// 4. Pivoting strategies ❌

// TODO: API Improvements
// 1. Expression templates for lazy evaluation ❌
// 2. Views/slices without copying ✅
//    ⚠️ Implemented with slice() method
// 3. Range-based constructors ❌
// 4. Matrix builders/initializers ✅
//    ✅ Basic constructors implemented
//    ✅ Factory methods (zeros, ones, identity, random, etc.)

// TODO: Applications Support
// 1. Statistics operations (covariance, PCA) ❌
// 2. Linear programming support ❌
// 3. Graph algorithms (adjacency matrices) ❌
// 4. Image processing operations ❌

// TODO: Testing & Validation
// 1. More comprehensive test cases ❌
//    ✅ Basic test suite exists
// 2. Benchmark suite ❌
// 3. Numerical accuracy verification ❌
// 4. Comparison with reference implementations ❌

// TODO: Documentation
// 1. API reference ❌
// 2. Usage examples ❌
// 3. Performance guidelines ❌
// 4. Mathematical background notes ❌

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

  struct Decomposition_LU {
    Matrix L;
    Matrix U;
    PermutationMatrix P;  // Row permutation
    PermutationMatrix Q;  // Column permutation (for full pivoting)
    int sign;             // Sign of permutation for determinant
    bool full_pivoting;   // Whether full pivoting was used

    Decomposition_LU(size_t n)
        : L(n, n), U(n, n), P(n), Q(n), sign(1), full_pivoting(false) {}
  };

  struct Decomposition_LDU {
    Matrix L;
    Matrix D;
    Matrix U;
  };
  struct Decomposition_QR {
    Matrix Q;
    Matrix R;
  };
  struct Decomposition_Cholesky {
    Matrix L;  // Lower triangular matrix
  };

  enum class SolutionType {
    EXACT_SOLUTION,      // Unique solution exists
    INFINITE_SOLUTIONS,  // Infinite solutions exist
    NO_SOLUTION

  };

  struct Solution {
    SolutionType type;

    Matrix XP;               // Exact solution
    std::vector<Matrix> XS;  // Infinite solutions
    std::string linearComb;  // For infinite solutions
  };

  struct HessenbergPanel {
    Matrix U_panel;  // Householder vectors
    Matrix Z_panel;  // Intermediate results for trailing updates
    Matrix T_panel;  // Triangular factor for accumulated transformations
  };

  // ===== Constructors =====

  // Default constructor - empty matrix
  Matrix()
      : storage_(std::make_shared<StoragePolicy>(0, 0)),
        view_rows_(0),
        view_cols_(0) {}
  Matrix(size_t dimension, T init_val = T{})
      : storage_(
            std::make_shared<StoragePolicy>(dimension, dimension, init_val)),
        view_rows_(dimension),
        view_cols_(dimension) {}

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

  //constructor to create matrix from arrays
  template <std::size_t N>
  Matrix(size_t rows, size_t cols, T (&data)[N])
      : storage_(std::make_shared<StoragePolicy>(rows, cols)),
        view_rows_(rows),
        view_cols_(cols) {
    if (rows * cols != N) {
      throw std::invalid_argument(
          "Array size does not match matrix dimensions");
    }
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        operator()(i, j) = data[i * cols + j];
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

  // Constructor for row or column vector
  Matrix(const std::vector<T>& data, VectorType vectorType)
      : storage_(std::make_shared<StoragePolicy>(
            vectorType == VectorType::RowVector ? 1 : data.size(),
            vectorType == VectorType::RowVector ? data.size() : 1)),
        view_rows_(vectorType == VectorType::RowVector ? 1 : data.size()),
        view_cols_(vectorType == VectorType::RowVector ? data.size() : 1) {
    for (size_t i = 0; i < data.size(); ++i) {
      if (vectorType == VectorType::RowVector) {
        operator()(0, i) = data[i];
      } else {
        operator()(i, 0) = data[i];
      }
    }
  }
  //constructor to create matrix from list of vectors
  Matrix(const std::vector<Matrix<T>>& vectors, Orientation orientation) {
    if (vectors.empty()) {
      view_rows_ = 0;
      view_cols_ = 0;
      storage_ = std::make_shared<StoragePolicy>(0, 0);
      return;
    }

    // Validate all vectors are compatible
    const size_t expected_dim =
        orientation == Orientation::Row ? vectors[0].cols() : vectors[0].rows();
    for (const auto& vec : vectors) {
      if (orientation == Orientation::Row &&
          (vec.rows() != 1 || vec.cols() != expected_dim)) {
        throw std::invalid_argument(
            "All vectors must be row vectors with same column count");
      }
      if (orientation == Orientation::Column &&
          (vec.cols() != 1 || vec.rows() != expected_dim)) {
        throw std::invalid_argument(
            "All vectors must be column vectors with same row count");
      }
    }

    // Create storage
    if (orientation == Orientation::Row) {
      view_rows_ = vectors.size();
      view_cols_ = expected_dim;
    } else {
      view_rows_ = expected_dim;
      view_cols_ = vectors.size();
    }
    storage_ = std::make_shared<StoragePolicy>(view_rows_, view_cols_);

    // Copy data
    for (size_t i = 0; i < vectors.size(); ++i) {
      if (orientation == Orientation::Row) {
        for (size_t j = 0; j < view_cols_; ++j) {
          operator()(i, j) = vectors[i](0, j);
        }
      } else {
        for (size_t j = 0; j < view_rows_; ++j) {
          operator()(j, i) = vectors[i](j, 0);
        }
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

  // // Copy assignment
  // Matrix& operator=(const Matrix& other) {
  //   if (this != &other) {
  //     storage_ = other.storage_->clone();
  //     view_row_start_ = other.view_row_start_;
  //     view_col_start_ = other.view_col_start_;
  //     view_rows_ = other.view_rows_;
  //     view_cols_ = other.view_cols_;
  //   }
  //   return *this;
  // }

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

  bool is_zero() const {
    const T epsilon = std::numeric_limits<T>::epsilon() * 100;
    for (size_t i = 0; i < rows(); i++) {
      for (size_t j = 0; j < cols(); j++) {
        if (std::abs(operator()(i, j)) > epsilon) {
          return false;
        }
      }
    }
    return true;
  }

  bool is_hessenberg(double tol) const {
    Matrix A = *this;
    size_t n = A.rows();
    for (size_t j = 0; j < n; ++j) {
      for (size_t i = j + 2; i < n; ++i) {
        if (std::abs(A(i, j)) > tol)
          return false;
      }
    }
    return true;
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

  // Manual insert and erase operations for the Matrix class
  void insert(const T& value) {
    // Insert at the end
    Matrix temp(view_rows_ + 1, view_cols_);
    for (size_t i = 0; i < view_rows_; i++) {
      for (size_t j = 0; j < view_cols_; j++) {
        temp(i, j) = operator()(i, j);
      }
    }
    for (size_t j = 0; j < view_cols_; j++) {
      temp(view_rows_, j) = value;
    }
    *this = temp;
  }

  template <typename ForwardIterator>
  void insert(ForwardIterator first, ForwardIterator last) {
    // Insert a range at the end
    size_t range_size = std::distance(first, last);
    size_t new_rows =
        range_size / view_cols_ + (range_size % view_cols_ > 0 ? 1 : 0);
    Matrix temp(view_rows_ + new_rows, view_cols_);

    // Copy existing data
    for (size_t i = 0; i < view_rows_; i++) {
      for (size_t j = 0; j < view_cols_; j++) {
        temp(i, j) = operator()(i, j);
      }
    }

    // Copy new data
    size_t idx = 0;
    for (auto it = first; it != last; ++it, ++idx) {
      size_t row = view_rows_ + idx / view_cols_;
      size_t col = idx % view_cols_;
      if (col < view_cols_) {
        temp(row, col) = *it;
      }
    }

    *this = temp;
  }

  // Insert at position
  template <typename InputIterator, typename ForwardIterator>
  void insert(InputIterator position, ForwardIterator first,
              ForwardIterator last) {
    // Create temporary vector of all data
    std::vector<T> all_data;
    all_data.reserve(view_rows_ * view_cols_ + std::distance(first, last));

    // Copy data before position
    for (auto it = begin(); it != position; ++it) {
      all_data.push_back(*it);
    }

    // Copy new data
    for (auto it = first; it != last; ++it) {
      all_data.push_back(*it);
    }

    // Copy data after position
    for (auto it = position; it != end(); ++it) {
      all_data.push_back(*it);
    }

    // Create new matrix with proper dimensions
    size_t new_size = all_data.size();
    size_t new_rows = new_size / view_cols_;
    Matrix temp(new_rows, view_cols_);
    for (size_t i = 0; i < new_size && i < new_rows * view_cols_; i++) {
      temp(i / view_cols_, i % view_cols_) = all_data[i];
    }

    *this = temp;
  }

  // Insert at position
  template <typename InputIterator>
  void insert(InputIterator position, const T& value) {
    // Create temporary vector of all data
    std::vector<T> all_data;
    all_data.reserve(view_rows_ * view_cols_ + 1);

    // Copy data before position
    for (auto it = begin(); it != position; ++it) {
      all_data.push_back(*it);
    }

    // Insert new value
    all_data.push_back(value);

    // Copy data after position
    for (auto it = position; it != end(); ++it) {
      all_data.push_back(*it);
    }

    // Create new matrix with proper dimensions
    size_t new_size = all_data.size();
    size_t new_rows = (new_size + view_cols_ - 1) / view_cols_;
    Matrix temp(new_rows, view_cols_);
    for (size_t i = 0; i < new_size; i++) {
      temp(i / view_cols_, i % view_cols_) = all_data[i];
    }

    *this = temp;
  }

  // Erase a single value
  iterator erase(iterator position) {
    size_t offset = position - begin();
    std::vector<T> all_data;
    all_data.reserve(view_rows_ * view_cols_ - 1);

    size_t idx = 0;
    for (auto it = begin(); it != end(); ++it, ++idx) {
      if (idx != offset) {
        all_data.push_back(*it);
      }
    }

    // Rebuild the matrix with new dimensions
    size_t new_rows = all_data.size() / view_cols_;
    if (all_data.size() % view_cols_ != 0)
      new_rows++;

    Matrix temp(new_rows, view_cols_);
    for (size_t i = 0; i < all_data.size(); i++) {
      temp(i / view_cols_, i % view_cols_) = all_data[i];
    }

    *this = temp;
    return begin() + offset;
  }

  // Erase a range of values
  iterator erase(iterator first, iterator last) {
    size_t start_offset = first - begin();
    size_t count = last - first;

    std::vector<T> all_data;
    all_data.reserve(view_rows_ * view_cols_ - count);

    size_t idx = 0;
    for (auto it = begin(); it != end(); ++it, ++idx) {
      if (idx < start_offset || idx >= start_offset + count) {
        all_data.push_back(*it);
      }
    }

    // Rebuild the matrix with new dimensions
    size_t new_rows = all_data.size() / view_cols_;
    if (all_data.size() % view_cols_ != 0)
      new_rows++;

    Matrix temp(new_rows, view_cols_);
    for (size_t i = 0; i < all_data.size(); i++) {
      temp(i / view_cols_, i % view_cols_) = all_data[i];
    }

    *this = temp;
    return begin() + start_offset;
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

  Matrix minor_matrix(const size_t row, const size_t column) const {
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
  //return a copy of the sected submatrix of existing  matrix
  Matrix sub_matrix(size_t row_start, size_t col_start, size_t rows,
                    size_t cols) const {
    if (row_start + rows > view_rows_ || col_start + cols > view_cols_)
      throw std::invalid_argument("Submatrix dimensions exceed matrix bounds");

    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        result(i, j) = operator()(row_start + i, col_start + j);
      }
    }
    return result;
  }
  //identical to set_block just method name change,which insert a submatrix to existing matrix
  void set_submatrix(size_t start_row, size_t start_col, const Matrix& sub) {
    if (start_row + sub.rows() > rows() || start_col + sub.cols() > cols()) {
      throw std::invalid_argument("Submatrix exceeds matrix dimensions");
    }
    auto target = view(start_row, start_col, sub.rows(), sub.cols());
    target = sub;  // Utilizes the assignment operator to copy elements
  }

  template <typename InputIterator, typename OutputIterator>
  void insert_submatrix(InputIterator srcBegin, InputIterator srcEnd,
                        OutputIterator destPos, size_t numRows, size_t numCols,
                        size_t position, size_t offset) {
    // Copy the values from source to destination with the appropriate offset
    for (size_t i = 0; i < numRows; ++i) {
      std::copy(srcBegin + i * numCols, srcBegin + (i + 1) * numCols,
                destPos + (position + i) * offset);
    }
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

  // =====  Hadamard Product with Block Processing =====
  Matrix hadamard(const Matrix& rhs) const {
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
              result(i, j) = operator()(i, j) * rhs(i, j);
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
              result(i, j) = operator()(i, j) * rhs(i, j);
            }
          }
        }
      }
    }

    return result;
  }
  Matrix Kronecker(const Matrix& rhs) const {
    const size_t a_rows = rows();
    const size_t a_cols = cols();
    const size_t b_rows = rhs.rows();
    const size_t b_cols = rhs.cols();

    Matrix result(a_rows * b_rows, a_cols * b_cols);

    if constexpr (ALGABER_OPENMP_ENABLED) {
#pragma omp parallel for collapse(2) num_threads(ThreadCount)
      for (size_t i = 0; i < a_rows; ++i) {
        for (size_t j = 0; j < a_cols; ++j) {
          const T a_val = operator()(i, j);
          for (size_t k = 0; k < b_rows; ++k) {
            for (size_t l = 0; l < b_cols; ++l) {
              result(i * b_rows + k, j * b_cols + l) = a_val * rhs(k, l);
            }
          }
        }
      }
    } else {
      for (size_t i = 0; i < a_rows; ++i) {
        for (size_t j = 0; j < a_cols; ++j) {
          const T a_val = operator()(i, j);
          for (size_t k = 0; k < b_rows; ++k) {
            for (size_t l = 0; l < b_cols; ++l) {
              result(i * b_rows + k, j * b_cols + l) = a_val * rhs(k, l);
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

  // Dot product (for vectors)-1
  T dot(const Matrix& rhs) const {
    // Check if either this or rhs is a vector (row or column)
    bool this_is_vector = view_rows_ == 1 || view_cols_ == 1;
    bool rhs_is_vector = rhs.view_rows_ == 1 || rhs.view_cols_ == 1;

    if (!this_is_vector || !rhs_is_vector)
      throw std::invalid_argument("Dot product requires at least one vector");

    size_t this_size = view_rows_ == 1 ? view_cols_ : view_rows_;
    size_t rhs_size = rhs.view_rows_ == 1 ? rhs.view_cols_ : rhs.view_rows_;

    if (this_size != rhs_size)
      throw std::invalid_argument(
          "Vectors must have same size for dot product");

    T result = T{0};

    for (size_t i = 0; i < this_size; ++i) {
      size_t this_row = view_rows_ == 1 ? 0 : i;
      size_t this_col = view_rows_ == 1 ? i : 0;
      size_t rhs_row = rhs.view_rows_ == 1 ? 0 : i;
      size_t rhs_col = rhs.view_rows_ == 1 ? i : 0;

      result += operator()(this_row, this_col) * rhs(rhs_row, rhs_col);
    }

    return result;
  }
  // Cross product (for 3D vectors only)-1
  Matrix cross(const Matrix& rhs) const {
    bool this_is_3d = (view_rows_ == 3 && view_cols_ == 1) ||
                      (view_rows_ == 1 && view_cols_ == 3);
    bool rhs_is_3d = (rhs.view_rows_ == 3 && rhs.view_cols_ == 1) ||
                     (rhs.view_rows_ == 1 && rhs.view_cols_ == 3);

    if (!this_is_3d || !rhs_is_3d)
      throw std::invalid_argument("Cross product requires 3D vectors");

    // Extract components
    T a1 = view_rows_ == 3 ? operator()(0, 0) : operator()(0, 0);
    T a2 = view_rows_ == 3 ? operator()(1, 0) : operator()(0, 1);
    T a3 = view_rows_ == 3 ? operator()(2, 0) : operator()(0, 2);

    T b1 = rhs.view_rows_ == 3 ? rhs(0, 0) : rhs(0, 0);
    T b2 = rhs.view_rows_ == 3 ? rhs(1, 0) : rhs(0, 1);
    T b3 = rhs.view_rows_ == 3 ? rhs(2, 0) : rhs(0, 2);

    // Compute cross product
    T c1 = a2 * b3 - a3 * b2;
    T c2 = a3 * b1 - a1 * b3;
    T c3 = a1 * b2 - a2 * b1;

    // Return result in the same orientation as this
    if (view_rows_ == 3) {
      return Matrix({{c1}, {c2}, {c3}});
    } else {
      return Matrix({{c1, c2, c3}});
    }
  }
  Matrix selectByMask(const Matrix<T>& bin, bool columns = false) {
    size_t dimension = columns ? view_cols_ : view_rows_;
    Matrix result;

    try {
      if (bin.cols() != 1)
        throw too_many_cols();
      if (bin.rows() > dimension)
        throw bad_size();

      for (size_t i = 0; i < bin.rows(); i++) {
        if (bin(i, 0)) {
          if (columns)
            result.addColumn(this->getColumn(i));
          else
            result.addRow(this->getRow(i));
        }
      }
    } catch (const std::invalid_argument& e) {
      std::cerr << "Error: " << e.what() << std::endl;

      throw;
    }

    return result;
  }

  // Extract a single row from the matrix
  Matrix getRow(size_t rowIndex) const {
    if (rowIndex >= view_rows_)
      throw row_outbound();

    Matrix result(static_cast<size_t>(1), static_cast<size_t>(view_cols_));
    for (size_t j = 0; j < view_cols_; ++j) {
      result(0, j) = operator()(rowIndex, j);
    }
    return result;
  }

  // Extract a single column from the matrix
  Matrix getColumn(size_t colIndex) const {
    if (colIndex >= view_cols_)
      throw col_outbound();

    Matrix result(static_cast<size_t>(view_rows_), static_cast<size_t>(1));
    for (size_t i = 0; i < view_rows_; ++i) {
      result(i, 0) = operator()(i, colIndex);
    }
    return result;
  }

  Matrix getRows(const Matrix bin) { return selectByMask(bin); }  //first

  Matrix getColumns(const Matrix bin) {
    return selectByMask(bin, true);
  }  //first

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
  // Matrix& operator-=(const Matrix& rhs) {
  //   if (view_rows_ != rhs.view_rows_ || view_cols_ != rhs.view_cols_)
  //     throw std::invalid_argument("Matrix dimensions mismatch");

  //   for (size_t i = 0; i < view_rows_; ++i)
  //     for (size_t j = 0; j < view_cols_; ++j)
  //       operator()(i, j) -= rhs(i, j);

  //   return *this;
  // }
  Matrix& operator-=(const Matrix& rhs) {
    if (view_rows_ != rhs.view_rows_ || view_cols_ != rhs.view_cols_)
      throw std::invalid_argument("Matrix dimensions mismatch for -=");

    const size_t block_size = BLOCK_SIZE;

    if constexpr (ALGABER_OPENMP_ENABLED) {
#pragma omp parallel for num_threads(ThreadCount) collapse(2)
      for (size_t i_outer = 0; i_outer < view_rows_; i_outer += block_size) {
        for (size_t j_outer = 0; j_outer < view_cols_; j_outer += block_size) {
          size_t i_end = std::min(i_outer + block_size, view_rows_);
          size_t j_end = std::min(j_outer + block_size, view_cols_);
          for (size_t i = i_outer; i < i_end; ++i) {
            for (size_t j = j_outer; j < j_end; ++j) {
              operator()(i, j) -= rhs(i, j);
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
              operator()(i, j) -= rhs(i, j);
            }
          }
        }
      }
    }

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

  Matrix& operator-=(const T& scalar) {
    const size_t block_size = BLOCK_SIZE;

    if constexpr (ALGABER_OPENMP_ENABLED) {
#pragma omp parallel for num_threads(ThreadCount) collapse(2)
      for (size_t i_outer = 0; i_outer < view_rows_; i_outer += block_size) {
        for (size_t j_outer = 0; j_outer < view_cols_; j_outer += block_size) {
          size_t i_end = std::min(i_outer + block_size, view_rows_);
          size_t j_end = std::min(j_outer + block_size, view_cols_);
          for (size_t i = i_outer; i < i_end; ++i) {
            for (size_t j = j_outer; j < j_end; ++j) {
              operator()(i, j) -= scalar;
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
              operator()(i, j) -= scalar;
            }
          }
        }
      }
    }

    return *this;
  }
  // In-place Hadamard product
  Matrix& hadamard_inplace(const Matrix& rhs) {
    if (view_rows_ != rhs.view_rows_ || view_cols_ != rhs.view_cols_)
      throw std::invalid_argument("Matrix dimensions mismatch");

    for (size_t i = 0; i < view_rows_; ++i)
      for (size_t j = 0; j < view_cols_; ++j)
        operator()(i, j) *= rhs(i, j);

    return *this;
  }

  Matrix& apply(std::function<T(T)> func) {
    const size_t block_size = BLOCK_SIZE;

    if constexpr (ALGABER_OPENMP_ENABLED) {
#pragma omp parallel for num_threads(ThreadCount) collapse(2)
      for (size_t i_outer = 0; i_outer < view_rows_; i_outer += block_size) {
        for (size_t j_outer = 0; j_outer < view_cols_; j_outer += block_size) {
          size_t i_end = std::min(i_outer + block_size, view_rows_);
          size_t j_end = std::min(j_outer + block_size, view_cols_);
          for (size_t i = i_outer; i < i_end; ++i) {
            for (size_t j = j_outer; j < j_end; ++j) {
              operator()(i, j) = func(operator()(i, j));
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
              operator()(i, j) = func(operator()(i, j));
            }
          }
        }
      }
    }
    return *this;
  }

  // Apply a function with indices to each element
  Matrix& apply_with_indices(std::function<T(T, size_t, size_t)> func) {
    for (size_t i = 0; i < view_rows_; ++i)
      for (size_t j = 0; j < view_cols_; ++j)
        operator()(i, j) = func(operator()(i, j), i, j);

    return *this;
  }

  Matrix map(std::function<T(T)> func) const {
    Matrix result(view_rows_, view_cols_);
    for (size_t i = 0; i < view_rows_; ++i)
      for (size_t j = 0; j < view_cols_; ++j)
        result(i, j) = func(operator()(i, j));

    return result;
  }

  Matrix map_with_indices(std::function<T(T, size_t, size_t)> func) const {
    Matrix result(view_rows_, view_cols_);
    const size_t block_size = BLOCK_SIZE;

    if constexpr (ALGABER_OPENMP_ENABLED) {
#pragma omp parallel for num_threads(ThreadCount) collapse(2)
      for (size_t i_outer = 0; i_outer < view_rows_; i_outer += block_size) {
        for (size_t j_outer = 0; j_outer < view_cols_; j_outer += block_size) {
          size_t i_end = std::min(i_outer + block_size, view_rows_);
          size_t j_end = std::min(j_outer + block_size, view_cols_);
          for (size_t i = i_outer; i < i_end; ++i) {
            for (size_t j = j_outer; j < j_end; ++j) {
              result(i, j) = func(operator()(i, j), i, j);
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
              result(i, j) = func(operator()(i, j), i, j);
            }
          }
        }
      }
    }
    return result;
  }

  // Get raw pointer to data (for interop with other libraries)
  T* data() noexcept {
    if (view_row_start_ == 0 && view_col_start_ == 0 &&
        view_rows_ == storage_->rows() && view_cols_ == storage_->cols())
      return storage_->data();
    else
      return nullptr;  // Cannot get contiguous data for view
  }

  const T* data() const noexcept {
    if (view_row_start_ == 0 && view_col_start_ == 0 &&
        view_rows_ == storage_->rows() && view_cols_ == storage_->cols())
      return storage_->data();
    else
      return nullptr;  // Cannot get contiguous data for view
  }

  // Sum of all elements
  T sum() const { return std::accumulate(begin(), end(), T{0}); }

  // Mean of all elements
  T mean() const {
    if (empty())
      throw std::invalid_argument("Cannot compute mean of empty matrix");
    return sum() / static_cast<T>(size());
  }

  // Min element
  T min() const {
    if (empty())
      throw std::invalid_argument("Cannot compute min of empty matrix");
    return *std::min_element(begin(), end());
  }

  // Max element
  T max() const {
    if (empty())
      throw std::invalid_argument("Cannot compute max of empty matrix");
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
  T norm() const { return frobenius_norm(); }
  //Typical getters-1
  void validateIndexes(size_t row, size_t col) const {
    if (row >= view_rows_ || col >= view_cols_)
      throw std::out_of_range("Index out of bounds");
  }
  // // Direct permutation application (optimized version)
  // void apply_row_permutation(const PermutationMatrix& P) {
  //   for (size_t i = 0; i < rows(); ++i) {
  //     const size_t target_row = P[i];
  //     if (target_row != i) {
  //       swap_rows(i, target_row);
  //     }
  //   }
  // }

  // void apply_col_permutation(const PermutationMatrix& Q) {
  //   for (size_t j = 0; j < cols(); ++j) {
  //     const size_t target_col = Q[j];
  //     if (target_col != j) {
  //       swap_columns(j, target_col);
  //     }
  //   }
  // }

  void validateInsertIndexes(size_t index, size_t insert_dim, size_t other_dim,
                             Orientation orientation) const {
    if (orientation == Orientation::Row) {
      if (insert_dim != 1) {
        throw std::invalid_argument("Too many rows provided");
      }
      if (other_dim != view_cols_)
        throw std::invalid_argument(
            "Incorrect number of columns for row insertion");
      if (index > view_rows_)
        throw std::out_of_range("Row index out of bounds");
    } else if (orientation == Orientation::Column) {
      if (insert_dim != 1) {
        throw std::invalid_argument("Too many columns provided");
      }
      if (other_dim != view_rows_)
        throw std::invalid_argument(
            "Incorrect number of rows for column insertion");
      if (index > view_cols_)
        throw std::out_of_range("Column index out of bounds");
    }
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

  void addRow(const Matrix& values, size_t position) {
    try {
      if (empty()) {
        view_rows_ = 1;  // Initialize to 1 row initially
        view_cols_ = values.cols();
        return;
      }

      validateInsertIndexes(position, values.cols(), values.rows(),
                            Orientation::Row);

      // Create a new matrix with the row inserted
      Matrix result(view_rows_ + 1, view_cols_);

      // Copy rows before the insertion point
      for (size_t i = 0; i < position; i++) {
        for (size_t j = 0; j < view_cols_; j++) {
          result(i, j) = operator()(i, j);
        }
      }

      // Insert the new row
      for (size_t j = 0; j < view_cols_; j++) {
        result(position, j) = values(0, j);
      }

      // Copy rows after the insertion point
      for (size_t i = position; i < view_rows_; i++) {
        for (size_t j = 0; j < view_cols_; j++) {
          result(i + 1, j) = operator()(i, j);
        }
      }

      // Update this matrix
      *this = result;
    } catch (const std::exception& e) {
      std::cerr << "Error: " << e.what() << std::endl;
      return;
    }
  }
  void addRow(std::initializer_list<T> rowValues, size_t position) {
    // Validate input
    try {
      if (empty()) {
        view_rows_ = 1;  // Initialize to 1 row initially
        view_cols_ = rowValues.size();

        // Create a new matrix with the single row
        Matrix result(1, view_cols_);
        size_t j = 0;
        for (auto val : rowValues) {
          result(0, j++) = val;
        }
        *this = result;
        return;
      } else {
        validateInsertIndexes(position, 1, view_cols_, Orientation::Row);

        // Create a new matrix with the row inserted
        Matrix result(view_rows_ + 1, view_cols_);

        // Copy rows before the insertion point
        for (size_t i = 0; i < position; i++) {
          for (size_t j = 0; j < view_cols_; j++) {
            result(i, j) = operator()(i, j);
          }
        }

        // Insert the new row
        size_t j = 0;
        for (auto val : rowValues) {
          if (j < view_cols_) {
            result(position, j++) = val;
          }
        }

        // Copy rows after the insertion point
        for (size_t i = position; i < view_rows_; i++) {
          for (size_t j = 0; j < view_cols_; j++) {
            result(i + 1, j) = operator()(i, j);
          }
        }

        // Update this matrix
        *this = result;
      }
    } catch (const std::invalid_argument& e) {
      std::cerr << "Error: " << e.what() << std::endl;
      return;
    }
  }
  void addColumn(const Matrix& values, size_t position) {
    try {
      if (empty()) {
        view_rows_ = values.rows();
        view_cols_ = 1;

        // Create a new matrix with the single column
        Matrix result(view_rows_, static_cast<size_t>(1));
        for (size_t i = 0; i < view_rows_; i++) {
          result(i, 0) = values(i, 0);
        }
        *this = result;
        return;
      }

      validateInsertIndexes(position, values.view_cols_, values.view_rows_,
                            Orientation::Column);

      // Create a new matrix with the column inserted
      Matrix result(static_cast<size_t>(view_rows_),
                    static_cast<size_t>(view_cols_ + 1));

      // Copy columns before the insertion point
      for (size_t i = 0; i < view_rows_; i++) {
        for (size_t j = 0; j < position; j++) {
          result(i, j) = operator()(i, j);
        }
      }

      // Insert the new column
      for (size_t i = 0; i < view_rows_; i++) {
        result(i, position) = values(i, 0);
      }

      // Copy columns after the insertion point
      for (size_t i = 0; i < view_rows_; i++) {
        for (size_t j = position; j < view_cols_; j++) {
          result(i, j + 1) = operator()(i, j);
        }
      }

      // Update this matrix
      *this = result;
    } catch (const std::exception& e) {
      std::cerr << "Error: " << e.what() << std::endl;
      return;
    }
  }

  void addColumn(std::initializer_list<T> colValues, size_t position) {
    // Validate input
    try {
      if (empty()) {
        view_rows_ = colValues.size();
        view_cols_ = 1;  // Initialize to 1 column initially

        // Create a new matrix with the single column
        Matrix result(view_rows_, static_cast<size_t>(1));
        size_t i = 0;
        for (auto val : colValues) {
          result(i++, 0) = val;
        }
        *this = result;
        return;
      } else {
        validateInsertIndexes(position, 1, view_rows_, Orientation::Column);

        // Create a new matrix with the column inserted
        Matrix result(static_cast<size_t>(view_rows_),
                      static_cast<size_t>(view_cols_ + 1));

        // Copy columns before the insertion point
        for (size_t i = 0; i < view_rows_; i++) {
          for (size_t j = 0; j < position; j++) {
            result(i, j) = operator()(i, j);
          }
        }

        // Insert the new column
        for (size_t i = 0; i < view_rows_ && i < colValues.size(); i++) {
          result(i, position) = *(colValues.begin() + i);
        }

        // Copy columns after the insertion point
        for (size_t i = 0; i < view_rows_; i++) {
          for (size_t j = position; j < view_cols_; j++) {
            result(i, j + 1) = operator()(i, j);
          }
        }

        // Update this matrix
        *this = result;
      }
    } catch (const std::invalid_argument& e) {
      std::cerr << "Error: " << e.what() << std::endl;
      return;
    }
  }

  void addRow(Matrix values) { addRow(values, view_rows_); }

  void addRow(std::initializer_list<T> rowValues) {
    addRow(rowValues, view_rows_);
  }

  void addColumn(Matrix values) { addColumn(values, view_cols_); }

  void addColumn(std::initializer_list<T> colValues) {
    addColumn(colValues, view_cols_);
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

  Matrix& removeRow(size_t position) {
    if (position >= view_rows_)
      throw std::out_of_range("Row index out of bounds");

    // Create a new matrix without the specified row
    Matrix result(view_rows_ - 1, view_cols_);

    // Copy rows before the position
    for (size_t i = 0; i < position; i++) {
      for (size_t j = 0; j < view_cols_; j++) {
        result(i, j) = operator()(i, j);
      }
    }

    // Copy rows after the position
    for (size_t i = position + 1; i < view_rows_; i++) {
      for (size_t j = 0; j < view_cols_; j++) {
        result(i - 1, j) = operator()(i, j);
      }
    }

    *this = result;
    return *this;
  }

  Matrix& removeColumn(size_t position) {
    if (position >= view_cols_)
      throw std::out_of_range("Column index out of bounds");

    // Create a new matrix without the specified column
    Matrix result(view_rows_, view_cols_ - 1);

    // Copy columns before the position
    for (size_t i = 0; i < view_rows_; i++) {
      for (size_t j = 0; j < position; j++) {
        result(i, j) = operator()(i, j);
      }
    }

    // Copy columns after the position
    for (size_t i = 0; i < view_rows_; i++) {
      for (size_t j = position + 1; j < view_cols_; j++) {
        result(i, j - 1) = operator()(i, j);
      }
    }

    *this = result;
    return *this;
  }

  template <typename U>
  Matrix<U> fill(U value) const {
    Matrix<U> filledMatrix(view_rows_, view_cols_);
    if constexpr (ALGABER_OPENMP_ENABLED) {
#pragma omp parallel for collapse(2) num_threads(ThreadCount)
      for (size_t i = 0; i < view_rows_; ++i)
        for (size_t j = 0; j < view_cols_; ++j)
          filledMatrix(i, j) = value;
    } else {
      for (size_t i = 0; i < view_rows_; ++i)
        for (size_t j = 0; j < view_cols_; ++j)
          filledMatrix(i, j) = value;
    }
    return filledMatrix;
  }

  template <typename U>
  Matrix<U> fill(U value) {
    if constexpr (ALGABER_OPENMP_ENABLED) {
#pragma omp parallel for collapse(2) num_threads(ThreadCount)
      for (size_t i = 0; i < view_rows_; ++i)
        for (size_t j = 0; j < view_cols_; ++j)
          operator()(i, j) = value;
    } else {
      for (size_t i = 0; i < view_rows_; ++i)
        for (size_t j = 0; j < view_cols_; ++j)
          operator()(i, j) = value;
    }

    return *this;
  }

  // Reshape matrix (in-place if possible)
  Matrix reshape(size_t new_rows, size_t new_cols) const {
    if (new_rows * new_cols != view_rows_ * view_cols_)
      throw std::invalid_argument("New dimensions must preserve element count");

    // If this is a full matrix (not a view), we can efficiently reshape
    if (view_row_start_ == 0 && view_col_start_ == 0 &&
        view_rows_ == storage_->rows() && view_cols_ == storage_->cols()) {
      Matrix result(storage_, 0, 0, new_rows, new_cols);
      return result;
    } else {
      // For views, we need to create a new matrix
      Matrix result(new_rows, new_cols);
      auto it_this = begin();
      auto it_result = result.begin();

      for (; it_this != end(); ++it_this, ++it_result) {
        *it_result = *it_this;
      }

      return result;
    }
  }

  // Flatten matrix to a row vector
  Matrix flatten() const { return reshape(1, view_rows_ * view_cols_); }

  Matrix to_ones() { return fill(T(1)); }

  Matrix to_zeros() { return fill(T(0)); }

  Matrix to_diagonal(T value) {
    size_t size = std::min(view_rows_, view_cols_);
    Matrix result(size, size);
    for (size_t i = 0; i < size; i++)
      result(i, i) = value;
    return result;
  }

  Matrix to_identity() { return to_diagonal(1); }

  static Matrix zeros(size_t rows, size_t cols) {
    return Matrix(rows, cols, T{0});
  }

  // ===== Static Factory Methods =====

  static Matrix ones(size_t rows, size_t cols) {
    return Matrix(rows, cols, T{1});
  }

  static Matrix identity(size_t n) {
    Matrix result(n, n, T{0});
    for (size_t i = 0; i < n; ++i)
      result(i, i) = T{1};
    return result;
  }
  static Matrix diagonal(const std::vector<T>& elements) {
    size_t dimension = elements.size();
    Matrix result(dimension, dimension);
    for (size_t i = 0; i < dimension; ++i) {
      result(i, i) = elements[i];
    }
    return result;
  }
  static Matrix unit_vector(size_t n, size_t i) {
    Matrix<T, StoragePolicy> vec(n, 1, T(0));
    vec(i, 0) = T(1);
    return vec;
  }

  static Matrix linspace(T start, T end, size_t count) {
    Matrix result(1, count);

    if (count <= 1) {
      if (count == 1)
        result(0, 0) = start;
      return result;
    }

    T step = (end - start) / static_cast<T>(count - 1);
    for (size_t i = 0; i < count; ++i) {
      result(0, i) = start + static_cast<T>(i) * step;
    }

    return result;
  }

  static Matrix arange(T start, T end, T step = T{1}) {
    if (step == T{0})
      throw std::invalid_argument("Step cannot be zero");

    size_t count = static_cast<size_t>(std::ceil((end - start) / step));
    Matrix result(1, count);

    for (size_t i = 0; i < count; ++i) {
      result(0, i) = start + static_cast<T>(i) * step;
    }

    return result;
  }

  static Matrix random(size_t rows, size_t cols, T min_val = T{0},
                       T max_val = T{1}) {
    std::random_device rd;
    std::mt19937 gen(rd());

    Matrix result(rows, cols);

    if constexpr (std::is_floating_point_v<T>) {
      if constexpr (ALGABER_OPENMP_ENABLED) {
#pragma omp parallel num_threads(ThreadCount)
        {
          // Each thread needs its own random generator to avoid contention
          std::mt19937 local_gen(rd() + omp_get_thread_num());
          std::uniform_real_distribution<T> dist(min_val, max_val);

#pragma omp for collapse(2)
          for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
              result(i, j) = dist(local_gen);
            }
          }
        }
      } else {
        std::uniform_real_distribution<T> dist(min_val, max_val);
        for (size_t i = 0; i < rows; ++i) {
          for (size_t j = 0; j < cols; ++j) {
            result(i, j) = dist(gen);
          }
        }
      }
    } else {
      if constexpr (ALGABER_OPENMP_ENABLED) {
#pragma omp parallel num_threads(ThreadCount)
        {
          // Each thread needs its own random generator
          std::mt19937 local_gen(rd() + omp_get_thread_num());
          std::uniform_int_distribution<int> dist(static_cast<int>(min_val),
                                                  static_cast<int>(max_val));

#pragma omp for collapse(2)
          for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
              result(i, j) = static_cast<T>(dist(local_gen));
            }
          }
        }
      } else {
        std::uniform_int_distribution<int> dist(static_cast<int>(min_val),
                                                static_cast<int>(max_val));
        for (size_t i = 0; i < rows; ++i) {
          for (size_t j = 0; j < cols; ++j) {
            result(i, j) = static_cast<T>(dist(gen));
          }
        }
      }
    }

    return result;
  }

  //Application Based functions (IMAGE PROCESSING)

  // Convolution
  Matrix convolve(const Matrix& kernel) const {
    const int k_rows = static_cast<int>(kernel.rows());
    const int k_cols = static_cast<int>(kernel.cols());
    const int pad_rows = k_rows / 2;
    const int pad_cols = k_cols / 2;

    Matrix result(view_rows_, view_cols_);

    if constexpr (ALGABER_OPENMP_ENABLED) {
#pragma omp parallel for collapse(2) num_threads(ThreadCount)
      for (size_t i = 0; i < view_rows_; ++i) {
        for (size_t j = 0; j < view_cols_; ++j) {
          T sum = T{0};

          for (int di = -pad_rows; di <= pad_rows; ++di) {
            for (int dj = -pad_cols; dj <= pad_cols; ++dj) {
              const int ri = static_cast<int>(i) + di;
              const int rj = static_cast<int>(j) + dj;

              // Handle boundary conditions (zero padding)
              if (ri >= 0 && ri < static_cast<int>(view_rows_) && rj >= 0 &&
                  rj < static_cast<int>(view_cols_)) {
                sum += operator()(ri, rj) *
                       kernel(di + pad_rows, dj + pad_cols);
              }
            }
          }

          result(i, j) = sum;
        }
      }
    } else {
      for (size_t i = 0; i < view_rows_; ++i) {
        for (size_t j = 0; j < view_cols_; ++j) {
          T sum = T{0};

          for (int di = -pad_rows; di <= pad_rows; ++di) {
            for (int dj = -pad_cols; dj <= pad_cols; ++dj) {
              const int ri = static_cast<int>(i) + di;
              const int rj = static_cast<int>(j) + dj;

              // Handle boundary conditions (zero padding)
              if (ri >= 0 && ri < static_cast<int>(view_rows_) && rj >= 0 &&
                  rj < static_cast<int>(view_cols_)) {
                sum += operator()(ri, rj) *
                       kernel(di + pad_rows, dj + pad_cols);
              }
            }
          }

          result(i, j) = sum;
        }
      }
    }

    return result;
  }

  Matrix filter(const Matrix& kernel,
                BorderType border = BorderType::ZERO) const {
    const int k_rows = static_cast<int>(kernel.rows());
    const int k_cols = static_cast<int>(kernel.cols());
    const int pad_rows = k_rows / 2;
    const int pad_cols = k_cols / 2;

    Matrix result(view_rows_, view_cols_);

    if constexpr (ALGABER_OPENMP_ENABLED) {
#pragma omp parallel for collapse(2) num_threads(ThreadCount)
      for (size_t i = 0; i < view_rows_; ++i) {
        for (size_t j = 0; j < view_cols_; ++j) {
          T sum = T{0};

          for (int di = -pad_rows; di <= pad_rows; ++di) {
            for (int dj = -pad_cols; dj <= pad_cols; ++dj) {
              int ri = static_cast<int>(i) + di;
              int rj = static_cast<int>(j) + dj;

              // Handle boundary conditions
              switch (border) {
                case BorderType::ZERO:
                  if (ri < 0 || ri >= static_cast<int>(view_rows_) || rj < 0 ||
                      rj >= static_cast<int>(view_cols_))
                    continue;  // Zero contribution
                  break;

                case BorderType::REFLECT:
                  if (ri < 0)
                    ri = -ri;
                  if (rj < 0)
                    rj = -rj;
                  if (ri >= static_cast<int>(view_rows_))
                    ri = 2 * static_cast<int>(view_rows_) - ri - 1;
                  if (rj >= static_cast<int>(view_cols_))
                    rj = 2 * static_cast<int>(view_cols_) - rj - 1;
                  break;

                case BorderType::REPLICATE:
                  ri = std::max(0,
                                std::min(ri, static_cast<int>(view_rows_) - 1));
                  rj = std::max(0,
                                std::min(rj, static_cast<int>(view_cols_) - 1));
                  break;

                case BorderType::WRAP:
                  ri = (ri + static_cast<int>(view_rows_)) %
                       static_cast<int>(view_rows_);
                  rj = (rj + static_cast<int>(view_cols_)) %
                       static_cast<int>(view_cols_);
                  break;
              }

              sum += operator()(ri, rj) * kernel(di + pad_rows, dj + pad_cols);
            }
          }

          result(i, j) = sum;
        }
      }
    } else {
      for (size_t i = 0; i < view_rows_; ++i) {
        for (size_t j = 0; j < view_cols_; ++j) {
          T sum = T{0};

          for (int di = -pad_rows; di <= pad_rows; ++di) {
            for (int dj = -pad_cols; dj <= pad_cols; ++dj) {
              int ri = static_cast<int>(i) + di;
              int rj = static_cast<int>(j) + dj;

              // Handle boundary conditions
              switch (border) {
                case BorderType::ZERO:
                  if (ri < 0 || ri >= static_cast<int>(view_rows_) || rj < 0 ||
                      rj >= static_cast<int>(view_cols_))
                    continue;  // Zero contribution
                  break;

                case BorderType::REFLECT:
                  if (ri < 0)
                    ri = -ri;
                  if (rj < 0)
                    rj = -rj;
                  if (ri >= static_cast<int>(view_rows_))
                    ri = 2 * static_cast<int>(view_rows_) - ri - 1;
                  if (rj >= static_cast<int>(view_cols_))
                    rj = 2 * static_cast<int>(view_cols_) - rj - 1;
                  break;

                case BorderType::REPLICATE:
                  ri = std::max(0,
                                std::min(ri, static_cast<int>(view_rows_) - 1));
                  rj = std::max(0,
                                std::min(rj, static_cast<int>(view_cols_) - 1));
                  break;

                case BorderType::WRAP:
                  ri = (ri + static_cast<int>(view_rows_)) %
                       static_cast<int>(view_rows_);
                  rj = (rj + static_cast<int>(view_cols_)) %
                       static_cast<int>(view_cols_);
                  break;
              }

              sum += operator()(ri, rj) * kernel(di + pad_rows, dj + pad_cols);
            }
          }

          result(i, j) = sum;
        }
      }
    }

    return result;
  }
  // Common image processing kernels
  static Matrix gaussian_kernel(int size, double sigma) {
    if (size % 2 == 0)
      size++;  // Ensure odd size

    Matrix kernel(size, size);
    int center = size / 2;
    double sum = 0.0;

    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        int x = i - center;
        int y = j - center;
        double value = std::exp(-(x * x + y * y) / (2.0 * sigma * sigma));
        kernel(i, j) = static_cast<T>(value);
        sum += value;
      }
    }

    // Normalize
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        kernel(i, j) = static_cast<T>(kernel(i, j) / sum);
      }
    }

    return kernel;
  }

  static Matrix sobel_x() {
    return Matrix<T>({{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}});
  }

  static Matrix sobel_y() {
    return Matrix<T>({{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}});
  }

  static Matrix laplacian() {
    return Matrix<T>({{0, 1, 0}, {1, -4, 1}, {0, 1, 0}});
  }

  // Image transformations
  Matrix resize(size_t new_rows, size_t new_cols,
                bool use_bilinear = true) const {
    Matrix result(new_rows, new_cols);

    const double scale_y = static_cast<double>(view_rows_) / new_rows;
    const double scale_x = static_cast<double>(view_cols_) / new_cols;

    if (use_bilinear) {
      // Bilinear interpolation
      for (size_t i = 0; i < new_rows; ++i) {
        for (size_t j = 0; j < new_cols; ++j) {
          double src_y = i * scale_y;
          double src_x = j * scale_x;

          size_t y1 = static_cast<size_t>(src_y);
          size_t x1 = static_cast<size_t>(src_x);
          size_t y2 = std::min(y1 + 1, view_rows_ - 1);
          size_t x2 = std::min(x1 + 1, view_cols_ - 1);

          double dy = src_y - y1;
          double dx = src_x - x1;

          result(i, j) =
              static_cast<T>((1 - dy) * (1 - dx) * operator()(y1, x1) +
                             dy * (1 - dx) * operator()(y2, x1) +
                             (1 - dy) * dx * operator()(y1, x2) +
                             dy * dx * operator()(y2, x2));
        }
      }
    } else {
      // Nearest neighbor interpolation
      for (size_t i = 0; i < new_rows; ++i) {
        for (size_t j = 0; j < new_cols; ++j) {
          size_t src_y = static_cast<size_t>(i * scale_y);
          size_t src_x = static_cast<size_t>(j * scale_x);

          src_y = std::min(src_y, view_rows_ - 1);
          src_x = std::min(src_x, view_cols_ - 1);

          result(i, j) = operator()(src_y, src_x);
        }
      }
    }

    return result;
  }

  // Rotate image (0, 90, 180, 270 degrees)
  Matrix rotate90(int times = 1) const {
    times = ((times % 4) + 4) % 4;  // Normalize to [0,3]

    if (times == 0)
      return *this;

    if (times == 2) {
      // 180 degrees - flip both dimensions
      Matrix result(view_rows_, view_cols_);
      for (size_t i = 0; i < view_rows_; ++i) {
        const size_t src_i = view_rows_ - 1 - i;
        for (size_t j = 0; j < view_cols_; ++j) {
          const size_t src_j = view_cols_ - 1 - j;
          result(i, j) = operator()(src_i, src_j);
        }
      }
      return result;
    }

    // 90 or 270 degrees - dimensions swap
    Matrix result(view_cols_, view_rows_);

    if (times == 1) {
      // 90 degrees clockwise
      for (size_t i = 0; i < view_rows_; ++i) {
        for (size_t j = 0; j < view_cols_; ++j) {
          result(j, view_rows_ - 1 - i) = operator()(i, j);
        }
      }
    } else {  // times == 3
      // 270 degrees clockwise (or 90 counter-clockwise)
      for (size_t i = 0; i < view_rows_; ++i) {
        for (size_t j = 0; j < view_cols_; ++j) {
          result(view_cols_ - 1 - j, i) = operator()(i, j);
        }
      }
    }

    return result;
  }

  // Threshold the image
  Matrix threshold(T threshold_value, T max_value = T{255}) const {
    Matrix result(view_rows_, view_cols_);
    for (size_t i = 0; i < view_rows_; ++i) {
      for (size_t j = 0; j < view_cols_; ++j) {
        result(i, j) = operator()(i, j) > threshold_value ? max_value : T{0};
      }
    }
    return result;
  }

  // Normalize values to range [0, 1] or [0, max_val]
  Matrix normalize(T max_val = T{1}) const {
    if (empty())
      return *this;

    T min_val = min();
    T max_val_current = max();
    T range = max_val_current - min_val;

    if (range == T{0}) {
      return Matrix(view_rows_, view_cols_, min_val == T{0} ? T{0} : max_val);
    }

    Matrix result(view_rows_, view_cols_);
    for (size_t i = 0; i < view_rows_; ++i) {
      for (size_t j = 0; j < view_cols_; ++j) {
        result(i, j) = (operator()(i, j) - min_val) / range * max_val;
      }
    }

    return result;
  }

  // Compute histogram
  std::vector<size_t> histogram(size_t bins = 256) const {
    std::vector<size_t> hist(bins, 0);

    if (empty() || bins == 0)
      return hist;

    T min_val = min();
    T max_val = max();
    T range = max_val - min_val;

    if (range == T{0}) {
      // All elements are the same
      size_t bin = (min_val == max_val && min_val == T{0}) ? 0 : bins - 1;
      hist[bin] = size();
      return hist;
    }

    for (size_t i = 0; i < view_rows_; ++i) {
      for (size_t j = 0; j < view_cols_; ++j) {
        T normalized = (operator()(i, j) - min_val) / range;
        size_t bin = std::min(static_cast<size_t>(normalized * bins), bins - 1);
        hist[bin]++;
      }
    }

    return hist;
  }

  // Extract channel from multi-channel image (assuming channels are interleaved)
  Matrix extract_channel(size_t channel_idx, size_t total_channels) const {
    if (view_cols_ % total_channels != 0)
      throw std::invalid_argument(
          "Column count must be multiple of channel count");

    size_t new_cols = view_cols_ / total_channels;
    Matrix result(view_rows_, new_cols);

    for (size_t i = 0; i < view_rows_; ++i) {
      for (size_t j = 0; j < new_cols; ++j) {
        result(i, j) = operator()(i, j * total_channels + channel_idx);
      }
    }

    return result;
  }

  // ===== Image Processing Function Implementations =====

  Matrix medianFilter(int kernel_size) const {
    if (kernel_size <= 0)
      throw std::invalid_argument("Kernel size must be positive");

    int adjusted_size = kernel_size % 2 == 0 ? kernel_size + 1 : kernel_size;
    const int radius = adjusted_size / 2;

    Matrix result(view_rows_, view_cols_);

    for (size_t i = 0; i < view_rows_; ++i) {
      for (size_t j = 0; j < view_cols_; ++j) {
        std::vector<T> neighbors;

        for (int di = -radius; di <= radius; ++di) {
          for (int dj = -radius; dj <= radius; ++dj) {
            int x = std::clamp(static_cast<int>(i) + di, 0,
                               static_cast<int>(view_rows_) - 1);
            int y = std::clamp(static_cast<int>(j) + dj, 0,
                               static_cast<int>(view_cols_) - 1);
            neighbors.push_back(operator()(x, y));
          }
        }

        std::sort(neighbors.begin(), neighbors.end());
        result(i, j) = neighbors[neighbors.size() / 2];
      }
    }
    return result;
  }

  Matrix bilateralFilter(double spatial_sigma, double range_sigma,
                         int size) const {
    if (spatial_sigma <= 0 || range_sigma <= 0)
      throw std::invalid_argument("Sigmas must be positive");

    int adjusted_size = size % 2 == 0 ? size + 1 : size;
    const int radius = adjusted_size / 2;

    Matrix result(view_rows_, view_cols_);

    for (size_t i = 0; i < view_rows_; ++i) {
      for (size_t j = 0; j < view_cols_; ++j) {
        const T center = operator()(i, j);
        double total = 0.0;
        double sum = 0.0;

        for (int di = -radius; di <= radius; ++di) {
          for (int dj = -radius; dj <= radius; ++dj) {
            int x = std::clamp(static_cast<int>(i) + di, 0,
                               static_cast<int>(view_rows_) - 1);
            int y = std::clamp(static_cast<int>(j) + dj, 0,
                               static_cast<int>(view_cols_) - 1);

            const T pixel = operator()(x, y);
            const double spatial =
                exp(-(di * di + dj * dj) / (2 * spatial_sigma * spatial_sigma));
            const double range =
                exp(-pow(pixel - center, 2) / (2 * range_sigma * range_sigma));
            const double weight = spatial * range;

            sum += static_cast<double>(pixel) * weight;
            total += weight;
          }
        }

        result(i, j) = static_cast<T>(total != 0 ? sum / total : center);
      }
    }
    return result;
  }

  Matrix adaptiveThreshold(int block_size, T c) const {
    if (block_size <= 0)
      throw std::invalid_argument("Block size must be positive");

    int adjusted_size = block_size % 2 == 0 ? block_size + 1 : block_size;
    const int radius = adjusted_size / 2;
    constexpr T max_val = static_cast<T>(255);

    Matrix result(view_rows_, view_cols_);

    for (size_t i = 0; i < view_rows_; ++i) {
      for (size_t j = 0; j < view_cols_; ++j) {
        T sum = T{0};
        int count = 0;

        for (int di = -radius; di <= radius; ++di) {
          for (int dj = -radius; dj <= radius; ++dj) {
            int x = std::clamp(static_cast<int>(i) + di, 0,
                               static_cast<int>(view_rows_) - 1);
            int y = std::clamp(static_cast<int>(j) + dj, 0,
                               static_cast<int>(view_cols_) - 1);
            sum += operator()(x, y);
            count++;
          }
        }

        T threshold = (sum / count) - c;
        result(i, j) = operator()(i, j) > threshold ? max_val : T{0};
      }
    }
    return result;
  }

  Matrix erode(const Matrix& kernel) const {
    const int k_rad_r = kernel.rows() / 2;
    const int k_rad_c = kernel.cols() / 2;

    Matrix result(view_rows_, view_cols_);

    for (size_t i = 0; i < view_rows_; ++i) {
      for (size_t j = 0; j < view_cols_; ++j) {
        T min_val = std::numeric_limits<T>::max();

        for (int di = -k_rad_r; di <= k_rad_r; ++di) {
          for (int dj = -k_rad_c; dj <= k_rad_c; ++dj) {
            if (kernel(di + k_rad_r, dj + k_rad_c) == T{0})
              continue;

            int x = std::clamp(static_cast<int>(i) + di, 0,
                               static_cast<int>(view_rows_) - 1);
            int y = std::clamp(static_cast<int>(j) + dj, 0,
                               static_cast<int>(view_cols_) - 1);

            min_val = std::min(min_val, operator()(x, y));
          }
        }
        result(i, j) = min_val;
      }
    }
    return result;
  }

  Matrix dilate(const Matrix& kernel) const {
    const int k_rad_r = kernel.rows() / 2;
    const int k_rad_c = kernel.cols() / 2;

    Matrix result(view_rows_, view_cols_);

    for (size_t i = 0; i < view_rows_; ++i) {
      for (size_t j = 0; j < view_cols_; ++j) {
        T max_val = std::numeric_limits<T>::lowest();

        for (int di = -k_rad_r; di <= k_rad_r; ++di) {
          for (int dj = -k_rad_c; dj <= k_rad_c; ++dj) {
            if (kernel(di + k_rad_r, dj + k_rad_c) == T{0})
              continue;

            int x = std::clamp(static_cast<int>(i) + di, 0,
                               static_cast<int>(view_rows_) - 1);
            int y = std::clamp(static_cast<int>(j) + dj, 0,
                               static_cast<int>(view_cols_) - 1);

            max_val = std::max(max_val, operator()(x, y));
          }
        }
        result(i, j) = max_val;
      }
    }
    return result;
  }

  std::vector<Matrix> splitChannels(size_t num_channels) const {
    if (view_cols_ % num_channels != 0)
      throw std::invalid_argument("Columns must be divisible by channel count");

    const size_t cols_per = view_cols_ / num_channels;
    std::vector<Matrix> channels;

    for (size_t c = 0; c < num_channels; ++c) {
      Matrix channel(view_rows_, cols_per);
      for (size_t i = 0; i < view_rows_; ++i) {
        for (size_t j = 0; j < cols_per; ++j) {
          channel(i, j) = operator()(i, j * num_channels + c);
        }
      }
      channels.push_back(std::move(channel));
    }
    return channels;
  }

  Matrix mergeChannels(const std::vector<Matrix>& channels) {
    if (channels.empty())
      return Matrix();

    const size_t num = channels.size();
    const size_t rows = channels[0].rows();
    const size_t cols = channels[0].cols();

    Matrix merged(rows, cols * num);

    for (size_t c = 0; c < num; ++c) {
      if (channels[c].rows() != rows || channels[c].cols() != cols)
        throw std::invalid_argument("All channels must have same dimensions");

      for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
          merged(i, j * num + c) = channels[c](i, j);
        }
      }
    }
    return merged;
  }

  Matrix crop(size_t row_start, size_t col_start, size_t rows,
              size_t cols) const {
    return view(row_start, col_start, rows, cols);
  }

  Matrix flip(bool horizontal, bool vertical) const {
    Matrix result(view_rows_, view_cols_);

    for (size_t i = 0; i < view_rows_; ++i) {
      const size_t src_i = vertical ? view_rows_ - 1 - i : i;
      for (size_t j = 0; j < view_cols_; ++j) {
        const size_t src_j = horizontal ? view_cols_ - 1 - j : j;
        result(i, j) = operator()(src_i, src_j);
      }
    }
    return result;
  }

  Matrix warpAffine(const Matrix& transform) const {
    if (transform.rows() != 2 || transform.cols() != 3)
      throw std::invalid_argument("Affine transform must be 2x3");

    Matrix result(view_rows_, view_cols_);

    for (size_t i = 0; i < view_rows_; ++i) {
      for (size_t j = 0; j < view_cols_; ++j) {
        const double x =
            transform(0, 0) * j + transform(0, 1) * i + transform(0, 2);
        const double y =
            transform(1, 0) * j + transform(1, 1) * i + transform(1, 2);

        const size_t x1 = std::clamp(static_cast<size_t>(x),
                                     static_cast<size_t>(0), view_cols_ - 1);
        const size_t y1 = std::clamp(static_cast<size_t>(y),
                                     static_cast<size_t>(0), view_rows_ - 1);
        const size_t x2 = std::min(x1 + 1, view_cols_ - 1);
        const size_t y2 = std::min(y1 + 1, view_rows_ - 1);

        const double dx = x - x1;
        const double dy = y - y1;

        const T val = static_cast<T>((1 - dx) * (1 - dy) * operator()(y1, x1) +
                                     dx * (1 - dy) * operator()(y1, x2) +
                                     (1 - dx) * dy * operator()(y2, x1) +
                                     dx * dy * operator()(y2, x2));

        result(i, j) = val;
      }
    }
    return result;
  }

  Matrix warpPerspective(const Matrix& transform) const {
    if (transform.rows() != 3 || transform.cols() != 3)
      throw std::invalid_argument("Perspective transform must be 3x3");

    Matrix result(view_rows_, view_cols_);

    for (size_t i = 0; i < view_rows_; ++i) {
      for (size_t j = 0; j < view_cols_; ++j) {
        const double xp =
            transform(0, 0) * j + transform(0, 1) * i + transform(0, 2);
        const double yp =
            transform(1, 0) * j + transform(1, 1) * i + transform(1, 2);
        const double wp =
            transform(2, 0) * j + transform(2, 1) * i + transform(2, 2);

        if (wp == 0) {
          result(i, j) = T{0};
          continue;
        }

        const double x = xp / wp;
        const double y = yp / wp;

        const size_t x1 = std::clamp(static_cast<size_t>(x),
                                     static_cast<size_t>(0), view_cols_ - 1);
        const size_t y1 = std::clamp(static_cast<size_t>(y),
                                     static_cast<size_t>(0), view_rows_ - 1);
        const size_t x2 = std::min(x1 + 1, view_cols_ - 1);
        const size_t y2 = std::min(y1 + 1, view_rows_ - 1);

        const double dx = x - x1;
        const double dy = y - y1;

        const T val = static_cast<T>((1 - dx) * (1 - dy) * operator()(y1, x1) +
                                     dx * (1 - dy) * operator()(y1, x2) +
                                     (1 - dx) * dy * operator()(y2, x1) +
                                     dx * dy * operator()(y2, x2));

        result(i, j) = val;
      }
    }
    return result;
  }

  //Application Based functions (LINEAR ALGEBRA)
  // ==================== Linear Algebra Functions ====================

  Matrix E_SwapOp(size_t r0, size_t r1) {
    if (r0 >= view_rows_ || r1 >= view_rows_) {
      throw std::invalid_argument("Row indices out of bounds");
    }

    Matrix result = *this;
    for (size_t j = 0; j < view_cols_; ++j) {
      T temp = result(r0, j);
      result(r0, j) = result(r1, j);
      result(r1, j) = temp;
    }
    return result;
  }

  Matrix E_subtract_product_row_op(T factor, size_t r0, size_t r1) {
    if (r0 >= view_rows_ || r1 >= view_rows_) {
      throw std::invalid_argument("Row indices out of bounds");
    }

    Matrix result = *this;
    for (size_t j = 0; j < view_cols_; ++j) {
      result(r1, j) -= factor * result(r0, j);
    }
    return result;
  }

  void normalize_pivot_row(Matrix& matrix, const T& pivotElement,
                           size_t processingRow,
                           size_t processingColumn) const {
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

  void eliminate_rows(Matrix& matrix, const T& pivotElement,
                      size_t processingRow, size_t processingColumn,
                      bool reduced_form) const {
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
  // Decomposition_LU to_LU() const {
  //   if (!is_square())
  //     throw std::invalid_argument("LU decomposition requires square matrix");

  //   size_t n = view_rows_;
  //   Matrix L = Matrix::identity(n);
  //   Matrix U = *this;

  //   for (size_t k = 0; k < n - 1; ++k) {
  //     for (size_t i = k + 1; i < n; ++i) {
  //       if (U(k, k) == T{0}) {
  //         throw std::runtime_error(
  //             "LU decomposition failed: zero pivot encountered");
  //       }

  //       T factor = U(i, k) / U(k, k);
  //       L(i, k) = factor;

  //       for (size_t j = k; j < n; ++j) {
  //         U(i, j) -= factor * U(k, j);
  //       }
  //     }
  //   }

  //   return {L, U};
  // }

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
        throw std::runtime_error(
            "LDU decomposition failed: zero diagonal in U");
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
  Decomposition_Cholesky to_cholesky() const {
    if (!is_square()) {
      throw std::invalid_argument(
          "Cholesky decomposition requires square matrix");
    }
    if (!is_symmetric()) {
      throw std::invalid_argument(
          "Cholesky decomposition requires symmetric matrix");
    }

    const size_t n = rows();
    Decomposition_Cholesky result;
    result.L = Matrix(n, n, T{0});

    for (size_t i = 0; i < n; ++i) {
      T sum = T{0};
      for (size_t k = 0; k < i; ++k) {
        sum += result.L(i, k) * result.L(i, k);
      }
      T a_ii = operator()(i, i);
      T diag_sq = a_ii - sum;
      if (diag_sq <= T{0}) {
        throw std::runtime_error("Matrix is not positive definite");
      }
      result.L(i, i) = std::sqrt(diag_sq);

      // for (size_t j = i + 1; j < n; ++j) {
      //   sum = T{0};
      //   for (size_t k = 0; k < i; ++k) {
      //     sum += result.L(j, k) * result.L(i, k);
      //   }
      //   result.L(j, i) = (operator()(j, i) - sum) / result.L(i, i);
      // }
      size_t block_size = BLOCK_SIZE;
      for (size_t j_outer = i + 1; j_outer < n; j_outer += block_size) {
        size_t j_end = std::min(j_outer + block_size, n);
        for (size_t j = j_outer; j < j_end; ++j) {
          T sum = T{0};
          for (size_t k_outer = 0; k_outer < i; k_outer += block_size) {
            size_t k_end = std::min(k_outer + block_size, i);
            for (size_t k = k_outer; k < k_end; ++k) {
              sum += result.L(j, k) * result.L(i, k);
            }
          }
          result.L(j, i) = (operator()(j, i) - sum) / result.L(i, i);
        }
      }
    }

    return result;
  }

  T det() const {
    if (!is_square())
      throw std::invalid_argument("Determinant requires square matrix");

    if (view_rows_ == 1) {
      return operator()(0, 0);
    }

    if (view_rows_ == 2) {
      return operator()(0, 0) * operator()(1, 1) - operator()(0, 1) *
                                                       operator()(1, 0);
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
          Matrix minor = minor_matrix(i, j);
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

  SolutionType findSolutionType(size_t numUnknowns) {
    bool hasNonZeroAZeroBColumn = false;
    bool hasAllZeroANonZeroBColumn = false;
    Matrix rrefMatrix = *this;
    size_t nCols = rrefMatrix->cols();
    size_t bColumn = nCols - 1;
    size_t AColumns = nCols - 2;
    for (size_t i = 0; i < rrefMatrix->rows(); ++i) {
      if (rrefMatrix(i, AColumns) != 0 && rrefMatrix(i, bColumn) == 0) {
        hasNonZeroAZeroBColumn = true;
        std::cout << "hasNonZeroAZeroBColumn\n";
      }

      if (rrefMatrix(i, AColumns) == 0 && rrefMatrix(i, bColumn) != 0) {
        hasAllZeroANonZeroBColumn = true;
        std::cout << "hasAllZeroANonZeroBColumn \n";
      }
    }

    if (hasNonZeroAZeroBColumn || hasAllZeroANonZeroBColumn) {
      return SolutionType::NO_SOLUTION;
    } else if (rrefMatrix->rank() < numUnknowns) {
      return SolutionType::INFINITE_SOLUTIONS;
    } else {
      return SolutionType::EXACT_SOLUTION;
    }
  }

  std::vector<T> backSubstitution() {
    Matrix echelonMatrix = *this;
    std::vector<T> solution(echelonMatrix->rows(), 0);

    for (int i = echelonMatrix->rows() - 1; i >= 0; --i) {
      T val = echelonMatrix(i, echelonMatrix->cols() - 1);
      for (size_t j = i + 1; j < echelonMatrix->cols() - 1; ++j) {
        val -= echelonMatrix(i, j) * solution[j];
      }
      solution[i] = val / echelonMatrix(i, i);
    }

    return solution;
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

  std::vector<std::vector<T>> calculateXs(
      const std::vector<size_t>& pivotColumns,
      const std::vector<size_t>& freeColumns, size_t numCols) {
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
      throw std::invalid_argument(
          "Right-hand side rows must match matrix rows");
    }

    // Create the solution matrix with the same dimensions as rhs
    Matrix X(rhs.rows(), rhs.cols());
    size_t block_size = BLOCK_SIZE;
    for (size_t col_outer = 0; col_outer < rhs.cols();
         col_outer += block_size) {
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
  Matrix block_jacobi(const Matrix& b, size_t block_size, double tolerance,
                      int max_iterations) {
    if (!is_square())
      throw std::invalid_argument("Matrix must be square");
    const size_t n = rows();
    Matrix x = Matrix::zeros(n, 1);      // Initial guess
    Matrix x_new = Matrix::zeros(n, 1);  // Initialize x_new with zeros

    for (int iter = 0; iter < max_iterations; ++iter) {
      Matrix r = b - (*this) * x;  // Residual using previous x

#pragma omp parallel for
      for (size_t i = 0; i < n; i += block_size) {
        size_t bs = std::min(block_size, n - i);
        // Extract residual block
        Matrix r_block = r.view(i, 0, bs, 1);
        Matrix z(size_t(bs), size_t(1));
        try {
          // Compute LU for this block
          auto D_block = view(i, i, bs, bs).clone();
          auto lu = D_block.to_LU(PivotingStrategy::PARTIAL);
          // Apply permutation P to residual
          Matrix Pb = Matrix::zeros(bs, 1);
          for (size_t k = 0; k < bs; ++k)
            Pb(k, 0) = r_block(lu.P[k], 0);
          // Solve Ly = Pb, Uz = y
          Matrix y = lu.L.solve_triangular(Pb, true, true);
          z = lu.U.solve_triangular(y, false, false);
        } catch (...) {
          // Singular or ill-conditioned block: use diagonal-only Jacobi
          z = Matrix::zeros(bs, 1);
          for (size_t k = 0; k < bs; ++k)
            z(k, 0) = r_block(k, 0) / operator()(i + k, i + k);
        }
        // Update the solution block
        x_new.set_block(i, 0, x.view(i, 0, bs, 1) + z);
      }

      // Check convergence using residual norm
      double residual_norm = (b - (*this) * x_new).norm();
      if (tolerance >= std::numeric_limits<T>::epsilon() &&
          residual_norm < tolerance) {
        x = x_new;
        return x;
      }
      x = x_new;  // Update solution for next iteration
    }
    // Max iterations reached: return last iterate (x)
    return x;
  }

  Matrix balanced_form_fast(size_t max_iterations = 5) const {
    static_assert(std::is_floating_point_v<T>,
                  "balanced_form requires floating-point types.");
    if (!is_square())
      throw std::invalid_argument("Matrix must be square");

    const size_t n = view_rows_;
    Matrix<T> balanced(*this);
    const T radix = static_cast<T>(2);
    const T eps = std::numeric_limits<T>::epsilon();
    const T convergence_factor = static_cast<T>(0.95);
    const T min_scaling = std::sqrt(radix);  // ~1.414
    // const T max_scaling = radix * min_scaling;  // ~2.828 (removed unused variable)

    bool converged = false;

    for (size_t iter = 0; !converged && iter < max_iterations; ++iter) {
      converged = true;  // Assume convergence unless scaling occurs

      // Compute off-diagonal row and column norms
      std::vector<T> row_norms(n, T(0));
      std::vector<T> col_norms(n, T(0));
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
          if (i != j) {
            T val = std::abs(balanced(i, j));
            row_norms[i] += val;
            col_norms[j] += val;
          }
        }
      }

      // Determine scaling factors for each row
      std::vector<T> scaling_factors(n, T(1));

      for (size_t i = 0; i < n; ++i) {
        T original_norm_sum = row_norms[i] + col_norms[i];
        if (original_norm_sum < eps ||
            (row_norms[i] < eps && col_norms[i] < eps))
          continue;  // Skip if norms are negligible

        T f = T(1);
        T row_norm = row_norms[i];
        T col_norm = col_norms[i];

        // Adjust scaling to balance row and column norms
        T target = col_norm / radix;
        while (row_norm < target) {
          f *= radix;
          row_norm *= radix;
          target /= radix;
        }

        target = col_norm * radix;
        while (row_norm > target) {
          f /= radix;
          row_norm /= radix;
          target *= radix;
        }

        // Check if scaling provides sufficient improvement
        T scaled_norm = row_norm + col_norm / f;
        if (scaled_norm < convergence_factor * original_norm_sum &&
            (f >= min_scaling || f <= T(1) / min_scaling)) {
          scaling_factors[i] = f;
          converged = false;  // Need another iteration
        }
      }

      // Apply scaling to the matrix
      for (size_t i = 0; i < n; ++i) {
        T f = scaling_factors[i];
        if (f == T(1))
          continue;

        // Scale row i by f and column i by 1/f
        for (size_t j = 0; j < n; ++j) {
          balanced(i, j) *= f;
          if (j != i)  // Avoid scaling diagonal twice
            balanced(j, i) /= f;
        }
      }

      // Early exit if no scaling was applied
      if (converged)
        break;
    }
    return balanced;
  }

  T compute_imbalance() const {
    static_assert(std::is_floating_point_v<T>,
                  "imbalance calculation requires floating-point types");
    if (!is_square())
      throw std::invalid_argument("Matrix must be square");

    const size_t n = view_rows_;

    std::vector<T> row_norms(n, T(0)), col_norms(n, T(0));
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        if (i != j) {
          T val = std::abs(operator()(i, j));
          row_norms[i] += val;
          col_norms[j] += val;
        }
      }
    }

    T max_ratio = T(1);
    for (size_t i = 0; i < n; ++i) {
      if (row_norms[i] <= T(0) || col_norms[i] <= T(0))
        continue;

      T ratio = row_norms[i] / col_norms[i];
      T inv_ratio = col_norms[i] / row_norms[i];
      max_ratio = std::max({max_ratio, ratio, inv_ratio});
    }

    return max_ratio;
  }

  Matrix balanced_form_two_phase(T /*epsilon*/) const {
    static_assert(std::is_floating_point_v<T>,
                  "balanced_form requires floating-point types");
    if (!is_square())
      throw std::invalid_argument("Matrix must be square");

    const size_t n = view_rows_;
    Matrix balanced(*this);
    const T radix = static_cast<T>(2);

    // The balancing loop consists of two phases: the raising phase and the
    // lowering phase. Number of iterations for each phase.
    constexpr size_t T_phase = 3;

    // Calculate initial imbalance
    T initial_imbalance = balanced.compute_imbalance();

    // Raising Phase - scales rows with small norms relative to columns
    for (size_t iter = 0; iter < T_phase; ++iter) {
      std::vector<T> row_norms(n, T(0));
      std::vector<T> col_norms(n, T(0));

      // Calculate row and column norms
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
          if (i != j) {
            T val = std::abs(balanced(i, j));
            row_norms[i] += val;
            col_norms[j] += val;
          }
        }
      }

      // Find row with minimum row/col norm ratio
      size_t i_raise = n;
      T min_ratio = std::numeric_limits<T>::max();
      for (size_t i = 0; i < n; ++i) {
        if (row_norms[i] > T(0) && col_norms[i] > T(0)) {
          T ratio = row_norms[i] / col_norms[i];
          if (ratio < min_ratio) {
            min_ratio = ratio;
            i_raise = i;
          }
        }
      }

      // Raising Phase - Apply scaling to the identified row
      if (i_raise < n) {
        T f = radix;
        // Scale row i_raise up by f
        for (size_t j = 0; j < n; ++j) {
          balanced(i_raise, j) *= f;
        }
        // Scale column i_raise down by f (include diagonal)
        for (size_t i = 0; i < n; ++i) {
          balanced(i, i_raise) /= f;
        }
      }
    }

    // Lowering Phase - scales rows with large norms relative to columns
    for (size_t iter = 0; iter < T_phase; ++iter) {
      std::vector<T> row_norms(n, T(0));
      std::vector<T> col_norms(n, T(0));

      // Calculate row and column norms
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
          if (i != j) {
            T val = std::abs(balanced(i, j));
            row_norms[i] += val;
            col_norms[j] += val;
          }
        }
      }

      // Find row with maximum row/col norm ratio
      size_t i_lower = n;
      T max_ratio = T(0);
      for (size_t i = 0; i < n; ++i) {
        if (row_norms[i] > T(0) && col_norms[i] > T(0)) {
          T ratio = row_norms[i] / col_norms[i];
          if (ratio > max_ratio) {
            max_ratio = ratio;
            i_lower = i;
          }
        }
      }

      // Lowering Phase - Apply scaling to the identified row
      if (i_lower < n) {
        T f = T(1) / radix;
        // Scale row i_lower down by f
        for (size_t j = 0; j < n; ++j) {
          balanced(i_lower, j) *= f;
        }
        // Scale column i_lower up by f (include diagonal)
        for (size_t i = 0; i < n; ++i) {
          balanced(i, i_lower) /= f;
        }
      }
    }

    // Check if balancing improved the condition
    T final_imbalance = balanced.compute_imbalance();
    if (final_imbalance > initial_imbalance) {
      // If balancing made things worse, return the original matrix
      return *this;
    }

    return balanced;
  }
  Matrix<T> HessenbergReduceGQvdGBlocked(size_t b) {
    Matrix<T> A = *this;  // Create a copy to modify
    size_t n = A.rows();

    for (size_t k = 0; k + 2 < n; k += b) {
      size_t kb = std::min(b, n - k - 2);  // Actual block size

      // Perform panel factorization
      auto [U_panel, Z_panel, T_panel] = A.hessenberg_panel(k, kb);

      // Update trailing submatrix
      size_t trailing_col_start = k + kb + 1;
      if (trailing_col_start < n) {
        // Right update: A[0:k, trailing_col_start:n] = A[0:k, trailing_col_start:n] * Q
        if (k > 0) {
          Matrix<T> ATR =
              A.view(0, trailing_col_start, k, n - trailing_col_start);
          Matrix<T> Y = ATR * U_panel;
          Matrix<T> T_inv = T_panel.inverse();
          Matrix<T> W = Y * T_inv;
          ATR -= W * U_panel.transpose();
          A.view(0, trailing_col_start, k, n - trailing_col_start) = ATR;
        }

        // Left update: A[k:n, trailing_col_start:n] = Q^T * A[k:n, trailing_col_start:n]
        Matrix<T> Atrail =
            A.view(k, trailing_col_start, n - k, n - trailing_col_start);
        Matrix<T> T_inv = T_panel.inverse();

        // Compute first application Q^T * Atrail
        Matrix<T> W = U_panel.transpose() * Atrail;
        W = T_inv.transpose() * W;
        Atrail -= U_panel * W;

        A.view(k, trailing_col_start, n - k, n - trailing_col_start) = Atrail;
      }
    }

    return A;
  }

  std::tuple<Matrix<T>, Matrix<T>, Matrix<T>, Matrix<T>> hessenberg_panel(
      size_t k, size_t b) {
    size_t n = this->rows();
    size_t m = n - k;
    Matrix<T> U_panel(size_t(m), size_t(b));
    Matrix<T> Z_panel(size_t(m), size_t(b));
    Matrix<T> T_panel = Matrix::zeros(b, b);

    // // Initialize T_panel to zero
    // for (size_t i = 0; i < b; ++i) {
    //   for (size_t j = 0; j < b; ++j) {
    //     T_panel(i, j) = T(0);
    //   }
    // }

    for (size_t j = 0; j < b; ++j) {
      size_t col = k + j;

      // Apply previous transformations to current column
      if (j > 0) {
        Matrix<T> U_prev = U_panel.view(0, 0, m, j);
        Matrix<T> current_col = this->view(k + j + 1, col, n - (k + j + 1), 1);
        Matrix<T> y = U_prev.transpose() * current_col;

        Matrix<T> T_sub = T_panel.view(0, 0, j, j);
        Matrix<T> z = T_sub.solve(y);

        current_col -= U_prev * z;
      }

      // Compute Householder vector
      Matrix<T> x = this->view(k + j + 1, col, n - (k + j + 1), 1);
      auto [u, tau] = x.house_v();

      // Store u in U_panel
      U_panel(j, j) = 1.0;
      if (u.rows() > 0) {
        U_panel(j + 1, j) = u;
      }
      T_panel(j, j) = tau;

      // Compute Z[:, j] = A * U[:, j]
      Matrix<T> A_panel = this->view(k, k, m, m);
      Matrix<T> U_col_j = U_panel.view(0, j, m, 1);
      Z_panel.view(0, j, m, 1) = A_panel * U_col_j;

      // Update T matrix
      if (j < b - 1) {
        for (size_t i = 0; i <= j; ++i) {
          for (size_t l = j + 1; l < b; ++l) {
            Matrix<T> U_col_i = U_panel.view(0, i, m, 1);
            Matrix<T> U_col_l = U_panel.view(0, l, m, 1);
            T tau_i = T_panel(i, i);
            T inner_prod = (U_col_i.transpose() * U_col_l)(0, 0);
            T_panel(i, l) += tau_i * inner_prod;
          }
        }
      }

      // Apply Householder transformation to trailing submatrix
      Matrix<T> trailing =
          this->view(k + j + 1, col + 1, n - (k + j + 1), n - (col + 1));
      Matrix<T> w = (u.transpose() * trailing).scale(tau);
      trailing -= u * w;
    }

    return {U_panel, Z_panel, T_panel};
  }

  // Linear system solver
  Matrix solve(const Matrix& b) const {
    if (!is_square())
      throw std::invalid_argument("Coefficient matrix must be square");

    if (view_rows_ != b.rows())
      throw std::invalid_argument("Incompatible dimensions for system solving");

    // Use the existing LU decomposition with full pivoting
    auto lu = to_LU(PivotingStrategy::FULL);

    size_t n = view_rows_;
    size_t m = b.cols();

    Matrix y(n, m);
    Matrix z(n, m);

    // Precompute PB = P * b
    Matrix PB = lu.P.to_matrix() * b;

    for (size_t j = 0; j < m; ++j) {
      for (size_t i = 0; i < n; ++i) {
        T sum = PB(i, j);
        for (size_t k = 0; k < i; ++k) {
          sum -= lu.L(i, k) * y(k, j);
        }
        y(i, j) = sum / lu.L(i, i);
      }
    }

    // Backward substitution: Solve U * z = y
    for (size_t j = 0; j < m; ++j) {
      for (int i = n - 1; i >= 0; --i) {
        T sum = y(i, j);
        for (size_t k = i + 1; k < n; ++k) {
          sum -= lu.U(i, k) * z(k, j);  // Ensure these are scalars
        }
        z(i, j) = sum / lu.U(i, i);
      }
    }

    // If full pivoting was used, apply column permutation to get the solution

    if (lu.full_pivoting) {
      return lu.Q.to_matrix() * z;
    } else {
      // No column permutation needed
      return z;
    }
  }

  std::tuple<Matrix, Matrix> qr_decomposition() const {
    if (view_rows_ < view_cols_)
      throw std::invalid_argument("QR decomposition requires rows >= columns");

    size_t m = view_rows_;
    size_t n = view_cols_;
    Matrix Q = Matrix::identity(m);
    Matrix R = this->clone();
    for (size_t j = 0; j < n; j++) {
      // Extract column
      Matrix x = R.sub_matrix(j, j, m - j, 1);
      // Compute Householder reflection
      auto [v, beta] = x.house_v();

      // Apply to R
      Matrix R_sub = R.sub_matrix(j, j, m - j, n - j);
      R_sub = R_sub.apply_householder_left(v, beta);
      R.set_block(j, j, R_sub);
      // Apply to Q
      Matrix Q_sub = Q.sub_matrix(0, j, m, m - j);
      Q_sub = Q_sub.apply_householder_right(v, beta);
      Q.set_block(0, j, Q_sub);
    }

    return {Q, R};
  }

  std::tuple<Matrix, Matrix> hessenberg_decomposition() const {
    if (!is_square())
      throw std::invalid_argument(
          "Hessenberg decomposition requires a square matrix");

    size_t n = view_rows_;
    Matrix H = this->clone();
    Matrix Q = Matrix::identity(n);

    for (size_t k = 0; k + 2 < n; k++) {
      // Extract column
      Matrix x = H.sub_matrix(k + 1, k, n - (k + 1), 1);
      // Compute Householder reflection
      auto [v, beta] = x.house_v();
      // Expand v to match the size needed for applying to the full blocks
      Matrix v_expanded(size_t(n - (k + 1)), size_t(1));
      for (size_t i = 0; i < v.rows(); i++) {
        v_expanded(i, 0) = v(i, 0);
      }

      // Apply to H from both sides
      // First from the left: H = P * H
      Matrix H_left = H.sub_matrix(k + 1, k, n - (k + 1), n - k);
      H_left = H_left.apply_householder_left(v_expanded, beta);
      H.set_submatrix(k + 1, k, H_left);

      // Then from the right: H = H * P
      Matrix H_right = H.sub_matrix(0, k + 1, n, n - (k + 1));
      H_right = H_right.apply_householder_right(v_expanded, beta);
      H.set_submatrix(0, k + 1, H_right);

      // Update Q
      Matrix Q_sub = Q.sub_matrix(0, k + 1, n, n - (k + 1));
      Q_sub = Q_sub.apply_householder_right(v_expanded, beta);
      Q.set_submatrix(0, k + 1, Q_sub);
    }

    return {H, Q};
  }

  // Algorithm 3: Householder reflector based HT reduction

  static std::tuple<Matrix, Matrix, Matrix, Matrix> householder_ht_reduction(
      const Matrix<T, StoragePolicy>& A_in,
      const Matrix<T, StoragePolicy>& B_in) {
    if (!A_in.is_square() || !B_in.is_square() || A_in.rows() != B_in.rows())
      throw std::invalid_argument(
          "Matrices A and B must be square and of the same size");

    size_t n = A_in.rows();
    if (n <= 1) {

      return {A_in, B_in, Matrix::identity(n), Matrix::identity(n)};
    }

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
    for (size_t j = 0; j + 2 < n; j++) {
      // Step 5: Calculate Householder reflector H2 for A
      Matrix A_col = A.get_column(j).sub_matrix(j + 1, 0, n - (j + 1), 1);

      auto [v2, beta2] = A_col.house_v();
      // Expand v2 to match the size needed for applying to the full blocks
      Matrix v2_expanded(size_t(n - (j + 1)), size_t(1));
      for (size_t i = 0; i < v2.rows(); i++) {
        v2_expanded(i, 0) = v2(i, 0);
      }

      // Step 6: Apply H2 to A and B from the left and to Q from the right
      Matrix A_block = A.sub_matrix(j + 1, j, n - (j + 1), n - j);
      A_block = A_block.apply_householder_left(v2_expanded, beta2);
      A.set_submatrix(j + 1, j, A_block);

      // B(j+1:n, j:n) = H2 * B(j+1:n, j:n)
      Matrix B_block = B.sub_matrix(j + 1, j, n - (j + 1), n - j);
      B_block = B_block.apply_householder_left(v2_expanded, beta2);
      B.set_submatrix(j + 1, j, B_block);

      // Q(:, j+1:n) = Q(:, j+1:n) * H2^H
      Matrix Q_block = Q.sub_matrix(0, j + 1, n, n - (j + 1));
      Q_block = Q_block.apply_householder_right(v2_expanded, beta2);
      Q.set_submatrix(0, j + 1, Q_block);

      // Step 7: Solve the linear system B_{j+1:n, j+1:n} * x = e1
      Matrix B_submatrix = B.sub_matrix(j + 1, j + 1, n - (j + 1), n - (j + 1));
      Matrix e1 = Matrix::unit_vector(n - (j + 1), 0);
      Matrix x = B_submatrix.solve(e1);
      // Ensure x is properly normalized for H3 calculation
      // T x_norm = x.norm();
      // if (x_norm > 0) {
      //   x = x / x_norm;
      // }
      // Step 8: Calculate Householder reflector H3 for x
      auto [v3, beta3] = x.house_v();
      // Step 9: Apply H3 to A, B, and Z from the right
      // A(:, j+1:n) = A(:, j+1:n) * H3
      Matrix A_right_block = A.sub_matrix(0, j + 1, n, n - (j + 1));
      A_right_block = A_right_block.apply_householder_right(v3, beta3);
      A.set_submatrix(0, j + 1, A_right_block);

      // B(:, j+1:n) = B(:, j+1:n) * H3
      Matrix B_right_block = B.sub_matrix(0, j + 1, n, n - (j + 1));
      B_right_block = B_right_block.apply_householder_right(v3, beta3);
      B.set_submatrix(0, j + 1, B_right_block);

      // Z(:, j+1:n) = Z(:, j+1:n) * H3
      Matrix Z_right_block = Z.sub_matrix(0, j + 1, n, n - (j + 1));
      Z_right_block = Z_right_block.apply_householder_right(v3, beta3);
      Z.set_submatrix(0, j + 1, Z_right_block);
    }

    return {A, B, Q, Z};
  }

  std::tuple<Matrix, Matrix, Matrix, Matrix>
  householder_ht_reduction_single_column(const Matrix& A_sub,
                                         const Matrix& B_sub, size_t col_idx) {

    size_t n = A_sub.rows();
    // Create copies to work with
    Matrix A = A_sub.clone();
    Matrix B = B_sub.clone();
    // Initialize Q and Z as identity matrices
    Matrix Q = Matrix::identity(n);
    Matrix Z = Matrix::identity(n);
    // This is a simplified version that focuses only on reducing column col_idx

    // Calculate Householder reflector for column col_idx
    Matrix A_col =
        A.get_column(col_idx).sub_matrix(col_idx + 1, 0, n - (col_idx + 1), 1);

    auto [v, beta] = A_col.house_v();

    // Create full-sized reflector vector padded with zeros
    Matrix v_full = Matrix::zeros(n, 1);
    for (size_t i = 0; i < v.rows(); i++) {
      v_full(col_idx + 1 + i, 0) = v(i, 0);
    }

    // Apply the reflector
    A = A.apply_householder_left(v_full, beta);
    B = B.apply_householder_left(v_full, beta);
    Q = Q.apply_householder_right(v_full, beta);

    // Make B upper triangular in this column range
    // For each row below the diagonal, eliminate the element
    for (size_t i = col_idx + 1; i < n; i++) {
      if (std::abs(B(i, col_idx)) > 1e-12) {
        // Create a Givens rotation to eliminate B(i, col_idx)
        T c, s;
        T r = std::hypot(B(col_idx, col_idx), B(i, col_idx));
        c = B(col_idx, col_idx) / r;
        s = -B(i, col_idx) / r;

        // Apply the Givens rotation to rows col_idx and i of B
        for (size_t j = col_idx; j < n; j++) {
          T temp = c * B(col_idx, j) - s * B(i, j);
          B(i, j) = s * B(col_idx, j) + c * B(i, j);
          B(col_idx, j) = temp;
        }

        // Apply the Givens rotation to rows col_idx and i of A
        for (size_t j = 0; j < n; j++) {
          T temp = c * A(col_idx, j) - s * A(i, j);
          A(i, j) = s * A(col_idx, j) + c * A(i, j);
          A(col_idx, j) = temp;
        }

        // Update Q to account for this transformation
        for (size_t j = 0; j < n; j++) {
          T temp = c * Q(j, col_idx) - s * Q(j, i);
          Q(j, i) = s * Q(j, col_idx) + c * Q(j, i);
          Q(j, col_idx) = temp;
        }
      }
    }

    return {A, B, Q, Z};
  }

  // Algorithm 6: Full Hessenberg-triangular reduction with preprocessing and iterative refinement

  std::tuple<Matrix, Matrix, Matrix, Matrix> hessenberg_triangular_reduction(
      const Matrix& A_in, const Matrix& B_in, T epsilon, T tol,
      size_t max_iter) {
    if (!A_in.is_square() || !B_in.is_square() || A_in.rows() != B_in.rows())
      throw std::invalid_argument(
          "Matrices A and B must be square and of the same size");

    size_t n = A_in.rows();

    // Create copies to work with
    Matrix A = A_in.clone();
    Matrix B = B_in.clone();

    // Step 1: Initialize k
    size_t k = 0;

    // Step 2: Calculate the RRRQ decomposition of B
    // For simplicity, we'll use QR decomposition of B^T, which gives B = P*R*Z
    Matrix B_T = B.transpose();
    auto [Q_B, R_B] = B_T.qr_decomposition();
    Matrix P = Q_B;
    Matrix Zc = R_B.transpose();

    // Steps 3-4: Update A and B
    A = P * A * Zc;
    B = P * B * Zc;

    // Steps 5-6: Initialize Q and Z
    Matrix Q = P.transpose();
    Matrix Z = Zc;

    // Calculate norm of B for comparison
    T B_norm = B.frobenius_norm();

    // Steps 7-10: Handle small diagonal elements in B
    while (k < n && std::abs(B(k, k)) < epsilon * B_norm) {
      // Step 8: Reduce a single column using appropriate HT reduction
      // This would typically use Algorithm 3 for the specific column
      Matrix A_sub = A.sub_matrix(k, k, n - k, n - k);
      Matrix B_sub = B.sub_matrix(k, k, n - k, n - k);

      auto [A_reduced, B_reduced, Q_k, Z_k] =
          householder_ht_reduction_single_column(A_sub, B_sub, k);

      // Update the matrices
      A.set_block(k, k, A_reduced);
      B.set_block(k, k, B_reduced);

      // Update Q and Z
      Matrix Q_full = Matrix::identity(n);
      Matrix Z_full = Matrix::identity(n);

      Q_full.set_block(k, k, Q_k);
      Z_full.set_block(k, k, Z_k);

      Q = Q * Q_full;
      Z = Z * Z_full;

      // Step 9: Increment k
      k++;
    }

    // Steps 11-24: Main iterative refinement loop
    bool is_hessenberg = false;
    size_t iter_count = 0;

    while (!is_hessenberg && iter_count < max_iter) {
      // Step 12: Compute X = Ak:n,k:n * inv(Bk:n,k:n)
      Matrix A_sub = A.sub_matrix(k, k, n - k, n - k);
      Matrix B_sub = B.sub_matrix(k, k, n - k, n - k);
      Matrix X = A_sub * B_sub.inverse();

      // Step 13: Calculate Qc so that Qc^T * X * Qc is Hessenberg
      auto [H, Qc] = X.hessenberg_decomposition();

      // Steps 14-16: Update A, B, and Q
      // Apply Qc^T from the left to A and B, and Qc from the right to Q
      Matrix A_left = A.sub_matrix(k, 0, n - k, n);
      A_left = Qc.transpose() * A_left;
      A.set_block(k, 0, A_left);

      Matrix B_left = B.sub_matrix(k, 0, n - k, n);
      B_left = Qc.transpose() * B_left;
      B.set_block(k, 0, B_left);

      Matrix Q_right = Q.sub_matrix(0, k, n, n - k);
      Q_right = Q_right * Qc;
      Q.set_block(0, k, Q_right);

      // Step 17: Calculate Zc so that Bk:n,k:n * Zc is upper triangular
      // For simplicity, we can use QR decomposition of B_sub^T
      B_T = B_sub.transpose();
      auto [Q_B_sub, R_B_sub] = B_T.qr_decomposition();
      Matrix Zc_sub = Q_B_sub;

      // Steps 18-20: Update A, B, and Z
      Matrix A_right = A.sub_matrix(0, k, n, n - k);
      A_right = A_right * Zc_sub;
      A.set_block(0, k, A_right);

      Matrix B_right = B.sub_matrix(0, k, n, n - k);
      B_right = B_right * Zc_sub;
      B.set_block(0, k, B_right);

      Matrix Z_right = Z.sub_matrix(0, k, n, n - k);
      Z_right = Z_right * Zc_sub;
      Z.set_block(0, k, Z_right);

      // Steps 21-23: Check for convergence and update k if necessary
      bool subdiagonal_small = true;
      for (size_t i = k + 2; i < n; i++) {
        if (std::abs(A(i, k)) > tol) {
          subdiagonal_small = false;
          break;
        }
      }

      if (subdiagonal_small) {
        k++;
      }

      // Check if A is in Hessenberg form
      // is_hessenberg = true;
      // for (size_t i = 0; i < n; i++) {
      //   for (size_t j = 0; j < i - 1; j++) {
      //     if (std::abs(A(i, j)) > tol) {
      //       is_hessenberg = false;
      //       break;
      //     }
      //   }
      //   if (!is_hessenberg)
      //     break;
      // }
      is_hessenberg = true;
      for (size_t i = 2; i < n; i++) {  // Start from i=2 since we need i-1 > 0
        for (size_t j = 0; j < i - 1; j++) {
          if (std::abs(A(i, j)) > tol) {
            is_hessenberg = false;
            break;
          }
        }
        if (!is_hessenberg)
          break;
      }

      iter_count++;
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

  //end of class
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