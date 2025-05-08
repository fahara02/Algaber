#ifndef STORAGE_POLICY_HPP
#define STORAGE_POLICY_HPP
#include <type_traits>
#include "types.hpp"

namespace algaber {  // ==================== Type traits and concepts ====================

// Forward declarations
template <Arithmetic T, typename StoragePolicy>
class Matrix;

template <Arithmetic T, typename StoragePolicy>
class MatrixView;

// ==================== Storage Policies ====================
// Base storage interface
template <Arithmetic T>
class StorageInterface {
 public:
  virtual ~StorageInterface() = default;

  virtual T& get(size_t row, size_t col) = 0;
  virtual const T& get(size_t row, size_t col) const = 0;

  virtual size_t rows() const noexcept = 0;
  virtual size_t cols() const noexcept = 0;
  virtual size_t stride() const noexcept = 0;

  virtual T* data() noexcept = 0;
  virtual const T* data() const noexcept = 0;

  virtual std::shared_ptr<StorageInterface<T>> clone() const = 0;
};

// Standard in-memory contiguous storage (row-major)
template <Arithmetic T>
class InMemoryStorage : public StorageInterface<T> {
 private:
  std::shared_ptr<std::vector<T>> data_;
  size_t rows_;
  size_t cols_;
  size_t stride_;  // For potential padding

 public:
  InMemoryStorage(size_t rows, size_t cols, T init_val = T{})
      : data_(std::make_shared<std::vector<T>>(rows * cols, init_val)),
        rows_(rows),
        cols_(cols),
        stride_(cols) {}

  // Create storage from existing data
  InMemoryStorage(T* external_data, size_t rows, size_t cols, size_t stride)
      : data_(std::make_shared<std::vector<T>>(external_data,
                                               external_data + rows * stride)),
        rows_(rows),
        cols_(cols),
        stride_(stride) {}

  // Copy constructor
  InMemoryStorage(const InMemoryStorage& other) = default;

  T& get(size_t row, size_t col) override {
    return (*data_)[row * stride_ + col];
  }

  const T& get(size_t row, size_t col) const override {
    return (*data_)[row * stride_ + col];
  }

  size_t rows() const noexcept override { return rows_; }
  size_t cols() const noexcept override { return cols_; }
  size_t stride() const noexcept override { return stride_; }

  T* data() noexcept override { return data_->data(); }
  const T* data() const noexcept override { return data_->data(); }

  std::shared_ptr<StorageInterface<T>> clone() const override {
    return std::make_shared<InMemoryStorage<T>>(*this);
  }
};

// Aligned memory storage for SIMD operations
template <Arithmetic T>
class AlignedStorage : public StorageInterface<T> {
 private:
  std::shared_ptr<std::vector<T, std::allocator<T>>> data_;
  size_t rows_;
  size_t cols_;
  size_t stride_;
  size_t alignment_;

 public:
  AlignedStorage(size_t rows, size_t cols, size_t alignment = 32,
                 T init_val = T{})
      : rows_(rows), cols_(cols), alignment_(alignment) {
    // Calculate stride with padding for alignment
    stride_ =
        (cols_ + (alignment_ / sizeof(T) - 1)) & ~(alignment_ / sizeof(T) - 1);
    data_ = std::make_shared<std::vector<T, std::allocator<T>>>(rows_ * stride_,
                                                                init_val);
  }

  T& get(size_t row, size_t col) override {
    return (*data_)[row * stride_ + col];
  }

  const T& get(size_t row, size_t col) const override {
    return (*data_)[row * stride_ + col];
  }

  size_t rows() const noexcept override { return rows_; }
  size_t cols() const noexcept override { return cols_; }
  size_t stride() const noexcept override { return stride_; }

  T* data() noexcept override { return data_->data(); }
  const T* data() const noexcept override { return data_->data(); }

  std::shared_ptr<StorageInterface<T>> clone() const override {
    return std::make_shared<AlignedStorage<T>>(*this);
  }
};

// Sparse storage for large matrices with many zero elements
template <Arithmetic T>
class SparseStorage : public StorageInterface<T> {
 private:
  size_t rows_;
  size_t cols_;
  std::map<std::pair<size_t, size_t>, T> elements_;
  T zero_value_;

 public:
  SparseStorage(size_t rows, size_t cols, T zero_value = T{})
      : rows_(rows), cols_(cols), zero_value_(zero_value) {}

  T& get(size_t row, size_t col) override {
    auto key = std::make_pair(row, col);
    auto it = elements_.find(key);
    if (it == elements_.end()) {
      elements_[key] = zero_value_;
      return elements_[key];
    }
    return it->second;
  }

  const T& get(size_t row, size_t col) const override {
    auto key = std::make_pair(row, col);
    auto it = elements_.find(key);
    if (it == elements_.end()) {
      return zero_value_;
    }
    return it->second;
  }

  size_t rows() const noexcept override { return rows_; }
  size_t cols() const noexcept override { return cols_; }
  size_t stride() const noexcept override { return cols_; }

  // Note: Sparse matrices don't have contiguous storage
  T* data() noexcept override { return nullptr; }
  const T* data() const noexcept override { return nullptr; }

  std::shared_ptr<StorageInterface<T>> clone() const override {
    return std::make_shared<SparseStorage<T>>(*this);
  }
};

template <Arithmetic T>
class ViewStorage : public StorageInterface<T> {
 private:
  T* data_;
  size_t rows_;
  size_t cols_;
  size_t stride_;

 public:
  ViewStorage(T* data, size_t rows, size_t cols, size_t stride)
      : data_(data), rows_(rows), cols_(cols), stride_(stride) {}

  T& get(size_t row, size_t col) override { return data_[row * stride_ + col]; }

  const T& get(size_t row, size_t col) const override {
    return data_[row * stride_ + col];
  }

  size_t rows() const noexcept override { return rows_; }
  size_t cols() const noexcept override { return cols_; }
  size_t stride() const noexcept override { return stride_; }

  T* data() noexcept override { return data_; }
  const T* data() const noexcept override { return data_; }

  std::shared_ptr<StorageInterface<T>> clone() const override {
    // When cloning, we create a copy of the data
    auto storage = std::make_shared<InMemoryStorage<T>>(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        storage->get(i, j) = get(i, j);
      }
    }
    return storage;
  }
};

}  // namespace algaber
#endif