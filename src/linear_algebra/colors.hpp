#ifndef UTILS_HPP
#define UTILS_HPP
#include "matrix.hpp"
namespace algaber {  // ==================== Image Processing Utilities ====================

// Common image filters
template <Arithmetic T>
class Filters {
 public:
  static Matrix<T> blur(size_t size = 3) {
    if (size % 2 == 0)
      size++;  // Ensure odd size
    return Matrix<T>::gaussian_kernel(size, size / 6.0);
  }

  static Matrix<T> sharpen() {
    return Matrix<T>({{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}});
  }

  static Matrix<T> emboss() {
    return Matrix<T>({{-2, -1, 0}, {-1, 1, 1}, {0, 1, 2}});
  }

  static Matrix<T> edge_detect() {
    return Matrix<T>({{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}});
  }
};

// Image conversion utilities
template <Arithmetic T>
class ImageConverter {
 public:
  // Convert RGB to grayscale
  static Matrix<T> rgb_to_gray(const Matrix<T>& rgb_image,
                               T r_weight = T{0.299}, T g_weight = T{0.587},
                               T b_weight = T{0.114}) {
    if (rgb_image.cols() % 3 != 0)
      throw std::invalid_argument("RGB image should have cols divisible by 3");

    size_t rows = rgb_image.rows();
    size_t cols = rgb_image.cols() / 3;
    Matrix<T> gray(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        size_t idx = j * 3;
        gray(i, j) = r_weight * rgb_image(i, idx) +
                     g_weight * rgb_image(i, idx + 1) +
                     b_weight * rgb_image(i, idx + 2);
      }
    }

    return gray;
  }

  // Split RGB channels
  static std::tuple<Matrix<T>, Matrix<T>, Matrix<T>> split_rgb(
      const Matrix<T>& rgb_image) {
    Matrix<T> r = rgb_image.extract_channel(0, 3);
    Matrix<T> g = rgb_image.extract_channel(1, 3);
    Matrix<T> b = rgb_image.extract_channel(2, 3);
    return {r, g, b};
  }

  // Merge RGB channels into single image
  static Matrix<T> merge_rgb(const Matrix<T>& r, const Matrix<T>& g,
                             const Matrix<T>& b) {
    if (r.rows() != g.rows() || r.rows() != b.rows() || r.cols() != g.cols() ||
        r.cols() != b.cols())
      throw std::invalid_argument("All channels must have same dimensions");

    size_t rows = r.rows();
    size_t cols = r.cols();
    Matrix<T> result(rows, cols * 3);

    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        result(i, j * 3) = r(i, j);
        result(i, j * 3 + 1) = g(i, j);
        result(i, j * 3 + 2) = b(i, j);
      }
    }

    return result;
  }
};
}  // namespace algaber
#endif