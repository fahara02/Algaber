#ifndef TYPES_HPP
#define TYPES_HPP
#include <type_traits>
namespace algaber {
template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

// Forward declarations
template <Arithmetic T, typename StoragePolicy>
class Matrix;

// For SIMD-friendly operations
template <typename T>
concept SIMDCompatible = requires {
  requires Arithmetic<T>;
  requires(sizeof(T) == 1) || (sizeof(T) == 2) || (sizeof(T) == 4) ||
      (sizeof(T) == 8);
};

// Image filtering with different boundary handling
enum class BorderType {
  ZERO,       // Zero padding
  REFLECT,    // Reflect at boundary
  REPLICATE,  // Replicate border values
  WRAP        // Circular/wrap around
};
}  // namespace algaber
#endif