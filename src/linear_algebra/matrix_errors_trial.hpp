#pragma once
#include <atomic>
#include <chrono>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <source_location>
#include <string>
#include <variant>

namespace algaber {

// ==============================================
// Core Error Types
// ==============================================

enum class ErrorCode {
  INVALID_DIMENSIONS,   // Matrix dimensions don't match operation
  DIMENSION_MISMATCH,   //MAtrix Dimension Not Match
  SINGULAR_MATRIX,      // Matrix is non-invertible
  CONVERGENCE_FAILURE,  // Iterative methods didn't converge
  OUT_OF_BOUNDS,        // Generic out of bounds access
  DIVIDE_BY_ZERO,       // Numerical division by zero
  RATE_LIMITED,         // Too many errors in time window
  NOT_SQUARE,           // Operation requires square matrix
  NOT_SOLVABLE,         // System has no solution
  CALLBACK_OVERRIDE     // Custom handler intercepted error
};

class MatrixException : public std::runtime_error {
 public:
  struct Context {
    std::source_location location;
    std::chrono::system_clock::time_point timestamp;
    std::string stack_trace;  // Optional debug info
  };

  MatrixException(ErrorCode code, std::string msg, Context ctx);
  MatrixException(ErrorCode code, std::string msg);

  ErrorCode code() const noexcept { return m_code; }
  const Context& context() const noexcept { return m_ctx; }

 private:
  static std::string format_msg(const std::string& msg, ErrorCode code,
                                const Context& ctx);
  ErrorCode m_code;
  Context m_ctx;
};

// ==============================================
// Result Type (Monadic Error Handling)
// ==============================================

template <typename T>
class Result {
 private:
  std::variant<T, ErrorCode> m_data;

 public:
  // Constructors
  Result(T value) : m_data(std::move(value)) {}
  Result(ErrorCode code) : m_data(code) {}

  // State checking
  bool is_ok() const noexcept { return std::holds_alternative<T>(m_data); }
  bool is_err() const noexcept {
    return std::holds_alternative<ErrorCode>(m_data);
  }

  // Value access
  T& value() & {
    if (!is_ok())
      throw MatrixException(std::get<ErrorCode>(m_data),
                            "Attempt to access value of failed result");
    return std::get<T>(m_data);
  }

  const T& value() const& {
    if (!is_ok())
      throw MatrixException(std::get<ErrorCode>(m_data),
                            "Attempt to access value of failed result");
    return std::get<T>(m_data);
  }

  ErrorCode error() const noexcept {
    return is_err() ? std::get<ErrorCode>(m_data) : ErrorCode{};
  }

  // Implicit conversion
  operator T&() & { return value(); }
  operator const T&() const& { return value(); }

  // Implicit conversion to U: unwraps value or throws on error
  operator T() const {
    if (is_err()) {
      throw MatrixException(error(), "Result implicit conversion failed");
    }
    return std::move(std::get<T>(m_data));
  }

  // Utilities
  template <typename U>
  T value_or(U&& default_value) const& {
    return is_ok() ? std::get<T>(m_data) : std::forward<U>(default_value);
  }

  explicit operator bool() const noexcept { return is_ok(); }

  std::optional<std::string> error_message() const {
    return is_err()
               ? std::make_optional(
                     MatrixException(std::get<ErrorCode>(m_data), "").what())
               : std::nullopt;
  }
};

template <typename U>
class Result<U&> {
 public:
  // Constructors
  Result(U& value) : m_data(std::ref(value)) {}
  Result(ErrorCode code) : m_data(code) {}

  // State checking
  bool is_ok() const noexcept {
    return std::holds_alternative<std::reference_wrapper<U>>(m_data);
  }
  bool is_err() const noexcept {
    return std::holds_alternative<ErrorCode>(m_data);
  }

  // Value access
  U& value() & {
    if (!is_ok())
      throw MatrixException(std::get<ErrorCode>(m_data),
                            "Attempt to access reference of failed result");
    return std::get<std::reference_wrapper<U>>(m_data).get();
  }
  const U& value() const& {
    if (!is_ok())
      throw MatrixException(std::get<ErrorCode>(m_data),
                            "Attempt to access reference of failed result");
    return std::get<std::reference_wrapper<U>>(m_data).get();
  }

  // Error access
  ErrorCode error() const noexcept {
    return is_err() ? std::get<ErrorCode>(m_data) : ErrorCode{};
  }

  explicit operator bool() const noexcept { return is_ok(); }

 private:
  std::variant<std::reference_wrapper<U>, ErrorCode> m_data;
};

// ==============================================
// Error Handling System
// ==============================================

namespace error {

enum class Mode {
  THROW,        // Throw exceptions on errors
  RETURN_CODE,  // Return error codes
  SILENT        // Suppress errors
};

namespace impl {
inline auto& thread_state() {
  struct ThreadState {
    Mode mode = Mode::THROW;
    std::function<void(ErrorCode, std::string_view)> handler;
    std::chrono::steady_clock::time_point last_error;
    std::atomic<size_t> error_count = 0;
  };
  thread_local ThreadState state;
  return state;
}

inline auto& global_mutex() {
  static std::mutex mtx;
  return mtx;
}
}  // namespace impl

// Configuration API
inline Mode mode() noexcept {
  return impl::thread_state().mode;
}

inline void set_mode(Mode m) noexcept {
  std::lock_guard lock(impl::global_mutex());
  impl::thread_state().mode = m;
}

inline void set_handler(auto&& handler) {
  std::lock_guard lock(impl::global_mutex());
  impl::thread_state().handler = std::forward<decltype(handler)>(handler);
}

inline void configure_rate_limit(
    std::chrono::milliseconds window = std::chrono::seconds(1),
    size_t max_errors = 100) noexcept {
  std::lock_guard lock(impl::global_mutex());
  auto& state = impl::thread_state();
  state.last_error = std::chrono::steady_clock::now() - window;
  state.error_count.store(max_errors, std::memory_order_relaxed);
}

// RAII mode control
class ScopedMode {
 public:
  explicit ScopedMode(Mode m) : m_prev(mode()) { set_mode(m); }
  ~ScopedMode() { set_mode(m_prev); }

  ScopedMode(const ScopedMode&) = delete;
  ScopedMode& operator=(const ScopedMode&) = delete;

 private:
  Mode m_prev;
};

}  // namespace error

// ==============================================
// Implementation Details
// ==============================================

namespace matrix::detail {

[[noreturn]]
inline void throw_error(ErrorCode code, std::string_view msg,
                        const std::source_location& loc) {
  MatrixException::Context ctx{.location = loc,
                               .timestamp = std::chrono::system_clock::now(),
#ifdef COLLECT_STACK_TRACES
                               .stack_trace = debug::get_stack_trace()
#else
                               .stack_trace = ""
#endif
  };
  throw MatrixException(code, std::string(msg), std::move(ctx));
}

template <typename F>
inline auto handle_error(ErrorCode code, std::string_view msg,
                         const std::source_location& loc, F&& cleanup) {
  auto& state = error::impl::thread_state();

  // Rate limiting
  const auto now = std::chrono::steady_clock::now();
  if (state.error_count.load(std::memory_order_relaxed) == 0) {
    code = ErrorCode::RATE_LIMITED;
    msg = "Error rate limit exceeded";
  } else {
    state.error_count.fetch_sub(1, std::memory_order_relaxed);
  }
  state.last_error = now;

  // Custom handler
  if (state.handler) {
    cleanup();
    state.handler(code, msg);
    if constexpr (!std::is_void_v<decltype(state.handler(code, msg))>) {
      return static_cast<int>(code);
    }
    return;
  }

  // Default handling
  cleanup();
  switch (state.mode) {
    case error::Mode::THROW:
      throw_error(code, msg, loc);
    case error::Mode::RETURN_CODE:
      return static_cast<int>(code);
    case error::Mode::SILENT:
      if constexpr (std::is_void_v<decltype(cleanup())>)
        return;
      else
        return static_cast<int>(code);
  }
}

}  // namespace matrix::detail

namespace detail {
template <typename>
struct is_result : std::false_type {};
template <typename T>
struct is_result<Result<T>> : std::true_type {};
template <typename T>
inline constexpr bool is_result_v = is_result<T>::value;
}  // namespace detail

// ==============================================
// Error Handling Macros
// ==============================================

#define MATRIX_RESULT(expr, fallback)                          \
  [&]() -> Result<decltype(expr)> {                            \
    try {                                                      \
      if constexpr (std::is_void_v<decltype(expr)>) {          \
        expr;                                                  \
        return Result<void>(ErrorCode{});                      \
      } else {                                                 \
        return Result(std::forward<decltype(expr)>(expr));     \
      }                                                        \
    } catch (const MatrixException& e) {                       \
      return Result<decltype(expr)>(e.code());                 \
    } catch (...) {                                            \
      return Result<decltype(expr)>(ErrorCode::UNKNOWN_ERROR); \
    }                                                          \
  }()

#define MATRIX_RESULT_ERROR_IF(cond, code)                                  \
  do {                                                                      \
    if (cond) {                                                             \
      using SelfT = decltype(*this);                                        \
      using R = std::conditional_t<std::is_lvalue_reference_v<SelfT>,       \
                                   Result<std::remove_reference_t<SelfT>&>, \
                                   Result<SelfT>>;                          \
      return R(code);                                                       \
    }                                                                       \
  } while (0)

// ==============================================
// Inline Implementations
// ==============================================

inline MatrixException::MatrixException(ErrorCode code, std::string msg,
                                        Context ctx)
    : runtime_error(format_msg(msg, code, ctx)),
      m_code(code),
      m_ctx(std::move(ctx)) {}

inline MatrixException::MatrixException(ErrorCode code, std::string msg)
    : MatrixException(code, std::move(msg),
                      Context{std::source_location::current(),
                              std::chrono::system_clock::now(), ""}) {}

inline std::string MatrixException::format_msg(const std::string& msg,
                                               ErrorCode code,
                                               const Context& ctx) {
  std::ostringstream oss;
  oss << "[" << ctx.timestamp << "]\n"
      << "Error " << static_cast<int>(code) << ": " << msg << "\n"
      << "Location: " << ctx.location.file_name() << ":" << ctx.location.line()
      << "\n"
      << "Function: " << ctx.location.function_name();
  if (!ctx.stack_trace.empty()) {
    oss << "\nStack:\n" << ctx.stack_trace;
  }
  return oss.str();
}

}  // namespace algaber