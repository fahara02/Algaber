#ifndef MATRIX_ERROR_HPP
#define MATRIX_ERROR_HPP

#include <cassert>
#include "exception"
namespace algaber {
struct divide_by_zero : public std::exception {
  virtual const char* what() const throw() {
    return "division by zero occured!";
  }
};
struct bad_size : public std::exception {
  virtual const char* what() const throw() {
    return "matrix row or column size is incompatible";
  }
};
struct incompatible_size : public std::exception {
  virtual const char* what() const throw() {
    return "matrix/matricies not compatible sizes";
  }
};
struct bad_argument : public std::exception {
  virtual const char* what() const throw() {
    return "invalid argument to function";
  }
};

struct out_of_range : public std::exception {
  virtual const char* what() const throw() {
    return "row or column index is out of range";
  }
};
struct row_outbound : public out_of_range {
  const char* what() const throw() {
    return "indexing of matrix row is out of bound";
  }
};
struct col_outbound : public out_of_range {
  const char* what() const throw() {
    return "indexing of matrix column is out of bound";
  }
};
struct too_many_rows : public bad_argument {
  const char* what() const throw() { return "only one row at a time"; }
};

struct too_many_cols : public bad_argument {
  const char* what() const throw() { return "only one column at a time"; }
};

struct bad_rows : public bad_size {
  const char* what() const throw() {
    return "new row vector column size do not match ";
  }
};

struct bad_cols : public bad_size {
  const char* what() const throw() {
    return "new column vector row size do not match ";
  }
};
struct bad_row_match : public bad_size {
  const char* what() const throw() {
    return "new vector's row size dont match existing row ";
  }
};

struct bad_col_match : public bad_size {
  const char* what() const throw() {
    return "new vector's column size dont match existing column ";
  }
};
struct not_square : public bad_size {
  const char* what() const throw() { return "matrix must be square"; }
};
struct not_invertible : public std::exception {
  const char* what() const throw() { return "matrix is not invertible"; }
};
struct not_solvable : public std::exception {
  const char* what() const throw() { return "System is not solvable"; }
};
}  // namespace algaber
#endif