#pragma once
/******************************************************************************
 *
 * mfmm
 * A high-performance fast multipole method library using C++.
 *
 * A fork of ExaFMM (BSD-3-Clause lisence).
 * Originally copyright Wang, Yokota and Barba.
 *
 * Modifications copyright HJA Bird.
 *
 ******************************************************************************/
#ifndef INCLUDE_MFMM_POTENTIAL_TRAITS_H_
#define INCLUDE_MFMM_POTENTIAL_TRAITS_H_

#include <Eigen/Dense>
#include <complex>
#include <type_traits>

namespace mfmm {

/// Use row-major memory order. Like C.
constexpr int row_major = Eigen::RowMajor;
/// Use column-major memory order. Like fortran.
constexpr int column_major = Eigen::ColMajor;
/// Describes resizable matrices.
constexpr int dynamic = Eigen::Dynamic;

/** Helper class to find whether a type is complex.
 *
 * @tparam T The type to test is std::complex.
 **/
template <typename T>
class is_complex : public std::false_type {};

template <typename T>
class is_complex<std::complex<T>> : public std::true_type {};

/** The value_type of a real or complex type.
 *
 * @tparam T The type to get the value real type of.
 **/
template <typename T>
class real_type_of {
 public:
  using type = T;
};

template <typename T>
class real_type_of<std::complex<T>> {
 public:
  using type = T;
};

/** Traits needed for an FMM with given potential type.
 *
 * @tparam PotentialT The type of the potential.
 **/
template <typename PotentialT>
struct potential_traits {
  /// The type of the potential.
  using potential_t = PotentialT;
  /// A real type with the same precision as the potential_t.
  using real_t = typename real_type_of<potential_t>::type;
  /// A complex type with the same precision as the potential_t.
  using complex_t = std::complex<real_t>;

  /// True if the potential type is complex.
  static constexpr bool isComplexPotential = is_complex<potential_t>::value;
  /// The epsilon of the real_t.
  static constexpr real_t epsilon = std::numeric_limits<real_t>::epsilon();

  /// A matrix of the potential_t.
  template <int Rows = dynamic, int Cols = dynamic, int RowOrder = row_major>
  using potential_matrix_t = Eigen::Matrix<potential_t, Rows, Cols, RowOrder>;
  /// A vector of the potential_t.
  template <int Rows = dynamic>
  using potential_vector_t = Eigen::Matrix<potential_t, Rows, 1>;
  /// A type to represent the grad of the potential
  using potential_grad_t = Eigen::Matrix<potential_t, 1, 3>;
  /// Represents a vector of potential_grad_t.
  template <int Rows = dynamic, int RowOrder = column_major>
  using potential_grad_vector_t = Eigen::Matrix<potential_t, Rows, 3, RowOrder>;
  /// A matrix of the real_t.
  template <int Rows = dynamic, int Cols = dynamic, int RowOrder = row_major>
  using real_matrix_t = Eigen::Matrix<real_t, Rows, Cols, RowOrder>;
  /// A vector of the real_t.
  template <int Rows = dynamic>
  using real_vector_t = Eigen::Matrix<real_t, Rows, 1>;
  /// A matrix of the complex_t.
  template <int Rows = dynamic, int Cols = dynamic, int RowOrder = row_major>
  using complex_matrix_t =
      typename Eigen::Matrix<complex_t, Rows, Cols, RowOrder>;
  /// A vector of the complex_t.
  template <int Rows = dynamic>
  using complex_vector_t = Eigen::Matrix<complex_t, Rows, 1>;

  /// Coordinate vector.
  using coord_t = Eigen::Matrix<real_t, 1, 3>;
  /// A vector (as in multiple) coordinates
  template <int Rows = dynamic, int RowOrder = column_major>
  using coord_matrix_t = Eigen::Matrix<real_t, Rows, 3, RowOrder>;
};
}  // namespace mfmm

#endif  // INCLUDE_MFMM_POTENTIAL_TRAITS_H_
