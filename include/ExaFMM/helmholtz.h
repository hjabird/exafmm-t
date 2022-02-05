#pragma once
/******************************************************************************
 *
 * ExaFMM
 * A high-performance fast multipole method library with C++ and
 * python interfaces.
 *
 * Originally Tingyu Wang, Rio Yokota and Lorena A. Barba
 * Modified by HJA Bird
 *
 ******************************************************************************/
#ifndef INCLUDE_EXAFMM_HELMHOLTZ_H_
#define INCLUDE_EXAFMM_HELMHOLTZ_H_
#include "exafmm.h"
#include "fmm.h"
#include "geometry.h"
#include "intrinsics.h"
#include "timer.h"

namespace ExaFMM {

class HelmholtzFmmKernel;

using HelmholtzFmm = Fmm<HelmholtzFmmKernel>;

//! A class defining the kernel function for the Helmholtz FMM.
class HelmholtzFmmKernel {
 public:
  using potential_t = std::complex<double>;

 private:
  using pt = potential_traits<potential_t>;
  template <int Rows = dynamic>
  using potential_vector_t = typename pt::template potential_vector_t<Rows>;
  using complex_t = typename pt::complex_t;
  using coord_t = typename pt::coord_t;
  template <int Rows = dynamic, int RowOrder = column_major>
  using coord_matrix_t = typename pt::template coord_matrix_t<Rows, RowOrder>;
  using potential_grad_t = typename pt::potential_grad_t;
  template <int Rows = dynamic>
  using potential_grad_vector_t = typename pt::potential_grad_vector_t<Rows>;

 public:
  // Argument types required for this kernel.
  using kernel_args_t = std::tuple<complex_t>;

  const complex_t wavek;  //!< Wave number k.

 private:
  const complex_t kappa;  //!< K used in Helmholtz kernel.
  const real_t kernelCoef;

 public:
  HelmholtzFmmKernel() = delete;

  HelmholtzFmmKernel(kernel_args_t kernelArgs)
      : wavek{std::get<0>(kernelArgs)},
        kappa{wavek / complex_t{16}},
        kernelCoef{real_t(1) / (64 * PI)} {};

  /** Compute the effect of source with a given strength at a given coordinate.
   * @param sourceCoord The coordinate of the source particle.
   * @param targetCoord The measurement location.
   * @return The potential measured at targetCoord.
   **/
  inline potential_t potential_P2P(const coord_t& sourceCoord,
                                   const coord_t& targetCoord) const noexcept {
    auto radius = (sourceCoord - targetCoord).norm();
    auto potential =
        radius == 0
            ? 0
            : kernelCoef * std::exp(radius * kappa * complex_t{0, 1}) / radius;
    return potential;
  }

  /** Compute the effect of source with a given strength at a given coordinate.
   * @param sourceCoord The coordinate of the source particle.
   * @param sourceStrength The strength of the source particle.
   * @param targetCoord The measurement location.
   * @return The potential measured at targetCoord.
   **/
  inline potential_t potential_P2P(const coord_t& sourceCoord,
                                   const potential_t& sourceStrength,
                                   const coord_t& targetCoord) const noexcept {
    return sourceStrength * potential_P2P(sourceCoord, targetCoord);
  }

  template <int NumSources, int NumTargets, int SourceRowOrder,
            int TargetRowOrder>
  potential_vector_t<NumTargets> potential_P2P(
      const coord_matrix_t<NumSources, SourceRowOrder>& sourceCoords,
      const potential_vector_t<NumSources>& sourceStrengths,
      const coord_matrix_t<NumTargets, TargetRowOrder>& targetCoords) {
    const size_t numSources = static_cast<int>(sourceCoords.rows());
    const size_t numTargets = static_cast<int>(targetCoords.rows());
    potential_vector_t<NumTargets> targetValues =
        potential_vector_t<NumTargets>::Zero(numTargets);
    for (size_t i{0}; i < numTargets; ++i) {
      for (size_t j{0}; j < numSources; ++j) {
        targetValues(i) += potential_P2P(
            sourceCoords.row(j), sourceStrengths(j), targetCoords.row(i));
      }
    }
    return targetValues;
  }

  /** Compute potentials and gradients at targets induced by sources
   * directly.
   * @param sourceCoords Vector of coordinates of sources.
   * @param targetCoords Vector of coordinates of targets.
   * @return Vector of potentials of targets.
   */
  inline potential_grad_t gradient_P2P(
      const coord_t& sourceCoord, const coord_t& targetCoord) const noexcept {
    auto radius = (sourceCoord - targetCoord).norm();
    auto coeff = radius == 0 ? 0
                             : complex_t{(1 + kappa.imag()) / (radius * radius),
                                         -kappa.real() / radius};
    auto potential = potential_P2P(sourceCoord, targetCoord);
    return coeff * potential * (sourceCoord - targetCoord);
  }

  inline potential_grad_t gradient_P2P(
      const coord_t& sourceCoord, const potential_t& sourceStrength,
      const coord_t& targetCoord) const noexcept {
    return sourceStrength * gradient_P2P(sourceCoord, targetCoord);
  }

  /** Compute potentials and gradients at targets induced by sources
   * directly.
   *
   * @param sourceCoords Vector of coordinates of sources.
   * @param sourceStrengths Vector of charges of sources.
   * @param targetCoords Vector of coordinates of targets.
   * @return Vector of potentials of targets.
   */
  template <int NumSources, int NumTargets, int SourceRowOrder,
            int TargetRowOrder>
  potential_grad_vector_t<NumTargets> gradient_P2P(
      const coord_matrix_t<NumSources, SourceRowOrder>& sourceCoords,
      const potential_vector_t<NumSources>& sourceStrengths,
      const coord_matrix_t<NumTargets, TargetRowOrder>& targetCoords) {
    const size_t numSources = static_cast<int>(sourceCoords.rows());
    const size_t numTargets = static_cast<int>(targetCoords.rows());
    potential_grad_vector_t<NumTargets> targetValues =
        potential_grad_vector_t<NumTargets>::Zero(numTargets, 3);
    for (size_t i{0}; i < numTargets; ++i) {
      for (size_t j{0}; j < numSources; ++j) {
        targetValues.row(i) += gradient_P2P(
            sourceCoords.row(j), sourceStrengths(j, 0), targetCoords.row(i));
      }
    }
    return targetValues;
  }
};
}  // namespace ExaFMM
#endif  // INCLUDE_EXAFMM_HELMHOLTZ_H_
