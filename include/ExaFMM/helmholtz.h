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
#include "potential_traits.h"
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
  using real_t = typename pt::real_t;
  using complex_t = typename pt::complex_t;
  using coord_t = typename pt::coord_t;
  using potential_grad_t = typename pt::potential_grad_t;

 public:
  // Argument types required for this kernel.
  using kernel_args_t = std::tuple<complex_t>;

  const complex_t wavek;  //!< Wave number k.

 private:
  static constexpr real_t kernelCoef{ real_t(1) / (4 * PI) };

 public:
  HelmholtzFmmKernel() = delete;

  HelmholtzFmmKernel(kernel_args_t kernelArgs)
      : wavek{std::get<0>(kernelArgs)} {};

  /** Compute the effect of source with a given strength at a given coordinate.
   * @param sourceCoord The coordinate of the source particle.
   * @param targetCoord The measurement location.
   * @return The potential measured at targetCoord.
   **/
  inline potential_t potential_P2P(const coord_t& sourceCoord,
                                   const coord_t& targetCoord) const noexcept {
    auto radius = (targetCoord - sourceCoord).norm();
    auto potential =
        radius == 0
            ? 0
            : kernelCoef * std::exp(radius * wavek * complex_t{0, 1}) / radius;
    return potential;
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
                             : (radius * complex_t{0, 1} * wavek - real_t{1}) /
                                   (radius * radius);
    auto potential = potential_P2P(sourceCoord, targetCoord);
    return coeff * potential * (targetCoord - sourceCoord);
  }
};
}  // namespace ExaFMM
#endif  // INCLUDE_EXAFMM_HELMHOLTZ_H_
