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
#ifndef INCLUDE_MFMM_LAPLACE_H_
#define INCLUDE_MFMM_LAPLACE_H_

#include "fmm.h"
#include "geometry.h"
#include "mfmm.h"
#include "potential_traits.h"
#include "timer.h"

namespace mfmm {

class LaplaceFmmKernel;

using LaplaceFmm = Fmm<LaplaceFmmKernel>;

//! A class defining the kernel function for the Helmholtz FMM.
class LaplaceFmmKernel {
 public:
  using potential_t = double;

  // Argument types required for this kernel.
  using kernel_args_t = std::tuple<>;

 private:
  using pt = potential_traits<potential_t>;
  using real_t = typename pt::real_t;
  using coord_t = typename pt::coord_t;
  using potential_grad_t = typename pt::potential_grad_t;

  static constexpr real_t kernelCoef{real_t(1) / (4 * PI)};

 public:
  LaplaceFmmKernel() = delete;
  // Laplace Kernel takes no arguments.
  LaplaceFmmKernel(kernel_args_t /* kernelArgs */){};

  /** Compute the effect of source with a given strength at a given coordinate.
   * @param sourceCoord The coordinate of the source particle.
   * @param targetCoord The measurement location.
   * @return The potential measured at targetCoord.
   **/
  inline potential_t potential_P2P(const coord_t& sourceCoord,
                                   const coord_t& targetCoord) const noexcept {
    auto radius = (targetCoord - sourceCoord).norm();
    auto potential = radius == 0 ? potential_t(0) : kernelCoef * 1 / radius;
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
    auto coeff = radius == 0 ? 0 : real_t{-1} / (radius * radius);
    auto potential = potential_P2P(sourceCoord, targetCoord);
    return coeff * potential * (targetCoord - sourceCoord);
  }
};
}  // namespace mfmm

#endif  // INCLUDE_MFMM_LAPLACE_H_
