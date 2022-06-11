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
#ifndef INCLUDE_MFMM_P2P_METHODS_H_
#define INCLUDE_MFMM_P2P_METHODS_H_

#include "fmm.h"
#include "geometry.h"
#include "mfmm.h"
#include "potential_traits.h"

namespace mfmm {

/** Extended P2P methods for a P2P kernel.
 * @tparam Kernel The P2P kernel.
 **/
template <class Kernel>
class p2p_methods : public Kernel {
 public:
  using potential_t = typename Kernel::potential_t;

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
  using potential_grad_vector_t =
      typename pt::template potential_grad_vector_t<Rows>;

 public:
  /// Inherit kernel class's constructors.
  using Kernel::Kernel;

  // Inherit kernel class's potential_P2P
  using Kernel::potential_P2P;
  // Inherit kernel class's gradient_P2P
  using Kernel::gradient_P2P;

  /** Compute the effect of source with a given strength at a given coordinate.
   * @param sourceCoord The coordinate of the source particle.
   * @param sourceStrength The strength of the source particle.
   * @param targetCoord The measurement location.
   * @return The potential measured at targetCoord.
   **/
  inline potential_t potential_P2P(const coord_t& sourceCoord,
                                   const potential_t& sourceStrength,
                                   const coord_t& targetCoord) const noexcept {
    return sourceStrength * this->potential_P2P(sourceCoord, targetCoord);
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
        targetValues(i, 0) += this->potential_P2P(
            sourceCoords.row(j), sourceStrengths(j, 0), targetCoords.row(i));
      }
    }
    return targetValues;
  }

  inline potential_grad_t gradient_P2P(
      const coord_t& sourceCoord, const potential_t& sourceStrength,
      const coord_t& targetCoord) const noexcept {
    return sourceStrength * this->gradient_P2P(sourceCoord, targetCoord);
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
        targetValues.row(i) += this->gradient_P2P(
            sourceCoords.row(j), sourceStrengths(j, 0), targetCoords.row(i));
      }
    }
    return targetValues;
  }
};
}  // namespace mfmm

#endif  // INCLUDE_MFMM_P2P_METHODS_H_
