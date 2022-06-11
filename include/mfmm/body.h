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
#ifndef INCLUDE_MFMM_BODY_H_
#define INCLUDE_MFMM_BODY_H_

#include <vector>

#include "potential_traits.h"

namespace mfmm {

/**
 * @brief Structure of bodies.
 *
 * @tparam PotentialT Value type of sources and targets (real or complex).
 */
template <typename PotentialT>
struct Body {
  int ibody;  //!< Initial body numbering for sorting back
  typename potential_traits<PotentialT>::coord_t X;           //!< Coordinates
  PotentialT q;                                               //!< Charge
  PotentialT p;                                               //!< Potential
  typename potential_traits<PotentialT>::potential_grad_t F;  //!< Gradient
};

template <typename PotentialT>
using Bodies = std::vector<Body<PotentialT>>;  //!< Vector of nodes
}  // namespace mfmm

#endif  // INCLUDE_MFMM_BODY_H_
