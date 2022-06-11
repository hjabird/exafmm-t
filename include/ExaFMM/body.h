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
#ifndef INCLUDE_EXAFMM_BODY_H_
#define INCLUDE_EXAFMM_BODY_H_

#include <vector>

#include "potential_traits.h"

namespace ExaFMM {

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
}  // namespace ExaFMM

#endif  // INCLUDE_EXAFMM_BODY_H_
