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
#ifndef INCLUDE_MFMM_MFMM_H_
#define INCLUDE_MFMM_MFMM_H_

#include <omp.h>

#include <cmath>
#include <map>
#include <set>
#include <vector>

#include "fft.h"
#include "node.h"
#include "octree_location.h"
#include "potential_traits.h"
#include "relative_coords.h"

namespace mfmm {

const int NCHILD = 8;

static constexpr auto PI = double(3.141592653589793238462643383279502884);

// alias template
using Keys = std::vector<std::set<octree_location>>;  //!< Vector of Morton keys
                                                      //!< of each level

detail::relative_coord_mapping<detail::coords_M2M> REL_COORD_M2M;
detail::relative_coord_mapping<detail::coords_L2L> REL_COORD_L2L;
detail::relative_coord_mapping<detail::coords_M2L> REL_COORD_M2L;
detail::relative_coord_mapping<detail::coords_M2L_helper> REL_COORD_M2L_helper;

/** M2L precomputation offset data.
 * @tparam RealT The type of real value to use.
 **/
template <typename RealT>
struct M2LData {
  /// Source's first child's upward_equiv's displacement
  std::vector<size_t> m_fftOffset;
  /// Target's first child's dnward_equiv's displacement
  std::vector<size_t> m_ifftOffset;
  std::vector<RealT> m_ifftScale;
  // Offset pairs of {sourceNodeOffset, TargetNodeOffset}.
  std::vector<std::pair<size_t, size_t>> m_interactionOffsetF;
  std::array<size_t, REL_COORD_M2L.size()> m_interactionCountOffset;
};

//// [M2L_relpos_idx][octant] -> M2L_Helper_relpos_idx
detail::M2L_idx_map M2L_INDEX_MAP;

}  // namespace mfmm
#endif  // INCLUDE_MFMM_MFMM_H_
