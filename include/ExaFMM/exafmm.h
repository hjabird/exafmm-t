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
#ifndef INCLUDE_EXAFMM_EXAFMM_H_
#define INCLUDE_EXAFMM_EXAFMM_H_

#include <omp.h>

#include <cmath>
#include <map>
#include <set>
#include <vector>

#include "args.h"
#include "fft.h"
#include "node.h"
#include "octree_location.h"
#include "potential_traits.h"
#include "relative_coords.h"

namespace ExaFMM {

const int MEM_ALIGN = 64;
const int CACHE_SIZE = 512;
const int NCHILD = 8;

static constexpr auto PI = double(3.141592653589793238462643383279502884);

//! Interaction types that need to be pre-computed.
typedef enum {
  M2M_Type = 0,
  L2L_Type = 1,
  M2L_Helper_Type = 2,
  M2L_Type = 3,
  Type_Count = 4
} Precompute_Type;

// alias template
using Keys = std::vector<std::set<octree_location>>;  //!< Vector of Morton keys
                                                      //!< of each level

detail::relative_coord_mapping<detail::coords_M2M> REL_COORD_M2M;
detail::relative_coord_mapping<detail::coords_L2L> REL_COORD_L2L;
detail::relative_coord_mapping<detail::coords_M2L> REL_COORD_M2L;
detail::relative_coord_mapping<detail::coords_M2L_helper> REL_COORD_M2L_helper;

//! M2L setup data
template <typename RealT>
struct M2LData {
  std::vector<size_t>
      fft_offset;  // source's first child's upward_equiv's displacement
  std::vector<size_t>
      ifft_offset;  // target's first child's dnward_equiv's displacement
  std::vector<RealT> ifft_scale;
  // Offset pairs of {sourceNodeOffset, TargetNodeOffset}.
  std::vector<std::pair<size_t, size_t>> interaction_offset_f;
  std::array<size_t, REL_COORD_M2L.size()> interaction_count_offset;
};

detail::M2L_idx_map
    M2L_INDEX_MAP;  //!< [M2L_relpos_idx][octant] -> M2L_Helper_relpos_idx
}  // namespace ExaFMM
#endif  // INCLUDE_EXAFMM_EXAFMM_H_
