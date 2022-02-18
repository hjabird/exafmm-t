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

#include <set>

#include "args.h"
#include "fft.h"
#include "node.h"
#include "potential_traits.h"

namespace ExaFMM {

const int MEM_ALIGN = 64;
const int CACHE_SIZE = 512;
const int NCHILD = 8;

static constexpr auto PI = real_t(3.141592653589793238462643383279502884);
using complex_t = std::complex<real_t>;  //!< Complex number type
typedef Eigen::Vector3i ivec3;           //!< Vector of 3 int types

typedef std::vector<real_t> RealVec;        //!< Vector of real_t types
typedef std::vector<complex_t> ComplexVec;  //!< Vector of complex_t types
typedef std::vector<real_t> AlignedVec;     //!< Aligned vector of real_t types

//! Interaction types that need to be pre-computed.
typedef enum {
  M2M_Type = 0,
  L2L_Type = 1,
  M2L_Helper_Type = 2,
  M2L_Type = 3,
  Type_Count = 4
} Precompute_Type;

// alias template
using Keys =
    std::vector<std::set<uint64_t>>;  //!< Vector of Morton keys of each level

//! M2L setup data
struct M2LData {
  std::vector<size_t>
      fft_offset;  // source's first child's upward_equiv's displacement
  std::vector<size_t>
      ifft_offset;  // target's first child's dnward_equiv's displacement
  RealVec ifft_scale;
  std::vector<size_t> interaction_offset_f;
  std::vector<size_t> interaction_count_offset;
};

// Relative coordinates and interaction lists
std::vector<std::vector<ivec3>>
    REL_COORD;  //!< Vector of possible relative coordinates (inner) of each
                //!< interaction type (outer)
std::vector<std::vector<int>>
    HASH_LUT;  //!< Vector of hash Lookup tables (inner) of relative positions
               //!< for each interaction type (outer)
std::vector<std::vector<int>>
    M2L_INDEX_MAP;  //!< [M2L_relpos_idx][octant] -> M2L_Helper_relpos_idx
}  // namespace ExaFMM
#endif  // INCLUDE_EXAFMM_EXAFMM_H_
