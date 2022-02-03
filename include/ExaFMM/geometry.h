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
#ifndef INCLUDE_EXAFMM_GEOMETRY_H_
#define INCLUDE_EXAFMM_GEOMETRY_H_

#include "exafmm.h"
#include "predefines.h"

namespace ExaFMM {
// Global variables REL_COORD, HASH_LUT, M2L_INDEX_MAP are now defined in
// exafmm_t.h.

/** Given a box, calculate the coordinates of surface points.
 *
 * Returns 6 (p-1)^2 + 2 coordinates:
 *   * 8 for corners
 *   * 12 * (p - 2) equispaced on the edges
 *   * 6 * (p - 2) ^ 2 on remaining surface grid points.
 *
 * @param p Order of expansion.
 * @param r0 Half side length of the bounding box (root node).
 * @param level Level of the box.
 * @param boxCentre Coordinates of the center of the box.
 * @param alpha Ratio between the side length of surface box and original box.
 *              Use 2.95 for upward check and downward equivalent surface,
 *              use 1.05 for upward equivalent and downward check surface.
 *
 * @return Vector of coordinates of surface points.
 */
template <typename PotentialT>
typename potential_traits<PotentialT>::template coord_matrix_t<dynamic>
box_surface_coordinates(
    int p, typename potential_traits<PotentialT>::real_t r0, int level,
    typename potential_traits<PotentialT>::coord_t boxCentre,
    typename potential_traits<PotentialT>::real_t alpha) {
  EXAFMM_ASSERT(p > 0);
  EXAFMM_ASSERT(r0 > 0);
  EXAFMM_ASSERT(level >= 0);
  EXAFMM_ASSERT(alpha > 0);
  using coord_t = typename potential_traits<PotentialT>::coord_t;
  using coord_matrix_t =
      typename potential_traits<PotentialT>::template coord_matrix_t<dynamic>;
  using real_t = typename potential_traits<PotentialT>::real_t;

  int n = 6 * (p - 1) * (p - 1) + 2;
  coord_matrix_t surfaceCoords(n, 3);
  surfaceCoords.row(0) = coord_t{-1.0, -1.0, -1.0};
  int count = 1;
  for (int i = 0; i < p - 1; i++) {
    for (int j = 0; j < p - 1; j++) {
      surfaceCoords.row(count) = coord_t{
          -1.0, (2.0 * (i + 1) - p + 1) / (p - 1), (2.0 * j - p + 1) / (p - 1)};
      count++;
    }
  }
  for (int i = 0; i < p - 1; i++) {
    for (int j = 0; j < p - 1; j++) {
      surfaceCoords.row(count) = coord_t{(2.0 * i - p + 1) / (p - 1), -1.0,
                                         (2.0 * (j + 1) - p + 1) / (p - 1)};
      count++;
    }
  }
  for (int i = 0; i < p - 1; i++) {
    for (int j = 0; j < p - 1; j++) {
      surfaceCoords.row(count) = coord_t{(2.0 * (i + 1) - p + 1) / (p - 1),
                                         (2.0 * j - p + 1) / (p - 1), -1.0};
      count++;
    }
  }
  // The reflection of all prior points for +ve side surfaces.
  surfaceCoords.bottomRows(n / 2) = -surfaceCoords.topRows(n / 2);
  // Scale the points from {2,2,2} size centred on {0,0,0} to desired size and
  // location.
  real_t r = r0 * std::pow(static_cast<real_t>(.5), level);
  real_t b = alpha * r;
  surfaceCoords *= b;
  surfaceCoords.rowwise() += boxCentre;
  return surfaceCoords;
}

/** Generate the convolution grid of a given box.
 *
 * @tparam PotentialT The potential type used in the problem.
 * @param p Order of expansion.
 * @param r0 Half side length of the bounding box (root node).
 * @param level Level of the box.
 * @param boxCentre Coordinates of the center of the box.
 *
 * @return Vector of coordinates of convolution grid.
 */
template <typename PotentialT>
potential_traits<PotentialT>::coord_matrix_t<dynamic> convolution_grid(
    int p, typename potential_traits<PotentialT>::real_t r0, int level,
    typename potential_traits<PotentialT>::coord_t& boxCentre) {
  EXAFMM_ASSERT(p > 0);
  EXAFMM_ASSERT(r0 > 0);
  EXAFMM_ASSERT(level >= 0);
  using real_t = typename potential_traits<PotentialT>::real_t;
  using coord_t = typename potential_traits<PotentialT>::coord_t;

  real_t d = 2 * r0 * std::pow(0.5, level);
  real_t a = d * 1.05;  // side length of upward equivalent/downward check box
  int n1 = p * 2;
  int n2 = n1 * n1;
  int n3 = n1 * n1 * n1;
  potential_traits<PotentialT>::coord_matrix_t<dynamic> gridCoords(n3, 3);
  for (size_t i = 0; i < n1; i++) {
    for (size_t j = 0; j < n1; j++) {
      for (size_t k = 0; k < n1; k++) {
        gridCoords.row(i + n1 * j + n2 * k) =
            coord_t({(i - p) * a / (p - 1), (j - p) * a / (p - 1),
                     (k - p) * a / (p - 1)});
        gridCoords.row(i + n1 * j + n2 * k) += boxCentre;
      }
    }
  }
  return gridCoords;
}

/** Generate the mapping from surface points to convolution grid used in FFT.
 *
 * @param p Order of expansion.
 *
 * @return A mapping from upward equivalent surface point index to convolution
 * grid index.
 */
template <typename PotentialT>
std::vector<int> generate_surf2conv_up(int p) {
  using real_t = typename potential_traits<PotentialT>::real_t;
  using coord_t = typename potential_traits<PotentialT>::coord_t;
  EXAFMM_ASSERT(p > 0);
  int n1 = 2 * p;
  coord_t c = coord_t::Ones(3) * 0.5 * (p - 1);
  auto surf = box_surface_coordinates<PotentialT>(p, 0.5, 0, c, real_t(p - 1));
  std::vector<int> map(6 * (p - 1) * (p - 1) + 2);
  for (size_t i = 0; i < map.size(); i++) {
    map[i] = (int)(p - 1 - surf(i, 0)) + ((int)(p - 1 - surf(i, 1))) * n1 +
             ((int)(p - 1 - surf(i, 2))) * n1 * n1;
  }
  return map;
}

/**
 * @brief Generate the mapping from surface points to convolution grid used in
 * IFFT.
 *
 * @param p Order of expansion.
 *
 * @return A mapping from downward check surface point index to convolution grid
 * index.
 */
template <typename PotentialT>
std::vector<int> generate_surf2conv_dn(int p) {
  using real_t = typename potential_traits<PotentialT>::real_t;
  using coord_t = typename potential_traits<PotentialT>::coord_t;
  EXAFMM_ASSERT(p > 0);
  int n1 = 2 * p;
  coord_t c;
  for (int d = 0; d < 3; d++) {
    c(d) = 0.5 * (p - 1);
  }
  auto surf = box_surface_coordinates<PotentialT>(p, 0.5, 0, c, real_t(p - 1));
  std::vector<int> map(6 * (p - 1) * (p - 1) + 2);
  for (size_t i = 0; i < map.size(); i++) {
    map[i] = (int)(2 * p - 1 - surf(i, 0)) +
             ((int)(2 * p - 1 - surf(i, 1))) * n1 +
             ((int)(2 * p - 1 - surf(i, 2))) * n1 * n1;
  }
  return map;
}

/** Compute the hash value of a relative position (coordinates).
 *
 * @param coord Coordinates that represent a relative position.
 *
 * @return Hash value of the relative position (x + 10y + 100z + 555).
 */
int hash(ivec3& coord) {
  const int n = 5;
  return ((coord[2] + n) * (2 * n) + (coord[1] + n)) * (2 * n) + (coord[0] + n);
}

/** Compute the coordinates of possible relative positions for operator t.
 *
 * @param max_r Max range.
 * @param min_r Min range.
 * @param step Step.
 * @param t Operator type (e.g. M2M, M2L)
 */
void init_rel_coord(int max_r, int min_r, int step, Precompute_Type t) {
  const int max_hash = 2000;
  HASH_LUT[t].resize(max_hash, -1);
  for (int k = -max_r; k <= max_r; k += step) {
    for (int j = -max_r; j <= max_r; j += step) {
      for (int i = -max_r; i <= max_r; i += step) {
        if (abs(i) >= min_r || abs(j) >= min_r || abs(k) >= min_r) {
          ivec3 coord;
          coord[0] = i;
          coord[1] = j;
          coord[2] = k;
          REL_COORD[t].push_back(coord);
          HASH_LUT[t][hash(coord)] = static_cast<int>(REL_COORD[t].size() - 1);
        }
      }
    }
  }
}

//! Generate a map that maps indices of M2L_Type to indices of M2L_Helper_Type
void generate_M2L_index_map() {
  int npos = static_cast<int>(
      REL_COORD[M2L_Type].size());  // number of relative coords for M2L_Type
  M2L_INDEX_MAP.resize(npos, std::vector<int>(NCHILD * NCHILD));
#pragma omp parallel for
  for (int i = 0; i < npos; ++i) {
    for (int j1 = 0; j1 < NCHILD; ++j1) {
      for (int j2 = 0; j2 < NCHILD; ++j2) {
        ivec3& parent_rel_coord = REL_COORD[M2L_Type][i];
        ivec3 child_rel_coord;
        child_rel_coord[0] =
            parent_rel_coord[0] * 2 - (j1 / 1) % 2 + (j2 / 1) % 2;
        child_rel_coord[1] =
            parent_rel_coord[1] * 2 - (j1 / 2) % 2 + (j2 / 2) % 2;
        child_rel_coord[2] =
            parent_rel_coord[2] * 2 - (j1 / 4) % 2 + (j2 / 4) % 2;
        int coord_hash = hash(child_rel_coord);
        int child_rel_idx = HASH_LUT[M2L_Helper_Type][coord_hash];
        int j = j2 * NCHILD + j1;
        M2L_INDEX_MAP[i][j] = child_rel_idx;
      }
    }
  }
}

//! Compute the relative positions for all operators and generate M2L index
//! mapping.
void init_rel_coord() {
  static bool is_initialized = false;
  if (!is_initialized) {
    REL_COORD.resize(Type_Count);
    HASH_LUT.resize(Type_Count);
    init_rel_coord(1, 1, 2, M2M_Type);
    init_rel_coord(1, 1, 2, L2L_Type);
    init_rel_coord(3, 2, 1, M2L_Helper_Type);
    init_rel_coord(1, 1, 1, M2L_Type);
    generate_M2L_index_map();
    is_initialized = true;
  }
}
}  // namespace ExaFMM
#endif  // INCLUDE_EXAFMM_GEOMETRY_H_
