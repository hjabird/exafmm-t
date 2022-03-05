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
#include "relative_coords.h"

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
typename potential_traits<PotentialT>::template coord_matrix_t<dynamic>
convolution_grid(int p, typename potential_traits<PotentialT>::real_t r0,
                 int level,
                 typename potential_traits<PotentialT>::coord_t& boxCentre) {
  using real_t = typename potential_traits<PotentialT>::real_t;
  using coord_t = typename potential_traits<PotentialT>::coord_t;

  real_t d = 2 * r0 * std::pow(0.5, level);
  real_t a = d * 1.05;  // side length of upward equivalent/downward check box
  int n1 = p * 2;
  int n2 = n1 * n1;
  int n3 = n1 * n1 * n1;
  typename potential_traits<PotentialT>::template coord_matrix_t<dynamic>
      gridCoords(n3, 3);
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n1; j++) {
      for (int k = 0; k < n1; k++) {
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

//! Generate a map that maps indices of M2L_Type to indices of M2L_Helper_Type
void generate_M2L_index_map() {
  // The number of relative coords for M2L_Type:
  constexpr int nPos = static_cast<int>(REL_COORD_M2L.size());  
  M2L_INDEX_MAP.resize(nPos, std::vector<int>(NCHILD * NCHILD));
#pragma omp parallel for
  for (int i = 0; i < nPos; ++i) {
    for (int j1 = 0; j1 < NCHILD; ++j1) {
      for (int j2 = 0; j2 < NCHILD; ++j2) {
        ivec3 parentRelCoord = REL_COORD_M2L[i];
        ivec3 childRelCoord;
        childRelCoord[0] = 
            parentRelCoord[0] * 2 - (j1 / 1) % 2 + (j2 / 1) % 2;
        childRelCoord[1] =
            parentRelCoord[1] * 2 - (j1 / 2) % 2 + (j2 / 2) % 2;
        childRelCoord[2] =
            parentRelCoord[2] * 2 - (j1 / 4) % 2 + (j2 / 4) % 2;
        int childRelIdx = REL_COORD_M2L_helper.hash(childRelCoord);
        int j = j2 * NCHILD + j1;
        M2L_INDEX_MAP[i][j] = childRelIdx;
      }
    }
  }
}

//! Compute the relative positions for all operators and generate M2L index
//! mapping.
void init_rel_coord() {
  static bool is_initialized = false;
  if (!is_initialized) {
    generate_M2L_index_map();
    is_initialized = true;
  }
}
}  // namespace ExaFMM
#endif  // INCLUDE_EXAFMM_GEOMETRY_H_
