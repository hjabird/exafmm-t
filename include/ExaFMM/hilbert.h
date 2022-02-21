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
#ifndef INCLUDE_EXAFMM_HILBERT_H_
#define INCLUDE_EXAFMM_HILBERT_H_
#include <stdint.h>

#include <cstdlib>

#include "exafmm.h"
#define EXAFMM_HILBERT 0  //! Set this to 1 for Hilbert

namespace ExaFMM {
/**
 * @brief Calculate levelwise offset of Hilbert key.
 *
 * @param level Level.
 * @return uint64_t Level offset.
 */
inline uint64_t levelOffset(int level) {
  return (((uint64_t)1 << 3 * level) - 1) / 7;
}

/**
 * @brief Get level from a Hilbert key.
 *
 * @param i Hilbert key.
 * @return int Level.
 */
int getLevel(uint64_t i) {
  int level = -1;
  uint64_t offset = 0;
  while (i >= offset) {
    level++;
    offset += (uint64_t)1 << 3 * level;
  }
  return level;
}

/**
 * @brief Get parent's Hilbert key with level offset.
 *
 * @param i Hilbert key of a node with level offset.
 * @return uint64_t Parent's Hilbert key.
 */
uint64_t getParent(uint64_t i) {
  int level = getLevel(i);
  return (i - levelOffset(level)) / 8 + levelOffset(level - 1);
}

/**
 * @brief Get first child's Hilbert key with level offset.
 *
 * @param i Hilbert key of a node with level offset.
 * @return uint64_t First child's Hilbert key.
 */
uint64_t getChild(uint64_t i) {
  int level = getLevel(i);
  return (i - levelOffset(level)) * 8 + levelOffset(level + 1);
}

/**
 * @brief Determine which octant the key belongs to.
 *
 * @param key Hilbert key.
 * @param offset Whether the key contains level offset, default to true.
 * @return int Octant.
 */
//!
int getOctant(uint64_t key, bool offset = true) {
  int level = getLevel(key);
  if (offset) key -= levelOffset(level);
  return key & 7;
}

/**
 * @brief Get Hilbert key from 3D index of a node.
 *
 * @param iX 3D index of a node, an integer triplet.
 * @param level Level of the node.
 * @param offset Whether to add level offset to the key, default to true.
 * @return uint64_t Hilbert key.
 */
uint64_t getKey(ivec3 iX, int level, bool offset = true) {
#if EXAFMM_HILBERT
  int M = 1 << (level - 1);
  for (int Q = M; Q > 1; Q >>= 1) {
    int R = Q - 1;
    for (int d = 0; d < 3; d++) {
      if (iX[d] & Q)
        iX[0] ^= R;
      else {
        int t = (iX[0] ^ iX[d]) & R;
        iX[0] ^= t;
        iX[d] ^= t;
      }
    }
  }
  for (int d = 1; d < 3; d++) iX[d] ^= iX[d - 1];
  int t = 0;
  for (int Q = M; Q > 1; Q >>= 1)
    if (iX[2] & Q) t ^= Q - 1;
  for (int d = 0; d < 3; d++) iX[d] ^= t;
#endif
  uint64_t i = 0;
  for (int l = 0; l < level; l++) {
    i |= (iX[2] & (uint64_t)1 << l) << 2 * l;
    i |= (iX[1] & (uint64_t)1 << l) << (2 * l + 1);
    i |= (iX[0] & (uint64_t)1 << l) << (2 * l + 2);
  }
  if (offset) i += levelOffset(level);
  return i;
}

/**
 * @brief Get 3D index from a Hilbert key with level offset.
 *
 * @param i Hilbert key with level offset.
 * @return ivec3 3D index, an integer triplet.
 */
ivec3 get3DIndex(uint64_t i) {
  int level = getLevel(i);
  i -= levelOffset(level);
  ivec3 iX = {0, 0, 0};
  for (int l = 0; l < level; l++) {
    iX[2] |= (i & (uint64_t)1 << 3 * l) >> 2 * l;
    iX[1] |= (i & (uint64_t)1 << (3 * l + 1)) >> (2 * l + 1);
    iX[0] |= (i & (uint64_t)1 << (3 * l + 2)) >> (2 * l + 2);
  }
#if EXAFMM_HILBERT
  int N = 2 << (level - 1);
  int t = iX[2] >> 1;
  for (int d = 2; d > 0; d--) iX[d] ^= iX[d - 1];
  iX[0] ^= t;
  for (int Q = 2; Q != N; Q <<= 1) {
    int R = Q - 1;
    for (int d = 2; d >= 0; d--) {
      if (iX[d] & Q)
        iX[0] ^= R;
      else {
        t = (iX[0] ^ iX[d]) & R;
        iX[0] ^= t;
        iX[d] ^= t;
      }
    }
  }
#endif
  return iX;
}

/**
 * @brief Get 3D index from a Hilbert key without level offset.
 *
 * @param i Hilbert key without level offset.
 * @param level Level.
 * @return ivec3 3D index, an integer triplet.
 */
ivec3 get3DIndex(uint64_t i, int level) {
  ivec3 iX = {0, 0, 0};
  for (int l = 0; l < level; l++) {
    iX[2] |= (i & (uint64_t)1 << 3 * l) >> 2 * l;
    iX[1] |= (i & (uint64_t)1 << (3 * l + 1)) >> (2 * l + 1);
    iX[0] |= (i & (uint64_t)1 << (3 * l + 2)) >> (2 * l + 2);
  }
#if EXAFMM_HILBERT
  int N = 2 << (level - 1);
  int t = iX[2] >> 1;
  for (int d = 2; d > 0; d--) iX[d] ^= iX[d - 1];
  iX[0] ^= t;
  for (int Q = 2; Q != N; Q <<= 1) {
    int R = Q - 1;
    for (int d = 2; d >= 0; d--) {
      if (iX[d] & Q)
        iX[0] ^= R;
      else {
        t = (iX[0] ^ iX[d]) & R;
        iX[0] ^= t;
        iX[d] ^= t;
      }
    }
  }
#endif
  return iX;
}

/**
 * @brief Given bounding box and level, get 3D index from 3D coordinates.
 * @tparam T A potential type.
 * @param X 3D coordinates.
 * @param level Level.
 * @param x0 Coordinates of the center of the bounding box.
 * @param r0 Half of the side length of the bounding box.
 * @return ivec3 3D index, an integer triplet.
 */
template <typename T>
ivec3 get3DIndex(typename potential_traits<T>::coord_t X, int level,
                 typename potential_traits<T>::coord_t x0,
                 typename potential_traits<T>::real_t r0) {
  using coord_t = typename potential_traits<T>::coord_t;
  coord_t Xmin = x0 - r0 * coord_t::Ones();
  typename potential_traits<T>::real_t dx = 2 * r0 / (1 << level);
  ivec3 iX;
  for (int d = 0; d < 3; d++) {
    iX[d] = static_cast<int>(std::floor((X[d] - Xmin[d]) / dx));
  }
  return iX;
}

/**
 * @brief Given bounding box and level, get 3D coordinates from 3D index.
 *
 * @param iX 3D index, an integer triplet.
 * @param level Level.
 * @param x0 Coordinates of the center of the bounding box.
 * @param r0 Half of the side length of the bounding box.
 * @return vec3 3D coordinates.
 */
template <typename PotentialT>
typename potential_traits<PotentialT>::coord_t getCoordinates(
    ivec3 iX, int level, typename potential_traits<PotentialT>::coord_t x0,
    typename potential_traits<PotentialT>::real_t r0) {
  using pt = potential_traits<PotentialT>;
  typename pt::coord_t Xmin = x0 - r0;
  typename pt::real_t dx = 2 * r0 / (1 << level);
  typename pt::coord_t X;
  for (int d = 0; d < 3; d++) {
    X[d] = (iX[d] + 0.5) * dx + Xmin[d];
  }
  return X;
}
}  // namespace ExaFMM
#endif  // INCLUDE_EXAFMM_HILBERT_H_
