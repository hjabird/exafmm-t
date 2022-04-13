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
#ifndef INCLUDE_EXAFMM_MORTON_H_
#define INCLUDE_EXAFMM_MORTON_H_
#include <stdint.h>

#include "potential_traits.h"
#include "relative_coords.h"

namespace ExaFMM {

class morton_key {
 public:
  morton_key() = default;
  morton_key(uint64_t x) : m_value(x){};

  uint64_t& operator()() { return m_value; }
  const uint64_t& operator()() const { return m_value; }

  bool operator==(const morton_key& other) const noexcept {
    return m_value == other.m_value;
  }
  bool operator!=(const morton_key& other) const noexcept {
    return m_value != other.m_value;
  }
  bool operator>(const morton_key& other) const noexcept {
    return m_value > other.m_value;
  }
  bool operator<(const morton_key& other) const noexcept {
    return m_value < other.m_value;
  }

  /** Get level of this key.
   * @return int Level.
   */
  inline int level() const {
    int level = -1;
    uint64_t offset = 0;
    while (m_value >= offset) {
      level++;
      offset += (uint64_t)1 << 3 * level;
    }
    return level;
  }

  /** Get parent's Morton key with level offset.
   * @return morton_key Parent's Morton key.
   */
  inline morton_key parent() const {
    int thisLevel = level();
    return (m_value - level_offset(thisLevel)()) / 8 +
           level_offset(thisLevel - 1)();
  }

  /** Calculate levelwise offset of Morton key.
   * @param level Level.
   * @return uint64_t Level offset.
   */
  static inline morton_key level_offset(int level) {
    return (((uint64_t)1 << 3 * level) - 1) / 7;
  }

  /** Get first child's Morton key with level offset.
   * @return morton_key First child's Morton key.
   */
  morton_key child() const {
    int thisLevel = level();
    return (m_value - level_offset(thisLevel)()) * 8 +
           level_offset(thisLevel + 1)();
  }

  /** Determine which octant the key belongs to.
   * @param offset Whether the key contains level offset, default to true.
   * @return int Octant.
   */
  int octant(bool offset = true) const {
    int thisLevel = level();
    morton_key retVal{m_value};
    if (offset) {
      retVal() -= level_offset(thisLevel)();
    }
    return retVal() & 7;
  }

  /** Get 3D index from a Morton key with level offset.
   * @return ivec3 3D index, an integer triplet.
   */
  ivec3 get_3D_index() const {
    int thisLevel = level();
    morton_key tmp{m_value};
    tmp() -= level_offset(thisLevel)();
    return tmp.get_3D_index(thisLevel);
  }

  /** Get 3D index from a Morton key without level offset.
   * @param level Level.
   * @return ivec3 3D index, an integer triplet.
   */
  ivec3 get_3D_index(int level) const {
    ivec3 iX = {0, 0, 0};
    morton_key tmp{m_value};
    for (int l = 0; l < level; l++) {
      iX[2] |= (tmp() & (uint64_t)1 << 3 * l) >> 2 * l;
      iX[1] |= (tmp() & (uint64_t)1 << (3 * l + 1)) >> (2 * l + 1);
      iX[0] |= (tmp() & (uint64_t)1 << (3 * l + 2)) >> (2 * l + 2);
    }
    return iX;
  }

 protected:
  uint64_t m_value;
};

/** Get Morton key from 3D index of a node.
 * @param iX 3D index of a node, an integer triplet.
 * @param level Level of the node.
 * @param offset Whether to add level offset to the key, default to true.
 * @return morton_key Morton key.
 */
morton_key getKey(ivec3 iX, int level, bool offset = true) {
  morton_key i = 0;
  for (int l = 0; l < level; l++) {
    i() |= (iX[2] & (uint64_t)1 << l) << 2 * l;
    i() |= (iX[1] & (uint64_t)1 << l) << (2 * l + 1);
    i() |= (iX[0] & (uint64_t)1 << l) << (2 * l + 2);
  }
  if (offset) {
    i() += i.level_offset(level)();
  }
  return i;
}

/** Given bounding box and level, get 3D index from 3D coordinates.
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
  ivec3 iX = ((X - Xmin) / dx).array().floor().cast<int>();
  return iX;
}

/** Given bounding box and level, get 3D coordinates from 3D index.
 * @tparam T A potential type.
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
  typename pt::coord_t X = (iX + coord_t::Ones() * 0.5) * dx + Xmin;
  return X;
}
}  // namespace ExaFMM

namespace std {
template <>
struct hash<ExaFMM::morton_key> {
  std::size_t operator()(const ExaFMM::morton_key& k) const {
    return std::hash<uint64_t>()(k());
  }
};
}  // namespace std

#endif  // INCLUDE_EXAFMM_MORTON_H_
