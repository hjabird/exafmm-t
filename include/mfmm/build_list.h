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
#ifndef INCLUDE_MFMM_BUILD_LIST_H_
#define INCLUDE_MFMM_BUILD_LIST_H_

#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "fmm.h"
#include "geometry.h"
#include "mfmm.h"
#include "octree_location.h"

namespace mfmm {

/** Generate the mapping from Morton keys to node indices in the tree.
 * @param nodes Tree.
 * @return Keys to indices mapping.
 */
template <typename T>
std::unordered_map<octree_location, size_t> get_key2id(const Nodes<T>& nodes) {
  std::unordered_map<octree_location, size_t> key2id;
  for (size_t i = 0; i < nodes.size(); ++i) {
    key2id[nodes[i].location()] = nodes[i].index();
  }
  return key2id;
}

/** Generate the set of keys of all leaf nodes.
 * @param nodes Tree.
 * @return Set of all leaf keys with level offset.
 */
template <typename T>
std::unordered_set<octree_location> get_leaf_keys(const Nodes<T>& nodes) {
  // we cannot use leafs to generate leaf keys, since it does not include
  // empty leaf nodes where ntrgs and nsrcs are 0.
  std::unordered_set<octree_location> leafKeys;
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (nodes[i].is_leaf()) {
      leafKeys.insert(nodes[i].location());
    }
  }
  return leafKeys;
}

/** Given the 3D index of an octant and its depth, return the key of
 * the leaf that contains the octant. If such leaf does not exist, return the
 * key of the original octant.
 * @param iX Integer index of the octant.
 * @param level The level of the octant.
 * @return Morton index with level offset.
 */
octree_location find_key(const ivec3& iX, int level,
                         const std::unordered_set<octree_location>& leafKeys) {
  octree_location originalKey(iX, level);
  octree_location currentKey = originalKey;
  while (level > 0) {
    if (leafKeys.find(currentKey) != leafKeys.end()) {  // if key is leaf
      return currentKey;
    } else {  // else go 1 level up
      currentKey = currentKey.parent();
      level--;
    }
  }
  return originalKey;
}

/** Build lists for P2P, P2L and M2P operators for a given node.
 * @param node Node.
 * @param nodes Tree.
 * @param leafKeys The set of all leaf keys.
 * @param key2id The mapping from a node's key to its index in the tree.
 */
template <typename FmmT>
void build_other_list(
    Node<typename FmmT::potential_t>* node,
    Nodes<typename FmmT::potential_t>& nodes, const FmmT& fmm,
    const std::unordered_set<octree_location>& leafKeys,
    const std::unordered_map<octree_location, size_t>& key2id) {
  using node_t = Node<typename FmmT::potential_t>;
  std::set<node_t*> p2pSet, m2pSet, p2lSet;
  node_t& currentNode = *node;
  if (currentNode.location() != octree_location(0, 0)) {
    node_t* parent = currentNode.parent();
    ivec3 min3dIdx = {0, 0, 0};
    ivec3 max3dIdx = ivec3::Ones(3) * (1 << node->location().level());
    ivec3 current3dIdx = currentNode.location().get_3D_index();
    ivec3 parent3dIdx = parent->location().get_3D_index();
    // search in every direction
    for (int i = -2; i < 4; i++) {
      for (int j = -2; j < 4; j++) {
        for (int k = -2; k < 4; k++) {
          ivec3 direction{i, j, k};
          direction += parent3dIdx * 2;
          if ((direction.array() >= min3dIdx.array()).all() &&
              (direction.array() < max3dIdx.array()).all() &&
              direction != current3dIdx) {
            octree_location resKey =
                find_key(direction, currentNode.location().level(), leafKeys);
            bool adj = resKey.is_adjacent(currentNode.location());
            node_t& res = nodes[key2id.at(resKey)];
            if (res.location().level() <
                currentNode.location().level()) {  // when res node is a leaf
              if (adj) {
                if (currentNode.is_leaf()) {
                  p2pSet.insert(&res);
                }
              } else {
                if (currentNode.is_leaf() &&
                    currentNode.num_targets() <= fmm.m_numSurf) {
                  p2pSet.insert(&res);
                } else {
                  p2lSet.insert(&res);
                }
              }
            }
            if (res.location().level() ==
                currentNode.location().level()) {  // when res is a colleague
              if (adj) {
                if (currentNode.is_leaf()) {
                  std::queue<node_t*> buffer;
                  buffer.push(&res);
                  while (!buffer.empty()) {
                    node_t& temp = *buffer.front();
                    buffer.pop();
                    if (!temp.location().is_adjacent(currentNode.location())) {
                      if (temp.is_leaf() &&
                          temp.num_sources() <= fmm.m_numSurf) {
                        p2pSet.insert(&temp);
                      } else {
                        m2pSet.insert(&temp);
                      }
                    } else {
                      if (temp.is_leaf()) {
                        p2pSet.insert(&temp);
                      } else {
                        for (int i = 0; i < NCHILD; i++) {
                          if (temp.has_child(i)) {
                            buffer.push(&temp.child(i));
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if (currentNode.is_leaf()) {
    p2pSet.insert(&currentNode);
  }
  for (auto i = p2pSet.begin(); i != p2pSet.end(); i++) {
    if ((*i) != nullptr) {
      currentNode.P2Plist().push_back(*i);
    }
  }
  for (auto i = p2lSet.begin(); i != p2lSet.end(); i++) {
    if ((*i) != nullptr) {
      currentNode.P2Llist().push_back(*i);
    }
  }
  for (auto i = m2pSet.begin(); i != m2pSet.end(); i++) {
    if ((*i) != nullptr) {
      currentNode.M2Plist().push_back(*i);
    }
  }
}

/** Build M2L interaction list for a given node.
 * @param node Node.
 * @param nodes Tree.
 * @param key2id The mapping from a node's key to its index in the tree.
 */
template <typename T>
void build_M2L_list(Node<T>* node, Nodes<T>& nodes,
                    const std::unordered_map<octree_location, size_t>& key2id) {
  using node_t = Node<T>;
  node->M2Llist().resize(REL_COORD_M2L.size(), nullptr);
  node_t& currentNode = *node;
  ivec3 min3dIdx = {0, 0, 0};
  ivec3 max3dIdx = ivec3::Ones(3) * (1 << currentNode.location().level());
  if (!currentNode.is_leaf()) {
    ivec3 current3dIdx = currentNode.location().get_3D_index();
    for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        for (int k = -1; k <= 1; k++) {
          if (i || j || k) {  // exclude current node itself
            ivec3 relativeCoord{i, j, k};
            ivec3 nearby3dIdx = current3dIdx + relativeCoord;
            if ((nearby3dIdx.array() >= min3dIdx.array()).all() &&
                (nearby3dIdx.array() < max3dIdx.array()).all()) {
              octree_location nearbyLoc(nearby3dIdx,
                                        currentNode.location().level());
              if (key2id.find(nearbyLoc) != key2id.end()) {
                node_t& nearbyNode = nodes[key2id.at(nearbyLoc)];
                if (!nearbyNode.is_leaf()) {
                  size_t idx = REL_COORD_M2L.hash(relativeCoord);
                  currentNode.M2Llist()[idx] = &nearbyNode;
                }
              }
            }
          }
        }
      }
    }
  }
}

/** Build lists for all operators for all nodes in the tree.
 * @param nodes Tree.
 * @param fmm The FMM instance.
 */
template <typename FmmT>
void build_list(Nodes<typename FmmT::potential_t>& nodes, const FmmT& fmm) {
  using node_t = Node<typename FmmT::potential_t>;
  std::unordered_map<octree_location, size_t> key2id = get_key2id(nodes);
  std::unordered_set<octree_location> leaf_keys = get_leaf_keys(nodes);
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < static_cast<int>(nodes.size()); i++) {
    node_t* node = &nodes[i];
    build_M2L_list(node, nodes, key2id);
    build_other_list(node, nodes, fmm, leaf_keys, key2id);
  }
}
}  // namespace mfmm
#endif  // INCLUDE_MFMM_BUILD_LIST_H_
