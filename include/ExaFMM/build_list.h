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
#ifndef INCLUDE_EXAFMM_BUILD_LIST_H_
#define INCLUDE_EXAFMM_BUILD_LIST_H_

#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "exafmm.h"
#include "fmm.h"
#include "geometry.h"
#include "hilbert.h"

namespace ExaFMM {

using std::abs;
using std::max;
using std::queue;
using std::set;
using std::unordered_map;
using std::unordered_set;

/**
 * @brief Generate the mapping from Hilbert keys to node indices in the tree.
 *
 * @param nodes Tree.
 * @return Keys to indices mapping.
 */
template <typename T>
unordered_map<uint64_t, size_t> get_key2id(const Nodes<T>& nodes) {
  unordered_map<uint64_t, size_t> key2id;
  for (size_t i = 0; i < nodes.size(); ++i) {
    key2id[nodes[i].key()] = nodes[i].index();
  }
  return key2id;
}

/**
 * @brief Generate the set of keys of all leaf nodes.
 *
 * @param nodes Tree.
 * @return Set of all leaf keys with level offset.
 */
template <typename T>
unordered_set<uint64_t> get_leaf_keys(const Nodes<T>& nodes) {
  // we cannot use leafs to generate leaf keys, since it does not include
  // empty leaf nodes where ntrgs and nsrcs are 0.
  unordered_set<uint64_t> leaf_keys;
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (nodes[i].is_leaf()) {
      leaf_keys.insert(nodes[i].key());
    }
  }
  return leaf_keys;
}

/**
 * @brief Given the 3D index of an octant and its depth, return the key of
 * the leaf that contains the octant. If such leaf does not exist, return the
 * key of the original octant.
 *
 * @param iX Integer index of the octant.
 * @param level The level of the octant.
 *
 * @return Hilbert index with level offset.
 */
uint64_t find_key(const ivec3& iX, int level,
                  const unordered_set<uint64_t>& leaf_keys) {
  uint64_t orig_key = getKey(iX, level, true);
  uint64_t curr_key = orig_key;
  while (level > 0) {
    if (leaf_keys.find(curr_key) != leaf_keys.end()) {  // if key is leaf
      return curr_key;
    } else {  // else go 1 level up
      curr_key = getParent(curr_key);
      level--;
    }
  }
  return orig_key;
}

/**
 * @brief Check the adjacency of two nodes.
 *
 * @param key_a, key_b Hilbert keys with level offset.
 */
bool is_adjacent(uint64_t key_a, uint64_t key_b) {
  int level_a = getLevel(key_a);
  int level_b = getLevel(key_b);
  int max_level = max(level_a, level_b);
  ivec3 iX_a = get3DIndex(key_a);
  ivec3 iX_b = get3DIndex(key_b);
  ivec3 iX_ac = (iX_a * 2 + ivec3{1, 1, 1}) *
                (1 << (max_level - level_a));  // center coordinates
  ivec3 iX_bc = (iX_b * 2 + ivec3{1, 1, 1}) *
                (1 << (max_level - level_b));  // center coordinates
  ivec3 diff = iX_ac - iX_bc;
  int max_diff = -1;  // L-infinity norm of diff
  for (int d = 0; d < 3; ++d) {
    diff[d] = abs(diff[d]);
    max_diff = max(max_diff, diff[d]);
  }
  int sum_radius = (1 << (max_level - level_a)) + (1 << (max_level - level_b));

  return (diff[0] <= sum_radius) && (diff[1] <= sum_radius) &&
         (diff[2] <= sum_radius) && (max_diff == sum_radius);
}

/**
 * @brief Build lists for P2P, P2L and M2P operators for a given node.
 *
 * @param node Node.
 * @param nodes Tree.
 * @param leaf_keys The set of all leaf keys.
 * @param key2id The mapping from a node's key to its index in the tree.
 */
template <typename FmmT>
void build_other_list(Node<typename FmmT::potential_t>* node,
                      Nodes<typename FmmT::potential_t>& nodes, const FmmT& fmm,
                      const unordered_set<uint64_t>& leaf_keys,
                      const unordered_map<uint64_t, size_t>& key2id) {
  using node_t = Node<typename FmmT::potential_t>;
  set<node_t*> P2P_set, M2P_set, P2L_set;
  node_t* curr = node;
  if (curr->key() != 0) {
    node_t* parent = curr->parent();
    ivec3 min_iX = {0, 0, 0};
    ivec3 max_iX = ivec3::Ones(3) * (1 << node->level());
    ivec3 curr_iX = get3DIndex(curr->key());
    ivec3 parent_iX = get3DIndex(parent->key());
    // search in every direction
    for (int i = -2; i < 4; i++) {
      for (int j = -2; j < 4; j++) {
        for (int k = -2; k < 4; k++) {
          ivec3 direction;
          direction[0] = i;
          direction[1] = j;
          direction[2] = k;
          direction += parent_iX * 2;
          if ((direction.array() >= min_iX.array()).all() &&
              (direction.array() < max_iX.array()).all() &&
              direction != curr_iX) {
            uint64_t res_key = find_key(direction, curr->level(), leaf_keys);
            bool adj = is_adjacent(res_key, curr->key());
            node_t* res = &nodes[key2id.at(res_key)];
            if (res->level() < curr->level()) {  // when res node is a leaf
              if (adj) {
                if (curr->is_leaf()) {
                  P2P_set.insert(res);
                }
              } else {
                if (curr->is_leaf() && curr->num_targets() <= fmm.m_numSurf) {
                  P2P_set.insert(res);
                } else {
                  P2L_set.insert(res);
                }
              }
            }
            if (res->level() == curr->level()) {  // when res is a colleague
              if (adj) {
                if (curr->is_leaf()) {
                  queue<node_t*> buffer;
                  buffer.push(res);
                  while (!buffer.empty()) {
                    node_t* temp = buffer.front();
                    buffer.pop();
                    if (!is_adjacent(temp->key(), curr->key())) {
                      if (temp->is_leaf() && temp->num_sources() <= fmm.m_numSurf) {
                        P2P_set.insert(temp);
                      } else {
                        M2P_set.insert(temp);
                      }
                    } else {
                      if (temp->is_leaf()) {
                        P2P_set.insert(temp);
                      } else {
                        for (int i = 0; i < NCHILD; i++) {
                          if (temp->has_child(i)) {
                            buffer.push(&temp->child(i));
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
  if (curr->is_leaf()) {
    P2P_set.insert(curr);
  }
  for (auto i = P2P_set.begin(); i != P2P_set.end(); i++) {
    if ((*i) != nullptr) {
      curr->P2Plist().push_back(*i);
    }
  }
  for (auto i = P2L_set.begin(); i != P2L_set.end(); i++) {
    if ((*i) != nullptr) {
      curr->P2Llist().push_back(*i);
    }
  }
  for (auto i = M2P_set.begin(); i != M2P_set.end(); i++) {
    if ((*i) != nullptr) {
      curr->M2Plist().push_back(*i);
    }
  }
}

/**
 * @brief Build M2L interaction list for a given node.
 *
 * @param node Node.
 * @param nodes Tree.
 * @param key2id The mapping from a node's key to its index in the tree.
 */
template <typename T>
void build_M2L_list(Node<T>* node, Nodes<T>& nodes,
                    const unordered_map<uint64_t, size_t>& key2id) {
  using node_t = Node<T>;
  node->M2Llist().resize(REL_COORD_M2L.size(), nullptr);
  node_t* curr = node;
  ivec3 min_iX = {0, 0, 0};
  ivec3 max_iX = ivec3::Ones(3) * (1 << curr->level());
  if (!node->is_leaf()) {
    ivec3 curr_iX = get3DIndex(curr->key());
    ivec3 col_iX;
    ivec3 rel_coord;
    for (int i = -1; i <= 1; i++) {
      rel_coord[0] = i;
      for (int j = -1; j <= 1; j++) {
        rel_coord[1] = j;
        for (int k = -1; k <= 1; k++) {
          rel_coord[2] = k;
          if (i || j || k) {  // exclude current node itself
            col_iX = curr_iX + rel_coord;
            if ((col_iX.array() >= min_iX.array()).all() &&
                (col_iX.array() < max_iX.array()).all()) {
              uint64_t col_key = getKey(col_iX, curr->level(), true);
              if (key2id.find(col_key) != key2id.end()) {
                node_t* col = &nodes[key2id.at(col_key)];
                if (!col->is_leaf()) {
                  int idx = REL_COORD_M2L.hash(rel_coord);
                  curr->M2Llist()[idx] = col;
                }
              }
            }
          }
        }
      }
    }
  }
}

/**
 * @brief Build lists for all operators for all nodes in the tree.
 *
 * @param nodes Tree.
 * @param fmm The FMM instance.
 */
template <typename FmmT>
void build_list(Nodes<typename FmmT::potential_t>& nodes, const FmmT& fmm) {
  using node_t = Node<typename FmmT::potential_t>;
  unordered_map<uint64_t, size_t> key2id = get_key2id(nodes);
  unordered_set<uint64_t> leaf_keys = get_leaf_keys(nodes);
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < static_cast<int>(nodes.size()); i++) {
    node_t* node = &nodes[i];
    build_M2L_list(node, nodes, key2id);
    build_other_list(node, nodes, fmm, leaf_keys, key2id);
  }
}
}  // namespace ExaFMM
#endif  // INCLUDE_EXAFMM_BUILD_LIST_H_
