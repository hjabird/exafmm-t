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
#include "morton_key.h"

namespace ExaFMM {

/** Generate the mapping from Morton keys to node indices in the tree.
 * @param nodes Tree.
 * @return Keys to indices mapping.
 */
template <typename T>
std::unordered_map<morton_key, size_t> get_key2id(const Nodes<T>& nodes) {
  std::unordered_map<morton_key, size_t> key2id;
  for (size_t i = 0; i < nodes.size(); ++i) {
    key2id[nodes[i].key()] = nodes[i].index();
  }
  return key2id;
}

/** Generate the set of keys of all leaf nodes.
 * @param nodes Tree.
 * @return Set of all leaf keys with level offset.
 */
template <typename T>
std::unordered_set<morton_key> get_leaf_keys(const Nodes<T>& nodes) {
  // we cannot use leafs to generate leaf keys, since it does not include
  // empty leaf nodes where ntrgs and nsrcs are 0.
  std::unordered_set<morton_key> leafKeys;
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (nodes[i].is_leaf()) {
      leafKeys.insert(nodes[i].key());
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
morton_key find_key(const ivec3& iX, int level,
                    const std::unordered_set<morton_key>& leafKeys) {
  morton_key originalKey(iX, level);
  morton_key currentKey = originalKey;
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
 *
 * @param node Node.
 * @param nodes Tree.
 * @param leaf_keys The set of all leaf keys.
 * @param key2id The mapping from a node's key to its index in the tree.
 */
template <typename FmmT>
void build_other_list(Node<typename FmmT::potential_t>* node,
                      Nodes<typename FmmT::potential_t>& nodes, const FmmT& fmm,
                      const std::unordered_set<morton_key>& leaf_keys,
                      const std::unordered_map<morton_key, size_t>& key2id) {
  using node_t = Node<typename FmmT::potential_t>;
  std::set<node_t*> P2P_set, M2P_set, P2L_set;
  node_t* curr = node;
  if (curr->key() != morton_key(0, 0)) {
    node_t* parent = curr->parent();
    ivec3 min_iX = {0, 0, 0};
    ivec3 max_iX = ivec3::Ones(3) * (1 << node->level());
    ivec3 curr_iX = curr->key().get_3D_index();
    ivec3 parent_iX = parent->key().get_3D_index();
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
            morton_key res_key = find_key(direction, curr->level(), leaf_keys);
            bool adj = res_key.is_adjacent(curr->key());
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
                  std::queue<node_t*> buffer;
                  buffer.push(res);
                  while (!buffer.empty()) {
                    node_t* temp = buffer.front();
                    buffer.pop();
                    if (!temp->key().is_adjacent(curr->key())) {
                      if (temp->is_leaf() &&
                          temp->num_sources() <= fmm.m_numSurf) {
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
                    const std::unordered_map<morton_key, size_t>& key2id) {
  using node_t = Node<T>;
  node->M2Llist().resize(REL_COORD_M2L.size(), nullptr);
  node_t* curr = node;
  ivec3 min_iX = {0, 0, 0};
  ivec3 max_iX = ivec3::Ones(3) * (1 << curr->level());
  if (!node->is_leaf()) {
    ivec3 curr_iX = curr->key().get_3D_index();
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
              morton_key col_key(col_iX, curr->level());
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
  std::unordered_map<morton_key, size_t> key2id = get_key2id(nodes);
  std::unordered_set<morton_key> leaf_keys = get_leaf_keys(nodes);
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < static_cast<int>(nodes.size()); i++) {
    node_t* node = &nodes[i];
    build_M2L_list(node, nodes, key2id);
    build_other_list(node, nodes, fmm, leaf_keys, key2id);
  }
}
}  // namespace ExaFMM
#endif  // INCLUDE_EXAFMM_BUILD_LIST_H_
