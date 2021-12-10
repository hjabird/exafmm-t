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
#ifndef INCLUDE_EXAFMM_BUILDTREE_H_
#define INCLUDE_EXAFMM_BUILDTREE_H_
#include <cassert>
#include <queue>
#include <tuple>
#include <unordered_map>

#include "exafmm.h"
#include "fmm.h"
#include "hilbert.h"

namespace ExaFMM {
using std::max;

/** Get bounding box of sources and targets.
 *
 * @tparam T Target's value type (real or complex).
 * @param sources Vector of sources.
 * @param targets Vector of targets.
 * @return A tuple of the box centre and the radius of the bounding box.
 */
template <typename PotentialT>
auto get_bounds(const std::vector<Body<PotentialT>>& sources,
                const std::vector<Body<PotentialT>>& targets) {
  potential_traits<PotentialT>::coord_t xMin = sources[0].X;
  potential_traits<PotentialT>::coord_t xMax = sources[0].X;
  for (size_t b = 0; b < sources.size(); ++b) {
    xMin = min(sources[b].X, xMin);
    xMax = max(sources[b].X, xMax);
  }
  for (size_t b = 0; b < targets.size(); ++b) {
    xMin = min(targets[b].X, xMin);
    xMax = max(targets[b].X, xMax);
  }
  potential_traits<PotentialT>::coord_t x0 = (xMax + xMin) / 2;
  potential_traits<PotentialT>::real_t r0 =
      std::max(max(x0 - xMin), max(xMax - x0));
  r0 *= 1.00001;
  return std::make_tuple{x0, r0};
}

template <class FmmT>
class adaptive_tree {
 public:
  using potential_t = typename FmmT::potential_t;
  using node_t = typename Node<potential_t>;
  using nodevec_t = typename std::vector<node_t>;
  using nodeptrvec_t = typename std::vector<node_t*>;
  using body_t = Body<potential_t>;
  using bodies_t = std::vector<body_t>;

  nodevec_t m_nodes;
  nodeptrvec_t m_leafs;
  nodeptrvec_t m_nonleafs;

  adaptive_tree() = delete;

  adaptive_tree(const bodies_t& sources, const bodies_t& targets,
                const FmmT& fmm) {
    bodies_t sources_buffer = sources;
    bodies_t targets_buffer = targets;
    m_nodes = nodevec_t(1);
    nodes[0].parent = nullptr;
    nodes[0].octant = 0;
    nodes[0].x = fmm.x0;
    nodes[0].r = fmm.r0;
    nodes[0].level = 0;
    nodes.reserve((sources.size() + targets.size()) * (32 / fmm.ncrit + 1));
    build_tree(&sources[0], &sources_buffer[0], 0, sources.size(), &targets[0],
               &targets_buffer[0], 0, targets.size(), &m_nodes[0], fmm);
    int depth = -1;
    for (const auto& leaf : m_leafs) {
      depth = std::max(leaf->level, depth);
    }
    fmm.depth = depth;
    return nodes;
  }

  //! Build nodes of tree adaptively using a top-down approach based on
  //! recursion
  void build_tree(body_t* sources, body_t* sources_buffer, int source_begin,
                  int source_end, body_t* targets, body_t* targets_buffer,
                  int target_begin, int target_end, body_t* node, FmmT& fmm,
                  bool direction = false) {
    //! Create a tree node
    node->idx = int(node - &m_nodes[0]);  // current node's index in nodes
    node->nsr cs = source_end - source_begin;
    node->ntrgs = target_end - target_begin;
    node->up_equiv.resize(fmm.nsurf, (T)(0.));
    node->dn_equiv.resize(fmm.nsurf, (T)(0.));
    ivec3 iX = get3DIndex(node->x, node->level, fmm.x0, fmm.r0);
    node->key = getKey(iX, node->level);

    //! If node is a leaf
    if (node->nsrcs <= fmm.ncrit && node->ntrgs <= fmm.ncrit) {
      node->is_leaf = true;
      node->trg_value.resize(
          node->ntrgs * 4,
          static_cast<potential_t>(0.));  // initialize target result vector
      if (node->nsrcs ||
          node->ntrgs) {  // do not add to leafs if a node is empty
        m_leafs.push_back(node);
      }
      if (direction) {
        for (int i = source_begin; i < source_end; i++) {
          sources_buffer[i].X = sources[i].X;
          sources_buffer[i].q = sources[i].q;
          sources_buffer[i].ibody = sources[i].ibody;
        }
        for (int i = target_begin; i < target_end; i++) {
          targets_buffer[i].X = targets[i].X;
          targets_buffer[i].ibody = targets[i].ibody;
        }
      }
      // Copy sources and targets' coords and values to leaf
      Body<T>* first_source =
          (direction ? sources_buffer : sources) + source_begin;
      Body<T>* first_target =
          (direction ? targets_buffer : targets) + target_begin;
      for (Body<T>* B = first_source; B < first_source + node->nsrcs; ++B) {
        for (int d = 0; d < 3; ++d) {
          node->src_coord.push_back(B->X[d]);
        }
        node->isrcs.push_back(B->ibody);
        node->src_value.push_back(B->q);
      }
      for (Body<T>* B = first_target; B < first_target + node->ntrgs; ++B) {
        for (int d = 0; d < 3; ++d) {
          node->trg_coord.push_back(B->X[d]);
        }
        node->itrgs.push_back(B->ibody);
      }
      return;
    }
    // Sort bodies and save in buffer
    std::vector<int> source_size, source_offsets;
    std::vector<int> target_size, target_offsets;
    sort_bodies(node, sources, sources_buffer, source_begin, source_end,
                source_size, source_offsets);  // sources_buffer is sorted
    sort_bodies(node, targets, targets_buffer, target_begin, target_end,
                target_size, target_offsets);  // targets_buffer is sorted
    //! Loop over children and recurse
    node->is_leaf = false;
    m_nonleafs.push_back(node);
    assert(nodes.capacity() >= nodes.size() + NCHILD);
    m_nodes.resize(m_nodes.size() + NCHILD);
    Node<T>* child = &m_nodes.back() - NCHILD + 1;
    node->children.resize(8, nullptr);
    for (int c = 0; c < 8; c++) {
      node->children[c] = &child[c];
      child[c].x = node->x;
      child[c].r = node->r / 2;
      for (int d = 0; d < 3; d++) {
        child[c].x[d] += child[c].r * (((c & 1 << d) >> d) * 2 - 1);
      }
      child[c].parent = node;
      child[c].octant = c;
      child[c].level = node->level + 1;
      build_tree(sources_buffer, sources, source_offsets[c],
                 source_offsets[c] + source_size[c], targets_buffer, targets,
                 target_offsets[c], target_offsets[c] + target_size[c],
                 &child[c], fmm, !direction);
    }
  }

  /** Sort a chunk of bodies in a node according to their octants
   *
   * @tparam T Target's value type (real or complex)
   * @param node The node that bodies are in
   * @param bodies The bodies to be sorted
   * @param buffer The sorted bodies
   * @param begin Begin index of the chunk
   * @param end End index of the chunk
   * @param size Vector of the counts of bodies in each octant after
   * @param offsets Vector of the offsets of sorted bodies in each octant
   */
  void sort_bodies(node_t* const node, body_t* const bodies,
                   body_t* const buffer, int begin, int end,
                   std::vector<int>& size, std::vector<int>& offsets) {
    using vec3 = potential_traits<PotentialT>::coord_t;
    // Count number of bodies in each octant
    size.resize(8, 0);
    vec3 X = node->x;  // the center of the node
    for (int i = begin; i < end; i++) {
      vec3& x = bodies[i].X;
      int octant = (x[0] > X[0]) + ((x[1] > X[1]) << 1) + ((x[2] > X[2]) << 2);
      size[octant]++;
    }
    // Exclusive scan to get offsets
    offsets.resize(8);
    int offset = begin;
    for (int i = 0; i < 8; i++) {
      offsets[i] = offset;
      offset += size[i];
    }
    // Sort bodies by octant
    std::vector<int> counter(offsets);
    for (int i = begin; i < end; i++) {
      vec3& x = bodies[i].X;
      int octant = (x[0] > X[0]) + ((x[1] > X[1]) << 1) + ((x[2] > X[2]) << 2);
      buffer[counter[octant]].X = bodies[i].X;
      buffer[counter[octant]].q = bodies[i].q;
      buffer[counter[octant]].ibody = bodies[i].ibody;
      counter[octant]++;
    }
  }
};

}  // namespace ExaFMM
#endif  // INCLUDE_EXAFMM_BUILDTREE_H_
