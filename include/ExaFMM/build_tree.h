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
  typename potential_traits<PotentialT>::coord_t xMin(sources[0].X);
  typename potential_traits<PotentialT>::coord_t xMax(sources[0].X);
  for (size_t b = 0; b < sources.size(); ++b) {
    xMin = sources[b].X.cwiseMin(xMin);
    xMax = sources[b].X.cwiseMax(xMax);
  }
  for (size_t b = 0; b < targets.size(); ++b) {
    xMin = targets[b].X.cwiseMin(xMin);
    xMax = targets[b].X.cwiseMax(xMax);
  }
  typename potential_traits<PotentialT>::coord_t x0 = (xMax + xMin) / 2;
  typename potential_traits<PotentialT>::real_t r0 =
      (x0 - xMin).cwiseAbs().maxCoeff();
  r0 *= 1.00001;
  return std::make_tuple(x0, r0);
}

template <class FmmT>
class adaptive_tree {
 public:
  using potential_t = typename FmmT::potential_t;
  using node_t = Node<potential_t>;
  using nodevec_t = std::vector<node_t>;
  using nodeptrvec_t = std::vector<node_t*>;
  using body_t = Body<potential_t>;
  using bodies_t = std::vector<body_t>;

 private:
  nodevec_t m_nodes;
  nodeptrvec_t m_leafs;
  nodeptrvec_t m_nonleafs;

 public:
  adaptive_tree() = delete;

  adaptive_tree(bodies_t& sources, bodies_t& targets, FmmT& fmm) {
    bodies_t sources_buffer = sources;
    bodies_t targets_buffer = targets;
    m_nodes = nodevec_t(1);
    m_nodes[0].set_geometry(fmm.x0, fmm.r0);
    m_nodes.reserve((sources.size() + targets.size()) * (32 / fmm.ncrit + 1));
    build_tree(&sources[0], &sources_buffer[0], 0, sources.size(), &targets[0],
               &targets_buffer[0], 0, targets.size(), &m_nodes[0], fmm);
    int depth = -1;
    for (const auto& leaf : m_leafs) {
      depth = std::max(leaf->level(), depth);
    }
    fmm.depth = depth;
  }

  nodevec_t& nodes() noexcept { return m_nodes; }

  nodeptrvec_t& leaves() noexcept { return m_leafs; }

  nodeptrvec_t& nonleaves() noexcept { return m_nonleafs; }

  //! Build nodes of tree adaptively using a top-down approach based on
  //! recursion
  void build_tree(body_t* sources, body_t* sourcesBuffer, size_t sourcesBegin,
                  size_t sourcesEnd, body_t* targets, body_t* targetsBuffer,
                  size_t targetsBegin, size_t targetsEnd, node_t* node,
                  FmmT& fmm, bool direction = false) {
    //! Create a tree node
    node->set_index(node - &m_nodes[0]);  // current node's index in nodes
    const size_t numSources = sourcesEnd - sourcesBegin;
    const size_t numTargets = targetsEnd - targetsBegin;
    const bool isLeaf{numSources <= fmm.ncrit && numTargets <= fmm.ncrit};
    node->set_num_sources_and_targets(numSources, numTargets, isLeaf);
    node->set_num_surfs(fmm.nsurf);
    const ivec3 iX =
        get3DIndex<potential_t>(node->centre(), node->level(), fmm.x0, fmm.r0);
    node->set_key(getKey(iX, node->level()));

    if (isLeaf) {
      if (numSources || numTargets) {  // do not add to leafs if a node is empty
        m_leafs.push_back(node);
      }
      if (direction) {
        for (size_t i{sourcesBegin}; i < sourcesEnd; i++) {
          sourcesBuffer[i] = sources[i];
        }
        for (size_t i{targetsBegin}; i < targetsEnd; i++) {
          targetsBuffer[i] = targets[i];
        }
      }
      // Copy sources and targets' coords and values to leaf
      body_t* first_source =
          (direction ? sourcesBuffer : sources) + sourcesBegin;
      body_t* first_target =
          (direction ? targetsBuffer : targets) + targetsBegin;
      for (size_t sourceIdx = 0; sourceIdx < numSources; ++sourceIdx) {
        node->set_source(sourceIdx, first_source[sourceIdx]);
      }
      for (int targetIdx = 0; targetIdx < node->num_targets(); ++targetIdx) {
        node->set_target(targetIdx, first_target[targetIdx]);
      }
    } else {  // !isLeaf
      m_nonleafs.push_back(node);
      // Sort bodies and save in buffer
      std::vector<int> source_size, source_offsets;
      std::vector<int> target_size, target_offsets;
      sort_bodies(node, sources, sourcesBuffer, sourcesBegin, sourcesEnd,
                  source_size, source_offsets);  // sourcesBuffer is sorted
      sort_bodies(node, targets, targetsBuffer, targetsBegin, targetsEnd,
                  target_size, target_offsets);  // targetsBuffer is sorted
      //! Loop over children and recurse
      assert(m_nodes.capacity() >= m_nodes.size() + NCHILD);
      m_nodes.resize(m_nodes.size() + NCHILD);
      node_t* child = &m_nodes.back() - NCHILD + 1;
      for (int octant = 0; octant < NCHILD; octant++) {
        node->set_child(child[octant], octant);
        build_tree(sourcesBuffer, sources, source_offsets[octant],
                   source_offsets[octant] + source_size[octant], targetsBuffer,
                   targets, target_offsets[octant],
                   target_offsets[octant] + target_size[octant], &child[octant],
                   fmm, !direction);
      }
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
                   body_t* const buffer, size_t begin, size_t end,
                   std::vector<int>& size, std::vector<int>& offsets) {
    using vec3 = typename potential_traits<potential_t>::coord_t;
    // Count number of bodies in each octant
    size.resize(NCHILD, 0);
    vec3 X = node->centre();
    for (size_t i = begin; i < end; i++) {
      vec3& x = bodies[i].X;
      int octant = (x[0] > X[0]) + ((x[1] > X[1]) << 1) + ((x[2] > X[2]) << 2);
      size[octant]++;
    }
    // Exclusive scan to get offsets
    offsets.resize(8);
    size_t offset = begin;
    for (int i = 0; i < 8; i++) {
      offsets[i] = static_cast<int>(offset);
      offset += size[i];
    }
    // Sort bodies by octant
    std::vector<int> counter(offsets);
    for (size_t i = begin; i < end; i++) {
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
