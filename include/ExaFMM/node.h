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
#ifndef INCLUDE_EXAFMM_NODE_H_
#define INCLUDE_EXAFMM_NODE_H_

#include <vector>

#include "body.h"
#include "potential_traits.h"

namespace ExaFMM {
/** Structure of nodes.
 *
 * @tparam Value type of sources and targets (real or complex).
 */
template <typename PotentialT>
class Node {
 private:
  using pt = potential_traits<PotentialT>;
  static constexpr int NCHILD{8};

 public:
  using potential_t = PotentialT;
  using real_t = typename pt::real_t;
  using potential_vector_t = typename pt::template potential_vector_t<dynamic>;
  using potential_grad_vector_t =
      typename pt::template potential_grad_vector_t<dynamic>;
  using coord_t = typename pt::coord_t;
  using coord_matrix_t = typename pt::template coord_matrix_t<dynamic>;
  using node_t = Node<potential_t>;
  using nodeptrvec_t = std::vector<node_t*>;

  Node()
      : m_idx{0},
        m_idxM2L{0},
        m_isLeaf{false},
        m_numTargets{0},
        m_numSources{0},
        m_x{0, 0, 0},
        m_r{0},
        m_key{0},
        m_level{0},
        m_octant{0},
        m_parent{nullptr},
        m_sourceCoords{},
        m_targetCoords{},
        m_sourceStrengths{},
        m_targetPotentials{},
        m_targetGradients{},
        m_upEquiv{},
        m_downEquiv{} {};

  void set_child(Node<PotentialT>& child, size_t octant) {
    assert(octant < NCHILD);
    assert(!m_isLeaf);
    assert(m_children.size() == NCHILD);
    assert(m_children[octant] == nullptr);
    m_children[octant] = &child;
    child.m_parent = this;
    child.m_level = m_level + 1;
    child.m_octant = static_cast<int>(octant);
    child.m_r = m_r / 2;
    child.m_x = m_x;
    for (int d = 0; d < 3; d++) {
      int mul = (((octant & 1 << d) >> d) * 2 - 1);
      child.m_x(d) += child.m_r * mul;
    }
  }

  size_t set_index(size_t newIndex) {
    size_t orig = m_idx;
    m_idx = newIndex;
    return orig;
  }

  uint64_t set_key(uint64_t newKey) {
    size_t orig = m_idx;
    m_key = newKey;
    return orig;
  }

  void set_geometry(const coord_t& centre, const real_t& radius) {
    assert(radius > 0);
    m_x = centre;
    m_r = radius;
  }

  void set_num_sources_and_targets(int numSources, int numTargets,
                                   bool isLeaf) {
    assert(numSources >= 0);
    m_isLeaf = isLeaf;
    m_numSources = numSources;
    m_numTargets = numTargets;
    if (isLeaf) {
      // Sources
      m_sourceCoords.conservativeResize(numSources, 3);
      m_sourceCoords.setZero();
      m_sourceStrengths.conservativeResize(numSources, 1);
      m_sourceStrengths.setZero();
      m_sourceInitialIdx.resize(numSources);
      // Targets
      m_targetCoords.conservativeResize(numTargets, 3);
      m_targetCoords.setZero();
      m_targetPotentials.conservativeResize(numTargets, 1);
      m_targetPotentials.setZero();
      m_targetGradients.conservativeResize(numTargets, 3);
      m_targetGradients.setZero();
      m_targetInitialIdx.resize(numTargets);
    } else {
      m_children.resize(NCHILD, nullptr);
      assert(m_sourceCoords.size() == 0);
      assert(m_sourceStrengths.size() == 0);
      assert(m_sourceInitialIdx.size() == 0);
      assert(m_targetCoords.size() == 0);
      assert(m_targetPotentials.size() == 0);
      assert(m_targetGradients.size() == 0);
      assert(m_targetInitialIdx.size() == 0);
    }
  }

  void set_num_surfs(int numSurfs) {
    m_upEquiv.conservativeResize(numSurfs, 1);
    m_upEquiv.setZero();
    m_downEquiv.conservativeResize(numSurfs, 1);
    m_downEquiv.setZero();
  }

  void set_source(size_t index, const Body<PotentialT>& body) {
    m_sourceCoords.row(index) = body.X;
    m_sourceStrengths(index) = body.q;
    m_sourceInitialIdx[index] = body.ibody;
  }

  void set_target(size_t index, const Body<PotentialT>& body) {
    m_targetCoords.row(index) = body.X;
    m_targetInitialIdx[index] = body.ibody;
  }

  void zero_target_values() {
    m_targetPotentials.setZero();
    m_targetGradients.setZero();
  }

  size_t index() const { return m_idx; }
  size_t& indexM2L() { return m_idxM2L; }
  size_t num_sources() { return m_numSources; }
  size_t num_targets() { return m_numTargets; }
  bool is_leaf() const { return m_isLeaf; }
  const int level() const { return m_level; }
  const uint64_t key() const { return m_key; }
  coord_t centre() const { return m_x; }
  Node* parent() const { return m_parent; }
  real_t radius() const { return m_r; }
  const coord_matrix_t& source_coords() { return m_sourceCoords; }
  const potential_vector_t& source_strengths() { return m_sourceStrengths; }
  const coord_matrix_t& target_coords() { return m_targetCoords; }
  potential_vector_t& target_potentials() { return m_targetPotentials; }
  potential_grad_vector_t& target_gradients() { return m_targetGradients; }
  nodeptrvec_t& P2Llist() { return m_P2Llist; }
  nodeptrvec_t& M2Plist() { return m_M2Plist; }
  nodeptrvec_t& P2Plist() { return m_P2Plist; }
  nodeptrvec_t& M2Llist() { return m_M2Llist; }
  bool has_child(int octant) {
    assert(octant < NCHILD);
    return !m_isLeaf && (m_children[octant] == nullptr);
  }
  Node& child(int octant) {
    assert(octant < m_children.size());
    return *m_children[octant];
  }
  potential_vector_t& up_equiv() { return m_upEquiv; }
  potential_vector_t& down_equiv() { return m_downEquiv; }

 protected:
  size_t m_idx;             //!< Index in the octree
  size_t m_idxM2L;          //!< Index in global M2L interaction list
  bool m_isLeaf;            //!< Whether the node is leaf
  int m_numTargets;         //!< Number of targets
  int m_numSources;         //!< Number of sources
  coord_t m_x;              //!< Coordinates of the center of the node
  real_t m_r;               //!< Radius of the node
  uint64_t m_key;           //!< Morton key
  int m_level;              //!< Level in the octree
  int m_octant;             //!< Octant
  Node* m_parent;           //!< Pointer to parent
  nodeptrvec_t m_children;  //!< Vector of pointers to child nodes
  nodeptrvec_t
      m_P2Llist;  //!< Vector of pointers to nodes in P2L interaction list
  nodeptrvec_t
      m_M2Plist;  //!< Vector of pointers to nodes in M2P interaction list
  nodeptrvec_t
      m_P2Plist;  //!< Vector of pointers to nodes in P2P interaction list
  nodeptrvec_t
      m_M2Llist;  //!< Vector of pointers to nodes in M2L interaction list
  std::vector<int> m_sourceInitialIdx;  //!< Vector of initial source numbering
  std::vector<int> m_targetInitialIdx;  //!< Vector of initial target numbering
  coord_matrix_t
      m_sourceCoords;  //!< Vector of coordinates of sources in the node
  coord_matrix_t
      m_targetCoords;  //!< Vector of coordinates of targets in the node
  potential_vector_t
      m_sourceStrengths;  //!< Vector of charges of sources in the node
  potential_vector_t
      m_targetPotentials;  //!< Vector of potentials targets in the node
  potential_grad_vector_t
      m_targetGradients;  //!< Vector of potentials targets in the node
  potential_vector_t
      m_upEquiv;  //!< Upward check potentials / Upward equivalent densities
  potential_vector_t m_downEquiv;  //!< Downward check potentials / Downward
                                   //!< equivalent densites
};

template <typename T>
using Nodes = std::vector<Node<T>>;  //!< Vector of nodes

template <typename T>
using NodePtrs = std::vector<Node<T>*>;  //!< Vector of Node pointers
}  // namespace ExaFMM

#endif  // INCLUDE_EXAFMM_NODE_H_
