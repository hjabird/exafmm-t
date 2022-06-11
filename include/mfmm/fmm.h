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
#ifndef INCLUDE_MFMM_FMM_H_
#define INCLUDE_MFMM_FMM_H_

#include <fftw3.h>

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <algorithm>  // std::fill
#include <fstream>
#include <numeric>

#include "geometry.h"
#include "mfmm.h"
#include "p2p_methods.h"
#include "timer.h"

namespace mfmm {

//! Base FMM class
template <class FmmKernel>
class Fmm : public p2p_methods<FmmKernel> {
 public:
  using potential_t = typename FmmKernel::potential_t;

 protected:
  using pt = potential_traits<potential_t>;

 public:
  using real_t = typename pt::real_t;
  using complex_t = typename pt::complex_t;
  using fmm_kernel_funcs_arg_t = typename FmmKernel::kernel_args_t;

  template <int Rows = dynamic, int Cols = dynamic, int RowOrder = row_major>
  using potential_matrix_t =
      typename pt::template potential_matrix_t<Rows, Cols, RowOrder>;
  template <int Rows = dynamic>
  using potential_vector_t = typename pt::template potential_vector_t<Rows>;
  template <int Rows = dynamic, int Cols = dynamic, int RowOrder = row_major>
  using real_matrix_t =
      typename pt::template real_matrix_t<Rows, Cols, RowOrder>;
  template <int Rows = dynamic>
  using real_vector_t = typename pt::template real_vector_t<Rows>;
  template <int Rows = dynamic, int Cols = dynamic, int RowOrder = row_major>
  using complex_matrix_t =
      typename pt::template complex_matrix_t<Rows, Cols, RowOrder>;
  template <int Rows = dynamic>
  using complex_vector_t = typename pt::template complex_vector_t<Rows>;
  using coord_t = typename pt::coord_t;
  template <int Rows = dynamic, int RowOrder = row_major>
  using coord_matrix_t = typename pt::template coord_matrix_t<Rows, RowOrder>;
  using node_t = Node<potential_t>;
  using nodevec_t = std::vector<node_t>;
  using nodeptrvec_t = std::vector<node_t*>;

  int m_p;              //!< Order of expansion
  int m_numSurf;        //!< Number of points on equivalent / check surface
  int m_numConvPoints;  //!< Number of points on convolution grid
  int m_numFreq;  //!< Number of coefficients in DFT (depending on whether T is
                  //!< real_t)
  int m_numCrit;  //!< Max number of bodies per leaf
  int m_depth;    //!< Depth of the tree
  real_t m_r0;    //!< Half of the side length of the bounding box
  coord_t m_x0;   //!< Coordinates of the center of root box

  Fmm() = delete;

  Fmm(int p, int nCrit,
      fmm_kernel_funcs_arg_t kernelArguments = fmm_kernel_funcs_arg_t{})
      : p2p_methods<FmmKernel>{kernelArguments},
        m_p{p},
        m_numCrit{nCrit},
        m_numSurf{6 * (p - 1) * (p - 1) + 2},
        m_numConvPoints{8 * p * p * p},
        m_numFreq{0} {
    m_numFreq = potential_traits<potential_t>::isComplexPotential
                    ? m_numConvPoints
                    : 4 * p * p * (p + 1);
  }

  ~Fmm() = default;

 protected:
  // Matrices for upwards check surface (& potentials) to upwards equivalent
  // surface (& densities).
  std::vector<potential_matrix_t<dynamic, dynamic>> m_matUC2E;
  // Matrices for downwards check surface (& potentials) to downards equivalent
  // surface (& densities).
  std::vector<potential_matrix_t<dynamic, dynamic>> m_matDC2E;

  std::vector<
      std::array<potential_matrix_t<dynamic, dynamic>, REL_COORD_M2M.size()>>
      m_matM2M;
  std::vector<
      std::array<potential_matrix_t<dynamic, dynamic>, REL_COORD_L2L.size()>>
      m_matL2L;
  std::vector<std::array<std::vector<complex_t>, REL_COORD_M2L.size()>>
      m_matM2L;

  // Data required for moment to local interaction. m_m2lData[octreeLevel] has
  // M2L data including offsets and interaction counts at the required level in
  // the octree.
  std::vector<M2LData<real_t>> m_m2lData;

 public:
  /** Compute the kernel matrix of a given kernel.
   *
   * The kernel matrix defines the interaction between the sources and the
   * targets: targetVal = kernelMatrix * sourceStrength.
   * This function evaluates the interaction kernel using unit source strength
   * to obtain each value in the matrix.
   *
   * @param sourceCoords Vector of source coordinates.
   * @param targetCoords Vector of target coordinates.
   * @return matrix Kernel matrix.
   */
  template <int NumSources = dynamic, int NumTargets = dynamic,
            int SourceRowOrder = row_major, int TargetRowOrder = row_major>
  auto kernel_matrix(
      const coord_matrix_t<NumSources, SourceRowOrder>& sourceCoords,
      const coord_matrix_t<NumTargets, TargetRowOrder>& targetCoords) {
    const auto sourceValue = potential_vector_t<1>::Ones();
    const size_t numSources = sourceCoords.rows();
    const size_t numTargets = targetCoords.rows();
    // Needs to be column major for 1 column.
    using return_t =
        Eigen::Matrix<potential_t, NumSources, NumTargets, column_major>;
    return_t kernelMatrix = return_t::Zero(numSources, numTargets);

    for (size_t i{0}; i < numSources; ++i) {
      for (size_t j{0}; j < numTargets; ++j) {
        kernelMatrix(i, j) =
            this->potential_P2P(sourceCoords.row(i), targetCoords.row(j));
      }
    }
    return kernelMatrix;
  }

  /** Compute particle to particle interactions.
   * @param leafs A vector of leaf nodes. For each element in this vector, add
   * the interaction from the sources in the element's P2P list without using
   * any equivalent particles.
   **/
  void operator_P2P(nodeptrvec_t& leafs) {
    nodeptrvec_t& targets = leafs;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(targets.size()); i++) {
      node_t* target = targets[i];
      nodeptrvec_t& sources = target->P2Plist();
      for (int j = 0; j < static_cast<int>(sources.size()); j++) {
        node_t* source = sources[j];
        target->target_potentials() += this->potential_P2P(
            source->source_coords(), source->source_strengths(),
            target->target_coords());
        target->target_gradients() += this->gradient_P2P(
            source->source_coords(), source->source_strengths(),
            target->target_coords());
      }
    }
  }

  /** Compute multiple to particle interactions.
   * @param leafs A vector of leaf nodes. For each element in this vector, add
   * the interaction from the sources in the element's M2P list. Uses source
   * equivalent particles with pre-computed source->up_equiv() potentials.
   **/
  void operator_M2P(nodeptrvec_t& leafs) {
    nodeptrvec_t& targets = leafs;
    coord_t c = coord_t::Zero(3);
    std::vector<coord_matrix_t<>> upEquivSurf;
    upEquivSurf.resize(m_depth + 1);
    for (int level = 0; level <= m_depth; level++) {
      upEquivSurf[level].resize(m_numSurf, 3);
      upEquivSurf[level] =
          box_surface_coordinates<potential_t>(m_p, m_r0, level, c, 1.05);
    }
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(targets.size()); i++) {
      node_t& target = *targets[i];
      nodeptrvec_t& sources = target.M2Plist();
      for (size_t j = 0; j < sources.size(); j++) {
        node_t& source = *sources[j];
        int level = source.location().level();
        // source node's equiv coord = relative equiv coord + node's center
        coord_matrix_t<> sourceEquivCoords{upEquivSurf[level]};
        sourceEquivCoords.rowwise() += source.centre();
        target.target_potentials() = this->potential_P2P(
            sourceEquivCoords, source.up_equiv(), target.target_coords());
        target.target_gradients() = this->gradient_P2P(
            sourceEquivCoords, source.up_equiv(), target.target_coords());
      }
    }
  }

  /** Particle to local operator.
   * @param nodes A vector of nodes to apply this operator to.
   **/
  void operator_P2L(nodevec_t& nodes) {
    nodevec_t& targets = nodes;
    std::vector<coord_matrix_t<>> dn_check_surf;
    dn_check_surf.resize(m_depth + 1);
    for (int level = 0; level <= m_depth; level++) {
      dn_check_surf[level].resize(m_numSurf, 3);
      dn_check_surf[level] = box_surface_coordinates<potential_t>(
          m_p, m_r0, level, coord_t::Zero(3), 1.05);
    }
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(targets.size()); i++) {
      node_t* target = &targets[i];
      nodeptrvec_t& sources = {target->P2Llist()};
      for (size_t j = 0; j < sources.size(); j++) {
        node_t* source = sources[j];
        int level = target->location().level();
        // target node's check coord = relative check coord + node's center
        coord_matrix_t<> targetCheckCoords(m_numSurf, 3);
        targetCheckCoords = dn_check_surf[level];
        targetCheckCoords.rowwise() += target->centre();
        target->down_equiv() =
            this->potential_P2P(source->source_coords(),
                                source->source_strengths(), targetCheckCoords);
      }
    }
  }

  /** Evaluate upward equivalent charges for all nodes in a post-order
   * traversal.
   * @param nodes Vector of all nodes.
   * @param leafs Vector of pointers to leaf nodes.
   */
  void upward_pass(nodevec_t& nodes, nodeptrvec_t& leafs, bool verbose = true) {
    start("P2M");
    operator_P2M(leafs);
    stop("P2M", verbose);
    start("M2M");
#pragma omp parallel
#pragma omp single nowait
    operator_M2M(nodes[0]);
    stop("M2M", verbose);
  }

  /** Evaluate potentials and gradients for all targets in a pre-order
   * traversal.
   * @param nodes Vector of all nodes.
   * @param leafs Vector of pointers to leaf nodes.
   */
  void downward_pass(nodevec_t& nodes, nodeptrvec_t& leafs,
                     bool verbose = true) {
    start("P2L");
    operator_P2L(nodes);
    stop("P2L", verbose);
    start("M2P");
    operator_M2P(leafs);
    stop("M2P", verbose);
    start("P2P");
    operator_P2P(leafs);
    stop("P2P", verbose);
    start("M2L");
    operator_M2L(nodes);
    stop("M2L", verbose);
    start("L2L");
    operator_L2L(nodes[0]);
    stop("L2L", verbose);
    start("L2P");
    operator_L2P(leafs);
    stop("L2P", verbose);
  }

  /** Check FMM accuracy by comparison to directly evaluated (N^2) solution.
   * @param leafs Vector of leaves.
   * @param sample Sample only some values, reducing computational cost.
   * @return The relative error of potential and gradient in L2 norm.
   */
  std::vector<real_t> verify(nodeptrvec_t& leafs, bool sample = false) {
    nodevec_t targets;  // vector of target nodes
    if (sample) {
      int nSamples = 10;
      size_t stride = leafs.size() / nSamples;
      for (size_t i = 0; i < nSamples; i++) {
        targets.push_back(*(leafs[i * stride]));
      }
    } else {  // compute all values directly without sampling
      for (size_t i = 0; i < leafs.size(); i++) {
        targets.push_back(*leafs[i]);
      }
    }

    nodevec_t targets2 = targets;  // target2 is used for direct summation
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(targets2.size()); i++) {
      node_t* target = &targets2[i];
      target->zero_target_values();
      for (size_t j = 0; j < leafs.size(); j++) {
        target->target_potentials() += this->potential_P2P(
            leafs[j]->source_coords(), leafs[j]->source_strengths(),
            target->target_coords());
        target->target_gradients() += this->gradient_P2P(
            leafs[j]->source_coords(), leafs[j]->source_strengths(),
            target->target_coords());
      }
    }

    // relative error in L2 norm
    double potentialDiff{0}, potentialNorm{0};
    double gradientDiff{0}, gradientNorm{0};
    for (size_t i = 0; i < targets.size(); i++) {
      potentialNorm += targets2[i].target_potentials().squaredNorm();
      potentialDiff +=
          (targets2[i].target_potentials() - targets[i].target_potentials())
              .squaredNorm();
      gradientNorm += targets2[i].target_gradients().squaredNorm();
      gradientDiff +=
          (targets2[i].target_gradients() - targets[i].target_gradients())
              .squaredNorm();
    }
    std::vector<real_t> err(2);
    err[0] = sqrt(potentialDiff / potentialNorm);
    err[1] = sqrt(gradientDiff / gradientNorm);
    return err;
  }

  /// Allocate memory for precomputed matrices.
  void initialize_matrix() {
    const int nSurf = m_numSurf;
    int depth = m_depth;
    m_matUC2E.resize(depth + 1, potential_matrix_t<>(nSurf, nSurf));
    m_matDC2E.resize(depth + 1, potential_matrix_t<>(nSurf, nSurf));
    m_matM2M.resize(depth + 1);
    m_matL2L.resize(depth + 1);
    for (int level = 0; level <= depth; ++level) {
      std::fill(m_matM2M[level].begin(), m_matM2M[level].end(),
                potential_matrix_t<>(nSurf, nSurf));
      std::fill(m_matL2L[level].begin(), m_matL2L[level].end(),
                potential_matrix_t<>(nSurf, nSurf));
    }
  }

  /** Precompute M2M and L2L matrices.
   * @note Requires that the matrices for computing equivalent source densities
   * from check potentials are precomputed. (matrices UC2E and DC2E).
   **/
  void precompute_M2M_L2L() {
    for (int level = 0; level <= m_depth; level++) {
      auto parent_up_check_surf = box_surface_coordinates<potential_t>(
          m_p, m_r0, level, {0, 0, 0}, 2.95);
      real_t s = m_r0 * std::pow(0.5, level + 1);
      int nPos = static_cast<int>(REL_COORD_M2M.size());
#pragma omp parallel for
      for (int i = 0; i < nPos; i++) {
        ivec3& coord = REL_COORD_M2M[i];
        coord_t childCoord(coord.cast<real_t>() * s);
        auto child_up_equiv_surf = box_surface_coordinates<potential_t>(
            m_p, m_r0, level + 1, childCoord, 1.05);
        // Parent upwards check surface to child upwards equivalent surface.
        // Downwards check to downwards equivalent is transpose of this.
        potential_matrix_t<> matrix_pc2ce =
            kernel_matrix(parent_up_check_surf, child_up_equiv_surf);
        m_matM2M[level][i] = m_matUC2E[level] * matrix_pc2ce;
        m_matL2L[level][i] = m_matDC2E[level] * matrix_pc2ce.transpose();
      }
    }
  }

  /// Precompute operator matrices.
  void precompute() {
    initialize_matrix();
    precompute_check2equiv();
    precompute_M2M_L2L();
    precompute_M2L();
  }

  /** Particle to multiple operator. Computes the equivalent source strengths of
   * a octree cell from the particles contained within it.
   * @param leafs A collection of leaf nodes to apply this operator to.
   **/
  void operator_P2M(nodeptrvec_t& leafs) {
    std::vector<coord_matrix_t<>> upCheckSurf(m_depth + 1,
                                              coord_matrix_t<>(m_numSurf, 3));
    for (int level = 0; level <= m_depth; level++) {
      upCheckSurf[level] = box_surface_coordinates<potential_t>(
          m_p, m_r0, level, {0, 0, 0}, 2.95);
    }
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(leafs.size()); i++) {
      node_t* leaf = leafs[i];
      int level = leaf->location().level();
      // calculate upward check potential induced by sources' charges
      coord_matrix_t<> check_coord{upCheckSurf[level]};
      check_coord.rowwise() += leaf->centre();
      leaf->up_equiv() = this->potential_P2P(
          leaf->source_coords(), leaf->source_strengths(), check_coord);
      Eigen::Matrix<potential_t, Eigen::Dynamic, 1> equiv =
          m_matUC2E[level] * leaf->up_equiv();
      for (int k = 0; k < m_numSurf; k++) {
        leaf->up_equiv()[k] = equiv[k];
      }
    }
  }

  /** Local to target operator.
   * @param leafs A collection of leaf nodes to apply this operator to.
   **/
  void operator_L2P(nodeptrvec_t& leafs) {
    std::vector<coord_matrix_t<>> downEquivSurf(m_depth + 1,
                                                coord_matrix_t<>(m_numSurf, 3));
    for (int level = 0; level <= m_depth; level++) {
      downEquivSurf[level] = box_surface_coordinates<potential_t>(
          m_p, m_r0, level, {0, 0, 0}, 2.95);
    }
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(leafs.size()); i++) {
      node_t* leaf = leafs[i];
      int level = leaf->location().level();
      // down check surface potential -> equivalent surface charge
      potential_vector_t<> equiv = m_matDC2E[level] * leaf->down_equiv();
      leaf->down_equiv() = equiv;
      // equivalent surface charge -> target potential
      coord_matrix_t<> equiv_coord(downEquivSurf[level]);
      equiv_coord.rowwise() += leaf->centre();
      leaf->target_potentials() += this->potential_P2P(
          equiv_coord, leaf->down_equiv(), leaf->target_coords());
      leaf->target_gradients() += this->gradient_P2P(
          equiv_coord, leaf->down_equiv(), leaf->target_coords());
    }
  }

  /** Multiple to multiple operator.
   * @param baseNode The top node in the octree to operate on.
   **/
  void operator_M2M(node_t& baseNode) {
    const int nSurf = m_numSurf;
    if (baseNode.is_leaf()) {
      return;
    }
#pragma omp parallel for schedule(dynamic)
    for (int octant = 0; octant < NCHILD; octant++) {
      if (baseNode.has_child(octant)) {
        operator_M2M(baseNode.child(octant));
      }
    }
    for (int octant = 0; octant < NCHILD; octant++) {
      if (baseNode.has_child(octant)) {
        int level = baseNode.location().level();
        potential_vector_t<> buffer =
            m_matM2M[level][octant] * baseNode.down_equiv();
        baseNode.up_equiv() += buffer;
      }
    }
  }

  /** Local to local operator.
   * @param baseNode The top node in the octree to operate on.
   **/
  void operator_L2L(node_t& baseNode) {
    const int nSurf = m_numSurf;
    if (baseNode.is_leaf()) {
      return;
    }
    for (int octant = 0; octant < NCHILD; octant++) {
      if (baseNode.has_child(octant)) {
        node_t& child = baseNode.child(octant);
        int level = baseNode.location().level();
        potential_vector_t<> buffer =
            m_matL2L[level][octant] * baseNode.down_equiv();
        child.down_equiv() += buffer;
      }
    }
#pragma omp parallel for schedule(dynamic)
    for (int octant = 0; octant < NCHILD; octant++) {
      if (baseNode.has_child(octant)) {
        operator_L2L(baseNode.child(octant));
      }
    }
  }

  /** Precomputations for moment to local operator.
   * Sets m_m2lData.
   * @param nonleafs A vector of pointers to the non-leafs nodes.
   **/
  void setup_M2L(nodeptrvec_t& nonleafs) {
    const int depth = m_depth;
    int nPos = static_cast<int>(REL_COORD_M2L.size());
    m_m2lData.resize(depth);
    // Collect all of the non-leaf nodes on a per-level basis.
    std::vector<nodeptrvec_t> targetNodes(depth);
    for (auto& leafPtr : nonleafs) {
      targetNodes[leafPtr->location().level()].push_back(leafPtr);
    }
    // prepare for m2lData for each level
    for (int l = 0; l < depth; l++) {
      m_m2lData[l] = setup_M2L(targetNodes[l], l);
    }
  }

  /** Compute moment to local operators for a given level.
   * @param levelNodes The non-leaf nodes at this level.
   * @param level The level in the octree.
   * @return An M2LData for this level.
   **/
  M2LData<real_t> setup_M2L(nodeptrvec_t& levelNodes, int level) {
    const int nPos = static_cast<int>(REL_COORD_M2L.size());
    const size_t fftSize = NCHILD * m_numFreq;
    nodeptrvec_t sourceNodes;
    {  // Add every m2l interaction from levelNodes to the sourceNodeSet.
      std::set<node_t*> sourceNodeSet;
      for (auto& node : levelNodes) {
        nodeptrvec_t& m2lList = node->M2Llist();
        for (int k = 0; k < nPos; k++) {
          if (m2lList[k] != nullptr) {
            sourceNodeSet.insert(m2lList[k]);
          }
        }
      }
      // Now turn that into a vector.
      for (auto it = sourceNodeSet.begin(); it != sourceNodeSet.end(); it++) {
        sourceNodes.push_back(*it);
      }
    }
    // prepare the indices of sourceNodes & levelNodes in all_up_equiv &
    // all_dn_equiv
    // displacement in all_up_equiv:
    std::vector<size_t> fftOffset(sourceNodes.size());
    for (size_t i = 0; i < sourceNodes.size(); i++) {
      fftOffset[i] = sourceNodes[i]->child(0).index() * m_numSurf;
    }
    // displacement in all_dn_equiv:
    std::vector<size_t> ifftOffset(levelNodes.size());
    for (size_t i = 0; i < levelNodes.size(); i++) {
      ifftOffset[i] = levelNodes[i]->child(0).index() * m_numSurf;
    }

    // calculate interaction_offset_f & interaction_count_offset
    std::vector<std::pair<size_t, size_t>> interactionOffsetF;
    std::array<size_t, nPos> interactionCountOffset;
    for (size_t i = 0; i < sourceNodes.size(); i++) {
      // node_id: node's index in sourceNodes list
      sourceNodes[i]->indexM2L() = i;
    }
    size_t interactionCountOffsetVar = 0;
    for (int k = 0; k < nPos; k++) {
      for (size_t i{0}; i < levelNodes.size(); i++) {
        nodeptrvec_t& M2L_list = levelNodes[i]->M2Llist();
        if (M2L_list[k] != nullptr) {
          // std::pair{source node's displacement in fftIn, target node's
          // displacement in fftOut}.
          interactionOffsetF.push_back(
              {M2L_list[k]->indexM2L() * fftSize, i * fftSize});
          interactionCountOffsetVar++;
        }
      }
      interactionCountOffset[k] = interactionCountOffsetVar;
    }
    M2LData<real_t> returnData;
    returnData.m_fftOffset = fftOffset;
    returnData.m_ifftOffset = ifftOffset;
    returnData.m_interactionOffsetF = interactionOffsetF;
    returnData.m_interactionCountOffset = interactionCountOffset;
    return returnData;
  }

  std::vector<complex_t> hadamard_product(
      std::array<size_t, static_cast<int>(REL_COORD_M2L.size())>&
          interactionCountOffset,
      std::vector<std::pair<size_t, size_t>>& interactionOffsetF,
      std::vector<complex_t>& fftIn,
      std::vector<std::vector<complex_matrix_t<NCHILD, NCHILD, column_major>>>&
          matrixM2L,
      size_t fftOutSize) {
    const size_t fftSize = NCHILD * m_numFreq;
    std::vector<complex_t> fftOut(fftOutSize, 0);

#pragma omp parallel for schedule(static)
    for (int k = 0; k < m_numFreq; k++) {
      for (size_t iPos = 0; iPos < interactionCountOffset.size(); iPos++) {
        // k-th freq's (row) offset in matrix_M2L:
        complex_matrix_t<NCHILD, NCHILD, column_major>& M = matrixM2L[iPos][k];
        size_t interactionCountOffset0 =
            (iPos == 0 ? 0 : interactionCountOffset[iPos - 1]);
        size_t interactionCountOffset1 = interactionCountOffset[iPos];
        // Matrix vector product {8} = [8,8] * {8} for all interactions:
        for (size_t j = interactionCountOffset0; j < interactionCountOffset1;
             j++) {
          using l_vector_t = Eigen::Matrix<complex_t, 8, 1>;
          using l_mapped_vector_t = Eigen::Map<l_vector_t>;
          auto in = l_mapped_vector_t(fftIn.data() +
                                      interactionOffsetF[j].first + k * NCHILD);
          auto out = l_mapped_vector_t(
              fftOut.data() + interactionOffsetF[j].second + k * NCHILD);
          out += M * in;
        }
      }
    }
    return fftOut;
  }

  void operator_M2L(nodevec_t& nodes) {
    const int nSurf = m_numSurf;
    size_t nNodes = nodes.size();
    constexpr size_t nPos = REL_COORD_M2L.size();
    std::vector<potential_t> allUpEquiv(nNodes * nSurf),
        allDnEquiv(nNodes * nSurf);
    // matrixM2L[nPos index][frequency index] -> 8*8 matrix.
    std::vector<std::vector<complex_matrix_t<NCHILD, NCHILD, column_major>>>
        matrixM2L(
            nPos,
            std::vector<complex_matrix_t<NCHILD, NCHILD, column_major>>(
                m_numFreq, complex_matrix_t<NCHILD, NCHILD, column_major>::Zero(
                               NCHILD, NCHILD)));

    // collect all upward equivalent charges
#pragma omp parallel for schedule(static)
    for (int i = 0; i < nNodes; ++i) {
      for (int j = 0; j < nSurf; ++j) {
        allUpEquiv[i * nSurf + j] = nodes[i].up_equiv()[j];
        allDnEquiv[i * nSurf + j] = nodes[i].down_equiv()[j];
      }
    }
    // FFT-accelerate M2L
    for (size_t l{0}; l < m_depth; ++l) {
      // load M2L matrix for current level
      for (size_t i{0}; i < nPos; ++i) {
        size_t mSize = NCHILD * NCHILD * m_numFreq * sizeof(complex_t);
        std::memcpy(matrixM2L[i].data(), m_matM2L[l][i].data(), mSize);
      }
      std::vector<complex_t> fftIn =
          fft_up_equiv(m_m2lData[l].m_fftOffset, allUpEquiv);
      size_t outputFftSize =
          m_m2lData[l].m_ifftOffset.size() * m_numFreq * NCHILD;
      std::vector<complex_t> fftOut = hadamard_product(
          m_m2lData[l].m_interactionCountOffset,
          m_m2lData[l].m_interactionOffsetF, fftIn, matrixM2L, outputFftSize);
      ifft_dn_check(m_m2lData[l].m_ifftOffset, fftOut, allDnEquiv);
    }
    // update all downward check potentials
#pragma omp parallel for schedule(static)
    for (int i = 0; i < nNodes; ++i) {
      for (int j = 0; j < nSurf; ++j) {
        nodes[i].down_equiv()[j] = allDnEquiv[i * nSurf + j];
      }
    }
  }

  /** Precompute upwards check to equiv (UC2E) and downwads check to equiv
   * (DC2E) matrices.
   * @Note See Ying et al. sec. 3.2.1. SVD is used here instead of of Tikhonov
   * regularization.
   **/
  void precompute_check2equiv() {
    coord_t boxCentre = coord_t::Zero(3);
    //#pragma omp parallel for
    for (int level = 0; level <= m_depth; ++level) {
      // compute kernel matrix
      auto upCheckSurf = box_surface_coordinates<potential_t>(m_p, m_r0, level,
                                                              boxCentre, 2.95);
      auto upEquivSurf = box_surface_coordinates<potential_t>(m_p, m_r0, level,
                                                              boxCentre, 1.05);
      // Upwards check surface to upwards equiv surface matrix. The down check
      // surf to down equiv matrix is the transpose of this.
      potential_matrix_t<> matrix_c2e = kernel_matrix(upCheckSurf, upEquivSurf);
      Eigen::BDCSVD<potential_matrix_t<>> svd(
          matrix_c2e, Eigen::ComputeFullU | Eigen::ComputeFullV);
      auto singularDiag = svd.singularValues();
      auto U = svd.matrixU();
      auto V = svd.matrixV();
      // Pseudo-inverse of singular values matrix, removing negligible terms.
      real_t max_S = std::reduce(
          singularDiag.data(), singularDiag.data() + singularDiag.size(), 0.,
          [](auto a1, auto a2) { return std::max(a1, a2); });
      for (int i = 0; i < m_numSurf; i++) {
        singularDiag(i) = singularDiag(i) > pt::epsilon * max_S * 4
                              ? 1.0 / singularDiag(i)
                              : 0.0;
      }
      auto S_inv = singularDiag.asDiagonal();
      // The psuedo-inverse of matrix_c2e. Upwards check to equivalent.
      m_matUC2E[level] = V * S_inv * U.adjoint();
      // Downwards check to downwards equivalent.
      m_matDC2E[level] = U.conjugate() * S_inv * V.transpose();
    }
  }

  /** Precompute M2L matrices.
   **/
  void precompute_M2L() {
    int fftSize = m_numFreq * NCHILD * NCHILD;
    std::array<std::vector<complex_t>, REL_COORD_M2L_helper.size()>
        matrix_M2L_Helper;
    m_matM2L.resize(m_depth);
    std::fill(matrix_M2L_Helper.begin(), matrix_M2L_Helper.end(),
              std::vector<complex_t>(m_numFreq));

    // create fft plan
    ivec3 dim = ivec3{m_p, m_p, m_p} * 2;
    fft<potential_t, fft_dir::forwards> fftPlan(3, dim.data());

    for (int level = 0; level < m_depth; ++level) {
      // compute M2L kernel matrix, perform DFT
      std::fill(m_matM2L[level].begin(), m_matM2L[level].end(),
                std::vector<complex_t>(fftSize));
#pragma omp parallel for
      for (int i = 0; i < static_cast<int>(REL_COORD_M2L_helper.size()); ++i) {
        coord_t boxCentre;
        for (int d = 0; d < 3; d++) {
          boxCentre[d] = REL_COORD_M2L_helper[i][d] * m_r0 *
                         std::pow(0.5, level);  // relative coords
        }
        coord_matrix_t<dynamic> convolutionCoords =
            convolution_grid<potential_t>(m_p, m_r0, level + 1,
                                          boxCentre);  // convolution grid
        // potentials on convolution grid
        auto convValue =
            kernel_matrix<dynamic>(convolutionCoords, coord_t{coord_t::Zero()});
        fftPlan.execute(convValue.data(), matrix_M2L_Helper[i].data());
      }
// convert M2L_Helper to M2L and reorder data layout to improve locality
#pragma omp parallel for
      for (int i{0}; i < static_cast<int>(REL_COORD_M2L.size()); ++i) {
        for (int j = 0; j < NCHILD * NCHILD;
             j++) {  // loop over child's relative positions
          int childRelIdx = M2L_INDEX_MAP[i][j];
          if (childRelIdx != 123456789) {
            for (int k = 0; k < m_numFreq; k++) {  // loop over frequencies
              int new_idx = k * (NCHILD * NCHILD) + j;
              m_matM2L[level][i][new_idx] = matrix_M2L_Helper[childRelIdx][k] /
                                            complex_t(m_numConvPoints);
            }
          }
        }
      }
    }
  }

  std::vector<complex_t> fft_up_equiv(std::vector<size_t>& fftOffset,
                                      std::vector<potential_t>& allUpEquiv) {
    const int nConv = m_numConvPoints;
    auto map = generate_surf2conv_up<potential_t>(m_p);

    size_t fftSize = NCHILD * m_numFreq;
    std::vector<complex_t> fftIn(fftOffset.size() * fftSize);
    ivec3 dim = ivec3{m_p, m_p, m_p} * 2;
    fft<potential_t, fft_dir::forwards> fftPlan(3, dim.data(), NCHILD, nConv,
                                                m_numFreq);
#pragma omp parallel for
    for (int node_idx = 0; node_idx < static_cast<int>(fftOffset.size());
         node_idx++) {
      std::vector<complex_t> buffer(fftSize, 0);
      std::vector<potential_t> equiv_t(NCHILD * nConv, potential_t(0.));

      for (int k = 0; k < m_numSurf; k++) {
        size_t idx = map[k];
        for (int j = 0; j < NCHILD; j++)
          equiv_t[idx + j * nConv] =
              allUpEquiv[fftOffset[node_idx] + j * m_numSurf + k];
      }
      fftPlan.execute(equiv_t.data(), buffer.data());
      for (int k = 0; k < m_numFreq; k++) {
        for (int j = 0; j < NCHILD; j++) {
          fftIn[fftSize * node_idx + NCHILD * k + j] =
              buffer[m_numFreq * j + k];
        }
      }
    }
    return fftIn;
  }

  void ifft_dn_check(std::vector<size_t>& ifftOffset,
                     std::vector<complex_t>& fftOut,
                     std::vector<potential_t>& allDownEquiv) {
    auto map = generate_surf2conv_dn<potential_t>(m_p);

    size_t fftSize = NCHILD * m_numFreq;
    ivec3 dim = ivec3{m_p, m_p, m_p} * 2;
    fft<potential_t, fft_dir::backwards> fftPlan(3, dim.data(), NCHILD,
                                                 m_numFreq, m_numConvPoints);
#pragma omp parallel for
    for (int node_idx = 0; node_idx < static_cast<int>(ifftOffset.size());
         node_idx++) {
      std::vector<complex_t> fqDomainData(fftSize, 0);
      std::vector<potential_t> tmDomainData(NCHILD * m_numConvPoints, 0);
      potential_t* downEquiv = &allDownEquiv[ifftOffset[node_idx]];
      for (int k = 0; k < m_numFreq; k++) {
        for (int j = 0; j < NCHILD; j++) {
          fqDomainData[m_numFreq * j + k] =
              fftOut[fftSize * node_idx + NCHILD * k + j];
        }
      }
      fftPlan.execute(fqDomainData.data(), tmDomainData.data());
      for (int k = 0; k < m_numSurf; k++) {
        size_t idx = map[k];
        for (int j = 0; j < NCHILD; j++)
          downEquiv[m_numSurf * j + k] +=
              tmDomainData[idx + j * m_numConvPoints];
      }
    }
  }
};

}  // namespace mfmm

#endif  // INCLUDE_MFMM_FMM_H_
