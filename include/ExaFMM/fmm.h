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
#ifndef INCLUDE_EXAFMM_FMM_H_
#define INCLUDE_EXAFMM_FMM_H_

#include <fftw3.h>

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <algorithm>  // std::fill
#include <fstream>
#include <numeric>

#include "exafmm.h"
#include "geometry.h"
#include "p2p_methods.h"
#include "timer.h"

namespace ExaFMM {

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

  int m_p;       //!< Order of expansion
  int m_numSurf;   //!< Number of points on equivalent / check surface
  int m_numConvPoints;   //!< Number of points on convolution grid
  int m_numFreq;   //!< Number of coefficients in DFT (depending on whether T is
               //!< real_t)
  int m_numCrit;   //!< Max number of bodies per leaf
  int m_depth;   //!< Depth of the tree
  real_t m_r0;   //!< Half of the side length of the bounding box
  coord_t m_x0;  //!< Coordinates of the center of root box
  bool m_isPrecomputed;   //!< Whether the matrix file is found
  std::string m_fileName;  //!< File name of the precomputation matrices

  Fmm() = delete;

  Fmm(int p, int nCrit,
      fmm_kernel_funcs_arg_t kernelArguments = fmm_kernel_funcs_arg_t{},
      std::string fileName = std::string("myPrecomputationMatrix.dat"))
      : p2p_methods<FmmKernel>{kernelArguments},
        m_p{p},
        m_numCrit{nCrit},
        m_fileName{fileName},
        m_numSurf{6 * (p - 1) * (p - 1) + 2},
        m_numConvPoints{8 * p * p * p},
        m_numFreq{0},
        m_isPrecomputed{false} {
        m_numFreq = potential_traits<potential_t>::isComplexPotential ? 
            m_numConvPoints : 4 * p * p * (p + 1);
  }

  ~Fmm() = default;

 protected:
  std::vector<potential_matrix_t<dynamic, dynamic>> matrix_UC2E_U;
  std::vector<potential_matrix_t<dynamic, dynamic>> matrix_UC2E_V;
  std::vector<potential_matrix_t<dynamic, dynamic>> matrix_DC2E_U;
  std::vector<potential_matrix_t<dynamic, dynamic>> matrix_DC2E_V;

  std::vector<std::vector<potential_matrix_t<dynamic, dynamic>>> matrix_M2M;
  std::vector<std::vector<potential_matrix_t<dynamic, dynamic>>> matrix_L2L;

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

  //! P2P operator.
  void P2P(nodeptrvec_t& leafs) {
    nodeptrvec_t& targets = leafs;
    //#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(targets.size()); i++) {
      node_t* target = targets[i];
      nodeptrvec_t& sources = target->P2Plist();
      for (size_t j = 0; j < sources.size(); j++) {
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

  //! M2P operator.
  void M2P(nodeptrvec_t& leafs) {
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
      node_t* target = targets[i];
      nodeptrvec_t& sources = target->M2Plist();
      for (size_t j = 0; j < sources.size(); j++) {
        node_t* source = sources[j];
        int level = source->level();
        // source node's equiv coord = relative equiv coord + node's center
        coord_matrix_t<> sourceEquivCoords{upEquivSurf[level]};
        sourceEquivCoords.rowwise() += source->centre();
        target->target_potentials() = this->potential_P2P(
            sourceEquivCoords, source->up_equiv(), target->target_coords());
        target->target_gradients() = this->gradient_P2P(
            sourceEquivCoords, source->up_equiv(), target->target_coords());
      }
    }
  }

  //! P2L operator.
  void P2L(nodevec_t& nodes) {
    nodevec_t& targets = nodes;
    std::vector<coord_matrix_t<>> dn_check_surf;
    dn_check_surf.resize(m_depth + 1);
    for (int level = 0; level <= m_depth; level++) {
      dn_check_surf[level].resize(m_numSurf, 3);
      dn_check_surf[level] =
          box_surface_coordinates<potential_t>(
              m_p, m_r0, level, coord_t::Zero(3), 1.05);
    }
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(targets.size()); i++) {
      node_t* target = &targets[i];
      nodeptrvec_t& sources = {target->P2Llist()};
      for (size_t j = 0; j < sources.size(); j++) {
        node_t* source = sources[j];
        int level = target->level();
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
   *
   * @param nodes Vector of all nodes.
   * @param leafs Vector of pointers to leaf nodes.
   */
  void upward_pass(nodevec_t& nodes, nodeptrvec_t& leafs, bool verbose = true) {
    start("P2M");
    P2M(leafs);
    stop("P2M", verbose);
    start("M2M");
#pragma omp parallel
#pragma omp single nowait
    M2M(&nodes[0]);
    stop("M2M", verbose);
  }

  /** Evaluate potentials and gradients for all targets in a pre-order
   * traversal.
   *
   * @param nodes Vector of all nodes.
   * @param leafs Vector of pointers to leaf nodes.
   */
  void downward_pass(nodevec_t& nodes, nodeptrvec_t& leafs,
                     bool verbose = true) {
    start("P2L");
    P2L(nodes);
    stop("P2L", verbose);
    start("M2P");
    M2P(leafs);
    stop("M2P", verbose);
    start("P2P");
    P2P(leafs);
    stop("P2P", verbose);
    start("M2L");
    M2L(nodes);
    stop("M2L", verbose);
    start("L2L");
    //#pragma omp parallel
    //#pragma omp single nowait
    L2L(&nodes[0]);
    stop("L2L", verbose);
    start("L2P");
    L2P(leafs);
    stop("L2P", verbose);
  }

  /** Check FMM accuracy by comparison to directly evaluated (N^2) solution.
   *
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
                                   //#pragma omp parallel for
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

  /* precomputation */
  //! Setup the sizes of precomputation matrices
  void initialize_matrix() {
    const int nSurf = m_numSurf;
    int depth = m_depth;
    matrix_UC2E_V.resize(depth + 1, potential_matrix_t<>(nSurf, nSurf));
    matrix_UC2E_U.resize(depth + 1, potential_matrix_t<>(nSurf, nSurf));
    matrix_DC2E_V.resize(depth + 1, potential_matrix_t<>(nSurf, nSurf));
    matrix_DC2E_U.resize(depth + 1, potential_matrix_t<>(nSurf, nSurf));
    matrix_M2M.resize(depth + 1);
    matrix_L2L.resize(depth + 1);
    for (int level = 0; level <= depth; ++level) {
      matrix_M2M[level].resize(REL_COORD[M2M_Type].size(),
                               potential_matrix_t<>(nSurf, nSurf));
      matrix_L2L[level].resize(REL_COORD[L2L_Type].size(),
                               potential_matrix_t<>(nSurf, nSurf));
    }
  }

  //! Precompute M2M and L2L
  void precompute_M2M() {
    coord_t parentCoord{0, 0, 0};
    for (int level = 0; level <= m_depth; level++) {
      auto parent_up_check_surf = box_surface_coordinates<potential_t>(
          m_p, m_r0, level, parentCoord, 2.95);
      real_t s = m_r0 * std::pow(0.5, level + 1);
      int npos = static_cast<int>(
          REL_COORD[M2M_Type].size());  // number of relative positions
#pragma omp parallel for
      for (int i = 0; i < npos; i++) {
        // compute kernel matrix
        ivec3& coord = REL_COORD[M2M_Type][i];
        coord_t childCoord{ parentCoord };
        for (int d{0}; d < 3; ++d) {
            childCoord(d) += coord[d] * s;
        }
        auto child_up_equiv_surf = box_surface_coordinates<potential_t>(
            m_p, m_r0, level + 1, childCoord, 1.05);
        potential_matrix_t<> matrix_pc2ce =
            kernel_matrix(parent_up_check_surf, child_up_equiv_surf);
        // M2M
        matrix_M2M[level][i] =
            matrix_UC2E_V[level] * matrix_UC2E_U[level] * matrix_pc2ce;
        // L2L
        matrix_L2L[level][i] = matrix_pc2ce.transpose() * matrix_DC2E_V[level] *
                               matrix_DC2E_U[level];
      }
    }
  }

  //! Save precomputation matrices
  void save_matrix(std::ofstream& file) {
    file.write(reinterpret_cast<char*>(&m_r0), sizeof(real_t));  // r0
    size_t size = m_numSurf * m_numSurf;
    for (int l = 0; l <= m_depth; l++) {
      // UC2E, DC2E
      file.write(reinterpret_cast<char*>(matrix_UC2E_U[l].data()),
                 size * sizeof(potential_t));
      file.write(reinterpret_cast<char*>(matrix_UC2E_V[l].data()),
                 size * sizeof(potential_t));
      file.write(reinterpret_cast<char*>(matrix_DC2E_U[l].data()),
                 size * sizeof(potential_t));
      file.write(reinterpret_cast<char*>(matrix_DC2E_V[l].data()),
                 size * sizeof(potential_t));
      // M2M, L2L
      for (auto& vec : matrix_M2M[l]) {
        file.write(reinterpret_cast<char*>(vec.data()),
                   size * sizeof(potential_t));
      }
      for (auto& vec : matrix_L2L[l]) {
        file.write(reinterpret_cast<char*>(vec.data()),
                   size * sizeof(potential_t));
      }
    }
  }

  //! Check and load precomputation matrices
  void load_matrix() {
    const int nSurf = m_numSurf;
    const int depth = m_depth;
    size_t size_M2L = m_numFreq * 2 * NCHILD * NCHILD;
    size_t fileSize =
        (2 * REL_COORD[M2M_Type].size() + 4) * nSurf * nSurf * (depth + 1) *
            sizeof(potential_t) +
        REL_COORD[M2L_Type].size() * size_M2L * depth * sizeof(real_t) +
        1 * sizeof(real_t);  // +1 denotes r0
    std::ifstream file(m_fileName, std::ifstream::binary);
    if (file.good()) {
      file.seekg(0, file.end);
      if (size_t(file.tellg()) == fileSize) {  // if file size is correct
        file.seekg(0, file.beg);  // move the position back to the beginning
        real_t r0;
        file.read(reinterpret_cast<char*>(&r0), sizeof(real_t));
        if (m_r0 == r0) {  // if radius match
          size_t size = nSurf * nSurf;
          for (int l = 0; l <= depth; l++) {
            // UC2E, DC2E
            file.read(reinterpret_cast<char*>(matrix_UC2E_U[l].data()),
                      size * sizeof(potential_t));
            file.read(reinterpret_cast<char*>(matrix_UC2E_V[l].data()),
                      size * sizeof(potential_t));
            file.read(reinterpret_cast<char*>(matrix_DC2E_U[l].data()),
                      size * sizeof(potential_t));
            file.read(reinterpret_cast<char*>(matrix_DC2E_V[l].data()),
                      size * sizeof(potential_t));
            // M2M, L2L
            for (auto& vec : matrix_M2M[l]) {
              file.read(reinterpret_cast<char*>(vec.data()),
                        size * sizeof(potential_t));
            }
            for (auto& vec : matrix_L2L[l]) {
              file.read(reinterpret_cast<char*>(vec.data()),
                        size * sizeof(potential_t));
            }
          }
          m_isPrecomputed = true;
        }
      }
    }
    file.close();
  }

  //! Precompute
  void precompute() {
    initialize_matrix();
    // load_matrix();
    if (!m_isPrecomputed) {
      precompute_check2equiv();
      precompute_M2M();
      std::remove(m_fileName.c_str());
      std::ofstream file(m_fileName, std::ios_base::binary);
      save_matrix(file);
      precompute_M2L(file);
      file.close();
    }
  }

  //! P2M operator
  void P2M(nodeptrvec_t& leafs) {
    std::vector<coord_matrix_t<>> upCheckSurf(
        m_depth + 1, coord_matrix_t<>(m_numSurf, 3));
    for (int level = 0; level <= m_depth; level++) {
        upCheckSurf[level] = box_surface_coordinates<potential_t>(
          m_p, m_r0, level, {0, 0, 0}, 2.95);
    }
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(leafs.size()); i++) {
      node_t* leaf = leafs[i];
      int level = leaf->level();
      // calculate upward check potential induced by sources' charges
      coord_matrix_t<> check_coord{ upCheckSurf[level]};
      check_coord.rowwise() += leaf->centre();
      leaf->up_equiv() = this->potential_P2P(
          leaf->source_coords(), leaf->source_strengths(), check_coord);
      Eigen::Matrix<potential_t, Eigen::Dynamic, 1> equiv =
          matrix_UC2E_V[level] * matrix_UC2E_U[level] * leaf->up_equiv();
      for (int k = 0; k < m_numSurf; k++) {
        leaf->up_equiv()[k] = equiv[k];
      }
    }
  }

  //! L2P operator
  void L2P(nodeptrvec_t& leafs) {
    std::vector<coord_matrix_t<>> downEquivSurf(
        m_depth + 1, coord_matrix_t<>(m_numSurf, 3));
    for (int level = 0; level <= m_depth; level++) {
      downEquivSurf[level] = box_surface_coordinates<potential_t>(
          m_p, m_r0, level, {0, 0, 0}, 2.95);
    }
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(leafs.size()); i++) {
      node_t* leaf = leafs[i];
      int level = leaf->level();
      // down check surface potential -> equivalent surface charge
      potential_vector_t<> equiv =
          matrix_DC2E_V[level] * matrix_DC2E_U[level] * leaf->down_equiv();
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

  //! M2M operator
  void M2M(node_t* node) {
    const int nSurf = m_numSurf;
    if (node->is_leaf()) return;
#pragma omp parallel for schedule(dynamic)
    for (int octant = 0; octant < 8; octant++) {
      if (node->has_child(octant)) {
        M2M(&node->child(octant));
      }
    }
    for (int octant = 0; octant < 8; octant++) {
      if (node->has_child(octant)) {
        int level = node->level();
        potential_vector_t<> buffer =
            matrix_M2M[level][octant] * node->down_equiv();
        node->up_equiv() += buffer;
      }
    }
  }

  //! L2L operator
  void L2L(node_t* node) {
    const int nSurf = m_numSurf;
    if (node->is_leaf()) return;
    for (int octant = 0; octant < 8; octant++) {
      if (node->has_child(octant)) {
        node_t& child = node->child(octant);
        int level = node->level();
        potential_vector_t<> buffer =
            matrix_L2L[level][octant] * node->down_equiv();
        child.down_equiv() += buffer;
      }
    }
#pragma omp parallel for schedule(dynamic)
    for (int octant = 0; octant < 8; octant++) {
      if (node->has_child(octant)) {
        L2L(&node->child(octant));
      }
    }
  }

  void M2L_setup(nodeptrvec_t& nonleafs) {
    const int depth = m_depth;
    int nPos = static_cast<int>(
        REL_COORD[M2L_Type].size());  // number of M2L relative positions
    m_m2lData.resize(depth);           // initialize m2ldata

    // construct lists of target nodes for M2L operator at each level
    std::vector<nodeptrvec_t> targetNodes(depth);
    for (size_t i = 0; i < nonleafs.size(); i++) {
      targetNodes[nonleafs[i]->level()].push_back(nonleafs[i]);
    }

    // prepare for m2ldata for each level
    for (int l = 0; l < depth; l++) {
      // construct M2L source nodes for current level
      std::set<node_t*> src_nodes_;
      for (size_t i = 0; i < targetNodes[l].size(); i++) {
        nodeptrvec_t& M2L_list = targetNodes[l][i]->M2Llist();
        for (int k = 0; k < nPos; k++) {
          if (M2L_list[k]) src_nodes_.insert(M2L_list[k]);
        }
      }
      nodeptrvec_t src_nodes;
      auto it = src_nodes_.begin();
      for (; it != src_nodes_.end(); it++) {
        src_nodes.push_back(*it);
      }
      // prepare the indices of src_nodes & targetNodes in all_up_equiv &
      // all_dn_equiv
      std::vector<size_t> fftOffset(
          src_nodes.size());  // displacement in all_up_equiv
      std::vector<size_t> ifftOffset(
          targetNodes[l].size());  // displacement in all_dn_equiv
      for (size_t i = 0; i < src_nodes.size(); i++) {
        fftOffset[i] = src_nodes[i]->child(0).index() * m_numSurf;
      }
      for (size_t i = 0; i < targetNodes[l].size(); i++) {
        ifftOffset[i] = targetNodes[l][i]->child(0).index() * m_numSurf;
      }

      // calculate interaction_offset_f & interaction_count_offset
      std::vector<size_t> interactionOffsetF;
      std::vector<size_t> interactionCountOffset;
      for (size_t i = 0; i < src_nodes.size(); i++) {
        src_nodes[i]->indexM2L() =
            i;  // node_id: node's index in src_nodes list
      }
      size_t nblk_trg = targetNodes[l].size() * sizeof(real_t) / CACHE_SIZE;
      if (nblk_trg == 0) { 
          nblk_trg = 1; 
      }
      size_t interactionCountOffsetVar = 0;
      size_t fftSize = 2 * NCHILD * m_numFreq;
      for (size_t iblk_trg = 0; iblk_trg < nblk_trg; iblk_trg++) {
        size_t blk_start = (targetNodes[l].size() * iblk_trg) / nblk_trg;
        size_t blk_end = (targetNodes[l].size() * (iblk_trg + 1)) / nblk_trg;
        for (int k = 0; k < nPos; k++) {
          for (size_t i = blk_start; i < blk_end; i++) {
            nodeptrvec_t& M2L_list = targetNodes[l][i]->M2Llist();
            if (M2L_list[k]) {
              interactionOffsetF.push_back(
                  M2L_list[k]->indexM2L() *
                  fftSize);  // src_node's displacement in fft_in
              interactionOffsetF.push_back(
                  i * fftSize);  // trg_node's displacement in fft_out
              interactionCountOffsetVar++;
            }
          }
          interactionCountOffset.push_back(interactionCountOffsetVar);
        }
      }
      m_m2lData[l].fft_offset = fftOffset;
      m_m2lData[l].ifft_offset = ifftOffset;
      m_m2lData[l].interaction_offset_f = interactionOffsetF;
      m_m2lData[l].interaction_count_offset = interactionCountOffset;
    }
  }

  void hadamard_product(std::vector<size_t>& interaction_count_offset,
                        std::vector<size_t>& interaction_offset_f,
                        std::vector<real_t>& fft_in,
                        std::vector<real_t>& fft_out,
                        std::vector<std::vector<real_t>>& matrix_M2L) {
    size_t fftSize = 2 * NCHILD * m_numFreq;
    std::vector<real_t> zero_vec0(fftSize, 0.);
    std::vector<real_t> zero_vec1(fftSize, 0.);

    size_t nPos = matrix_M2L.size();
    size_t nblk_inter =
        interaction_count_offset.size();  // num of blocks of interactions
    size_t nblk_trg = nblk_inter / nPos;  // num of blocks based on targetNodes
    int BLOCK_SIZE = CACHE_SIZE * 2 / sizeof(real_t);
    std::vector<real_t*> IN_(BLOCK_SIZE * nblk_inter);
    std::vector<real_t*> OUT_(BLOCK_SIZE * nblk_inter);

    // initialize fft_out with zero
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(fft_out.capacity() / fftSize); ++i) {
      std::memset(fft_out.data() + i * fftSize, 0, fftSize * sizeof(real_t));
    }

#pragma omp parallel for
    for (int iblk_inter = 0; iblk_inter < static_cast<int>(nblk_inter);
         iblk_inter++) {
      size_t interaction_count_offset0 =
          (iblk_inter == 0 ? 0 : interaction_count_offset[iblk_inter - 1]);
      size_t interaction_count_offset1 = interaction_count_offset[iblk_inter];
      size_t interaction_count =
          interaction_count_offset1 - interaction_count_offset0;
      for (size_t j = 0; j < interaction_count; j++) {
        IN_[BLOCK_SIZE * iblk_inter + j] =
            &fft_in[interaction_offset_f[(interaction_count_offset0 + j) * 2 +
                                         0]];
        OUT_[BLOCK_SIZE * iblk_inter + j] =
            &fft_out[interaction_offset_f[(interaction_count_offset0 + j) * 2 +
                                          1]];
      }
      IN_[BLOCK_SIZE * iblk_inter + interaction_count] = &zero_vec0[0];
      OUT_[BLOCK_SIZE * iblk_inter + interaction_count] = &zero_vec1[0];
    }

    for (size_t iblk_trg = 0; iblk_trg < nblk_trg; iblk_trg++) {
      //#pragma omp parallel for
      for (int k = 0; k < m_numFreq; k++) {
        for (size_t ipos = 0; ipos < nPos; ipos++) {
          size_t iblk_inter = iblk_trg * nPos + ipos;
          size_t interaction_count_offset0 =
              (iblk_inter == 0 ? 0 : interaction_count_offset[iblk_inter - 1]);
          size_t interaction_count_offset1 =
              interaction_count_offset[iblk_inter];
          size_t interaction_count =
              interaction_count_offset1 - interaction_count_offset0;
          real_t** IN = &IN_[BLOCK_SIZE * iblk_inter];
          real_t** OUT = &OUT_[BLOCK_SIZE * iblk_inter];
          real_t* M =
              &matrix_M2L[ipos]
                         [k * 2 * NCHILD *
                          NCHILD];  // k-th freq's (row) offset in matrix_M2L
          for (size_t j = 0; j < interaction_count; j += 2) {
            complex_t* M_ = reinterpret_cast<complex_t*>(M);
            complex_t* IN0 = reinterpret_cast<complex_t*>(
                IN[j + 0] + k * NCHILD * 2);  // go to k-th freq chunk
            complex_t* IN1 =
                reinterpret_cast<complex_t*>(IN[j + 1] + k * NCHILD * 2);
            complex_t* OUT0 =
                reinterpret_cast<complex_t*>(OUT[j + 0] + k * NCHILD * 2);
            complex_t* OUT1 =
                reinterpret_cast<complex_t*>(OUT[j + 1] + k * NCHILD * 2);
            using l_matrix_t = Eigen::Matrix<complex_t, 8, 8, column_major>;
            using l_vector_t = Eigen::Matrix<complex_t, 8, 1>;
            using l_mapped_matrix_t = Eigen::Map<l_matrix_t>;
            using l_mapped_vector_t = Eigen::Map<l_vector_t>;
            auto mat = l_mapped_matrix_t(M_);
            auto in0 = l_mapped_vector_t(IN0);
            auto in1 = l_mapped_vector_t(IN1);
            auto out0 = l_mapped_vector_t(OUT0);
            auto out1 = l_mapped_vector_t(OUT1);
            out0 += mat * in0;
            out1 += mat * in1;
          }
        }
      }
    }
  }

  void M2L(nodevec_t& nodes) {
    const int nSurf = m_numSurf;
    int fftSize = 2 * NCHILD * m_numFreq;
    size_t nNodes = nodes.size();
    size_t nPos = REL_COORD[M2L_Type].size();  // number of relative positions

    // allocate memory
    std::vector<potential_t> allUpEquiv, allDnEquiv;
    allUpEquiv.resize(nNodes * nSurf);
    allDnEquiv.resize(nNodes * nSurf);
    std::vector<std::vector<real_t>> matrix_M2L(
        nPos, std::vector<real_t>(fftSize * NCHILD, 0));

    // setup ifstream of M2L precomputation matrix
    std::ifstream ifile(m_fileName, std::ifstream::binary);
    ifile.seekg(0, ifile.end);
    size_t fSize = ifile.tellg();  // file size in bytes
    size_t mSize = NCHILD * NCHILD * m_numFreq * 2 *
                   sizeof(real_t);  // size in bytes for each M2L matrix
    ifile.seekg(fSize - m_depth * nPos * mSize,
                ifile.beg);  // go to the start of M2L section

    // collect all upward equivalent charges
    //#pragma omp parallel for collapse(2)
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
        ifile.read(reinterpret_cast<char*>(matrix_M2L[i].data()), mSize);
      }
      std::vector<real_t> fftIn, fftOut;
      fftIn.resize(m_m2lData[l].fft_offset.size() * fftSize);
      fftOut.resize(m_m2lData[l].ifft_offset.size() * fftSize);
      fft_up_equiv(m_m2lData[l].fft_offset, allUpEquiv, fftIn);
      hadamard_product(m_m2lData[l].interaction_count_offset,
          m_m2lData[l].interaction_offset_f, fftIn, fftOut,
                       matrix_M2L);
      ifft_dn_check(m_m2lData[l].ifft_offset, fftOut, allDnEquiv);
    }
    // update all downward check potentials
    //#pragma omp parallel for collapse(2)
    for (int i = 0; i < nNodes; ++i) {
      for (int j = 0; j < nSurf; ++j) {
        nodes[i].down_equiv()[j] = allDnEquiv[i * nSurf + j];
      }
    }
    ifile.close();  // close ifstream
  }

  /** Precompute UC2E and DC2E matrices.
   *
   * @note See Fong and Darve, Black-box fast multipole method 2009 for relevent
   * literature.
   **/
  void precompute_check2equiv() {
    coord_t boxCentre = coord_t::Zero(3);
    //#pragma omp parallel for
    for (int level = 0; level <= m_depth; ++level) {
      // compute kernel matrix
      auto upCheckSurf = box_surface_coordinates<potential_t>(
          m_p, m_r0, level, boxCentre, 2.95);
      auto upEquivSurf = box_surface_coordinates<potential_t>(
          m_p, m_r0, level, boxCentre, 1.05);
      potential_matrix_t<> matrix_c2e =
          kernel_matrix(upCheckSurf, upEquivSurf);
      Eigen::BDCSVD<potential_matrix_t<>> svd(
          matrix_c2e, Eigen::ComputeFullU | Eigen::ComputeFullV);
      auto singularDiag = svd.singularValues();
      auto U = svd.matrixU();
      auto V = svd.matrixV();
      // pseudo-inverse, removing negligible terms.
      real_t max_S = std::reduce(
          singularDiag.data(), singularDiag.data() + singularDiag.size(), 0.,
          [](auto a1, auto a2) { return std::max(a1, a2); });
      for (int i = 0; i < m_numSurf; i++) {
        singularDiag(i) = singularDiag(i) > pt::epsilon * max_S * 4
                              ? 1.0 / singularDiag(i)
                              : 0.0;
      }
      auto S_inv = singularDiag.asDiagonal();
      matrix_UC2E_U[level] = U.adjoint();
      matrix_UC2E_V[level] = V * S_inv;
      matrix_DC2E_U[level] = V.transpose();
      matrix_DC2E_V[level] = U.conjugate() * S_inv;
    }
  }

  // ################################################################################
  void precompute_M2L(std::ofstream& file) {
    int fftSize = m_numFreq * NCHILD * NCHILD;
    std::vector<std::vector<complex_t>> matrix_M2L_Helper(
        REL_COORD[M2L_Helper_Type].size(), std::vector<complex_t>(m_numFreq));
    std::vector<std::vector<complex_t>> matrix_M2L(REL_COORD[M2L_Type].size(),
                                                std::vector<complex_t>(fftSize));
    // create fft plan
    std::vector<potential_t> fftwIn(m_numConvPoints);
    std::vector<complex_t> fftwOut(m_numFreq);
    ivec3 dim = ivec3{m_p, m_p, m_p} * 2;

    using fft_t = fft<potential_t, fft_dir::forwards>;
    fft_t::plan_t plan;
    plan = fft_t::plan_one(3, dim.data(), fftwIn.data(), fftwOut.data());
    coord_t targetCoord = coord_t::Zero();
    for (int l = 1; l < m_depth + 1; ++l) {
      // compute M2L kernel matrix, perform DFT
      //#pragma omp parallel for
      for (int i = 0; i < static_cast<int>(REL_COORD[M2L_Helper_Type].size());
           ++i) {
        coord_t boxCentre;
        for (int d = 0; d < 3; d++) {
          boxCentre[d] = REL_COORD[M2L_Helper_Type][i][d] * m_r0 *
                         std::pow(0.5, l - 1);  // relative coords
        }
        coord_matrix_t<dynamic> convolutionCoords =
            convolution_grid<potential_t>(m_p, m_r0, l,
                                          boxCentre);  // convolution grid
        // potentials on convolution grid
        auto convValue =
            kernel_matrix<dynamic>(convolutionCoords, targetCoord);
        fft_t::execute(plan, convValue.data(), matrix_M2L_Helper[i].data());
      }
      // convert M2L_Helper to M2L and reorder data layout to improve locality
      //#pragma omp parallel for
      for (int i{0}; i < static_cast<int>(REL_COORD[M2L_Type].size()); ++i) {
        for (int j = 0; j < NCHILD * NCHILD;
             j++) {  // loop over child's relative positions
          int childRelIdx = M2L_INDEX_MAP[i][j];
          if (childRelIdx != -1) {
            for (int k = 0; k < m_numFreq; k++) {  // loop over frequencies
              int new_idx = k * (NCHILD * NCHILD) + j;
              matrix_M2L[i][new_idx] =
                  matrix_M2L_Helper[childRelIdx][k] / complex_t(m_numConvPoints);
            }
          }
        }
      }
      // write to file
      for (auto& vec : matrix_M2L) {
        file.write(reinterpret_cast<char*>(vec.data()),
            fftSize * sizeof(complex_t));
      }
    }
    fft_t::destroy_plan(plan);
  }

  void fft_up_equiv(std::vector<size_t>& fft_offset,
                    std::vector<potential_t>& all_up_equiv,
                    std::vector<real_t>& fft_in) {
    const int nConv = m_numConvPoints;
    auto map = generate_surf2conv_up<potential_t>(m_p);

    size_t fftSize = NCHILD * m_numFreq;
    std::vector<potential_t> fftwIn(nConv * NCHILD);
    std::vector<complex_t> fftwOut(fftSize);
    ivec3 dim = ivec3{m_p, m_p, m_p } * 2;
    using fft_t = fft<potential_t, fft_dir::forwards>;
    fft_t::plan_t plan;
    plan = fft_t::plan_many(3, dim.data(), NCHILD,
        fftwIn.data(), nullptr, 1, nConv,
        fftwOut.data(), nullptr, 1, m_numFreq);
#pragma omp parallel for
    for (int node_idx = 0; node_idx < static_cast<int>(fft_offset.size());
         node_idx++) {
      std::vector<complex_t> buffer(fftSize, 0);
      std::vector<potential_t> equiv_t(NCHILD * nConv, potential_t(0.));

      potential_t* up_equiv =
          &all_up_equiv[fft_offset[node_idx]];  // offset ptr of node's 8
                                                // child's up_equiv in
                                                // all_up_equiv, size=8*m_numSurf_
      complex_t* up_equiv_f = reinterpret_cast<complex_t*>(
          &fft_in[fftSize * 2 * node_idx]);  // offset ptr of node_idx in fft_in
                                         // vector, size=fftsize

      for (int k = 0; k < m_numSurf; k++) {
        size_t idx = map[k];
        for (int j = 0; j < NCHILD; j++)
          equiv_t[idx + j * nConv] = up_equiv[j * m_numSurf + k];
      }
      fft_t::execute(plan, &equiv_t[0], buffer.data());
      for (int k = 0; k < m_numFreq; k++) {
        for (int j = 0; j < NCHILD; j++) {
          up_equiv_f[NCHILD * k + j] = buffer[m_numFreq * j + k];
        }
      }
    }
    fft_t::destroy_plan(plan);
  }

  void ifft_dn_check(std::vector<size_t>& ifft_offset,
                     std::vector<real_t>& fft_out,
                     std::vector<potential_t>& all_dn_equiv) {
    auto map = generate_surf2conv_dn<potential_t>(m_p);

    size_t fftSize = NCHILD * m_numFreq;
    std::vector<complex_t> fftwIn(fftSize);
    std::vector<potential_t> fftwOut(m_numConvPoints * NCHILD);
    ivec3 dim = ivec3{m_p, m_p, m_p} * 2;

    using fft_t = fft<potential_t, fft_dir::backwards>;
    fft_t::plan_t plan;
    plan = fft_t::plan_many(
      3, dim.data(), NCHILD, 
      fftwIn.data(), nullptr, 1, m_numFreq,
      fftwOut.data(), nullptr, 1, m_numConvPoints);

#pragma omp parallel for
    for (int node_idx = 0; node_idx < static_cast<int>(ifft_offset.size());
         node_idx++) {
      std::vector<complex_t> fqDomainData(fftSize, 0);
      std::vector<potential_t> tmDomainData(NCHILD * m_numConvPoints, 0);
      complex_t* dn_check_f = reinterpret_cast<complex_t*>(
          &fft_out[fftSize * 2 * node_idx]);
      potential_t* dn_equiv = &all_dn_equiv[ifft_offset[node_idx]];
      for (int k = 0; k < m_numFreq; k++){
        for (int j = 0; j < NCHILD; j++) {
            fqDomainData[m_numFreq * j + k] = dn_check_f[NCHILD * k + j];
        }
      }
      fft_t::execute(plan, fqDomainData.data(), tmDomainData.data());
      for (int k = 0; k < m_numSurf; k++) {
        size_t idx = map[k];
        for (int j = 0; j < NCHILD; j++)
          dn_equiv[m_numSurf * j + k] += tmDomainData[idx + j * m_numConvPoints];
      }
    }
  fft_t::destroy_plan(plan);
  }
};

}  // namespace ExaFMM

#endif  // INCLUDE_EXAFMM_FMM_H_
