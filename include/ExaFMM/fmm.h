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

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <cassert>
#include <cstring>  // std::memset
#include <fstream>  // std::ofstream
#include <numeric>
#include <type_traits>  // std::is_same

#include "fmm_base.h"
#include "intrinsics.h"

namespace ExaFMM {

template <typename T>
class Fmm : public FmmBase<T> {
  using matrix_t =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using vector_t = Eigen::Matrix<T, Eigen::Dynamic, 1>;

 public:
  /** Vectors of precomputed matrices.
   *
   * U matrices are ...
   * D matrices are ...
   * C2E indicates ...
   * Subscript U indicates ...
   * Subscript V indicates ...
   **/
  std::vector<matrix_t> matrix_UC2E_U;
  std::vector<matrix_t> matrix_UC2E_V;
  std::vector<matrix_t> matrix_DC2E_U;
  std::vector<matrix_t> matrix_DC2E_V;

  std::vector<std::vector<matrix_t>> matrix_M2M;
  std::vector<std::vector<matrix_t>> matrix_L2L;

  std::vector<M2LData> m2ldata;

  /* constructors */
  Fmm() {}

  Fmm(int p_, int ncrit_, std::string filename_ = std::string())
      : FmmBase<T>(p_, ncrit_, filename_) {}

  /* precomputation */
  //! Setup the sizes of precomputation matrices
  void initialize_matrix() {
    int& nsurf_ = this->nsurf;
    int& depth_ = this->depth;
    matrix_UC2E_V.resize(depth_ + 1, matrix_t(nsurf_, nsurf_));
    matrix_UC2E_U.resize(depth_ + 1, matrix_t(nsurf_, nsurf_));
    matrix_DC2E_V.resize(depth_ + 1, matrix_t(nsurf_, nsurf_));
    matrix_DC2E_U.resize(depth_ + 1, matrix_t(nsurf_, nsurf_));
    matrix_M2M.resize(depth_ + 1);
    matrix_L2L.resize(depth_ + 1);
    for (int level = 0; level <= depth_; ++level) {
      matrix_M2M[level].resize(REL_COORD[M2M_Type].size(),
                               matrix_t(nsurf_, nsurf_));
      matrix_L2L[level].resize(REL_COORD[L2L_Type].size(),
                               matrix_t(nsurf_, nsurf_));
    }
  }

  //! Precompute M2M and L2L
  void precompute_M2M() {
    int& nsurf_ = this->nsurf;
    real_t parent_coord[3] = {0, 0, 0};
    for (int level = 0; level <= this->depth; level++) {
      RealVec parent_up_check_surf =
          box_surface_coordinates(this->p, this->r0, level, parent_coord, 2.95);
      real_t s = this->r0 * powf(0.5, level + 1);
      int npos = REL_COORD[M2M_Type].size();  // number of relative positions
#pragma omp parallel for
      for (int i = 0; i < npos; i++) {
        // compute kernel matrix
        ivec3& coord = REL_COORD[M2M_Type][i];
        real_t child_coord[3] = {parent_coord[0] + coord[0] * s,
                                 parent_coord[1] + coord[1] * s,
                                 parent_coord[2] + coord[2] * s};
        RealVec child_up_equiv_surf = box_surface_coordinates(
            this->p, this->r0, level + 1, child_coord, 1.05);
        matrix_t matrix_pc2ce(nsurf_, nsurf_);
        this->kernel_matrix(parent_up_check_surf, child_up_equiv_surf,
                            matrix_pc2ce);
        // M2M
        matrix_M2M[level][i] =
            matrix_UC2E_V[level] * matrix_UC2E_U[level] * matrix_pc2ce;
        // L2L
        matrix_L2L[level][i] = matrix_pc2ce.transpose() * matrix_DC2E_V[level] *
                               matrix_DC2E_U[level];
      }
    }
  }

  //! Precompute M2L
  void precompute_M2L(std::ofstream& file) {}

  //! Save precomputation matrices
  void save_matrix(std::ofstream& file) {
    file.write(reinterpret_cast<char*>(&this->r0), sizeof(real_t));  // r0
    size_t size = this->nsurf * this->nsurf;
    for (int l = 0; l <= this->depth; l++) {
      // UC2E, DC2E
      file.write(reinterpret_cast<char*>(matrix_UC2E_U[l].data()),
                 size * sizeof(T));
      file.write(reinterpret_cast<char*>(matrix_UC2E_V[l].data()),
                 size * sizeof(T));
      file.write(reinterpret_cast<char*>(matrix_DC2E_U[l].data()),
                 size * sizeof(T));
      file.write(reinterpret_cast<char*>(matrix_DC2E_V[l].data()),
                 size * sizeof(T));
      // M2M, L2L
      for (auto& vec : matrix_M2M[l]) {
        file.write(reinterpret_cast<char*>(vec.data()), size * sizeof(T));
      }
      for (auto& vec : matrix_L2L[l]) {
        file.write(reinterpret_cast<char*>(vec.data()), size * sizeof(T));
      }
    }
  }

  //! Check and load precomputation matrices
  void load_matrix() {
    int& nsurf_ = this->nsurf;
    int& depth_ = this->depth;
    size_t size_M2L = this->nfreq * 2 * NCHILD * NCHILD;
    size_t file_size =
        (2 * REL_COORD[M2M_Type].size() + 4) * nsurf_ * nsurf_ * (depth_ + 1) *
            sizeof(T) +
        REL_COORD[M2L_Type].size() * size_M2L * depth_ * sizeof(real_t) +
        1 * sizeof(real_t);  // +1 denotes r0
    std::ifstream file(this->filename, std::ifstream::binary);
    if (file.good()) {
      file.seekg(0, file.end);
      if (size_t(file.tellg()) == file_size) {  // if file size is correct
        file.seekg(0, file.beg);  // move the position back to the beginning
        real_t r0_;
        file.read(reinterpret_cast<char*>(&r0_), sizeof(real_t));
        if (this->r0 == r0_) {  // if radius match
          size_t size = nsurf_ * nsurf_;
          for (int l = 0; l <= depth_; l++) {
            // UC2E, DC2E
            file.read(reinterpret_cast<char*>(matrix_UC2E_U[l].data()),
                      size * sizeof(T));
            file.read(reinterpret_cast<char*>(matrix_UC2E_V[l].data()),
                      size * sizeof(T));
            file.read(reinterpret_cast<char*>(matrix_DC2E_U[l].data()),
                      size * sizeof(T));
            file.read(reinterpret_cast<char*>(matrix_DC2E_V[l].data()),
                      size * sizeof(T));
            // M2M, L2L
            for (auto& vec : matrix_M2M[l]) {
              file.read(reinterpret_cast<char*>(vec.data()), size * sizeof(T));
            }
            for (auto& vec : matrix_L2L[l]) {
              file.read(reinterpret_cast<char*>(vec.data()), size * sizeof(T));
            }
          }
          this->is_precomputed = true;
        }
      }
    }
    file.close();
  }

  //! Precompute
  void precompute() {
    initialize_matrix();
    // load_matrix();
    if (!this->is_precomputed) {
      precompute_check2equiv();
      precompute_M2M();
      std::remove(this->filename.c_str());
      std::ofstream file(this->filename, std::ofstream::binary);
      save_matrix(file);
      precompute_M2L(file);
      file.close();
    }
  }

  //! P2M operator
  void P2M(NodePtrs<T>& leafs) {
    int& nsurf_ = this->nsurf;
    real_t c[3] = {0, 0, 0};
    std::vector<RealVec> up_check_surf;
    up_check_surf.resize(this->depth + 1);
    for (int level = 0; level <= this->depth; level++) {
      up_check_surf[level].resize(nsurf_ * 3);
      up_check_surf[level] =
          box_surface_coordinates(this->p, this->r0, level, c, 2.95);
    }
#pragma omp parallel for
    for (long long int i = 0; i < leafs.size(); i++) {
      Node<T>* leaf = leafs[i];
      int level = leaf->level;
      // calculate upward check potential induced by sources' charges
      RealVec check_coord(nsurf_ * 3);
      for (int k = 0; k < nsurf_; k++) {
        check_coord[3 * k + 0] = up_check_surf[level][3 * k + 0] + leaf->x[0];
        check_coord[3 * k + 1] = up_check_surf[level][3 * k + 1] + leaf->x[1];
        check_coord[3 * k + 2] = up_check_surf[level][3 * k + 2] + leaf->x[2];
      }
      this->potential_P2P(leaf->src_coord, leaf->src_value, check_coord,
                          leaf->up_equiv);
      using eigen_vec_t = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;
      eigen_vec_t eigenUpEquiv(&(leaf->up_equiv[0]), nsurf_);
      Eigen::Matrix<T, Eigen::Dynamic, 1> equiv =
          matrix_UC2E_V[level] * matrix_UC2E_U[level] * eigenUpEquiv;
      for (int k = 0; k < nsurf_; k++) {
        leaf->up_equiv[k] = equiv[k];
      }
    }
  }

  //! L2P operator
  void L2P(NodePtrs<T>& leafs) {
    int& nsurf_ = this->nsurf;
    real_t c[3] = {0, 0, 0};
    std::vector<RealVec> dn_equiv_surf;
    dn_equiv_surf.resize(this->depth + 1);
    for (int level = 0; level <= this->depth; level++) {
      dn_equiv_surf[level].resize(nsurf_ * 3);
      dn_equiv_surf[level] =
          box_surface_coordinates(this->p, this->r0, level, c, 2.95);
    }
#pragma omp parallel for
    for (long long i = 0; i < leafs.size(); i++) {
      Node<T>* leaf = leafs[i];
      int level = leaf->level;
      // down check surface potential -> equivalent surface charge
      using eigen_vec_t = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;
      eigen_vec_t eigenDnEquiv(&(leaf->dn_equiv[0]), nsurf_);
      Eigen::Matrix<T, Eigen::Dynamic, 1> equiv =
          matrix_DC2E_V[level] * matrix_DC2E_U[level] * eigenDnEquiv;
      for (int k = 0; k < nsurf_; k++) {
        leaf->dn_equiv[k] = equiv[k];
      }
      // equivalent surface charge -> target potential
      RealVec equiv_coord(nsurf_ * 3);
      for (int k = 0; k < nsurf_; k++) {
        equiv_coord[3 * k + 0] = dn_equiv_surf[level][3 * k + 0] + leaf->x[0];
        equiv_coord[3 * k + 1] = dn_equiv_surf[level][3 * k + 1] + leaf->x[1];
        equiv_coord[3 * k + 2] = dn_equiv_surf[level][3 * k + 2] + leaf->x[2];
      }
      this->gradient_P2P(equiv_coord, leaf->dn_equiv, leaf->trg_coord,
                         leaf->trg_value);
    }
  }

  //! M2M operator
  void M2M(Node<T>* node) {
    using eigen_vec_t = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;
    int& nsurf_ = this->nsurf;
    if (node->is_leaf) return;
#pragma omp parallel for schedule(dynamic)
    for (int octant = 0; octant < 8; octant++) {
      if (node->children[octant]) M2M(node->children[octant]);
    }
    for (int octant = 0; octant < 8; octant++) {
      if (node->children[octant]) {
        Node<T>* child = node->children[octant];
        int level = node->level;
        eigen_vec_t eigenDnEquiv(&(node->dn_equiv[0]), nsurf_);
        auto buffer = matrix_M2M[level][octant] * eigenDnEquiv;
        for (int k = 0; k < nsurf_; k++) {
          node->up_equiv[k] += buffer[k];
        }
      }
    }
  }

  //! L2L operator
  void L2L(Node<T>* node) {
    using eigen_vec_t = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;
    int& nsurf_ = this->nsurf;
    if (node->is_leaf) return;
    for (int octant = 0; octant < 8; octant++) {
      if (node->children[octant]) {
        Node<T>* child = node->children[octant];
        int level = node->level;
        eigen_vec_t eigenDnEquiv(&(node->dn_equiv[0]), nsurf_);
        Eigen::Matrix<T, Eigen::Dynamic, 1> buffer =
            matrix_L2L[level][octant] * eigenDnEquiv;
        for (int k = 0; k < nsurf_; k++) {
          child->dn_equiv[k] += buffer[k];
        }
      }
    }
#pragma omp parallel for schedule(dynamic)
    for (int octant = 0; octant < 8; octant++) {
      if (node->children[octant]) {
        L2L(node->children[octant]);
      }
    }
  }

  void M2L_setup(NodePtrs<T>& nonleafs) {
    int& nsurf_ = this->nsurf;
    int& depth_ = this->depth;
    int npos = REL_COORD[M2L_Type].size();  // number of M2L relative positions
    m2ldata.resize(depth_);                 // initialize m2ldata

    // construct lists of target nodes for M2L operator at each level
    std::vector<NodePtrs<T>> trg_nodes(depth_);
    for (size_t i = 0; i < nonleafs.size(); i++) {
      trg_nodes[nonleafs[i]->level].push_back(nonleafs[i]);
    }

    // prepare for m2ldata for each level
    for (int l = 0; l < depth_; l++) {
      // construct M2L source nodes for current level
      std::set<Node<T>*> src_nodes_;
      for (size_t i = 0; i < trg_nodes[l].size(); i++) {
        NodePtrs<T>& M2L_list = trg_nodes[l][i]->M2L_list;
        for (int k = 0; k < npos; k++) {
          if (M2L_list[k]) src_nodes_.insert(M2L_list[k]);
        }
      }
      NodePtrs<T> src_nodes;
      auto it = src_nodes_.begin();
      for (; it != src_nodes_.end(); it++) {
        src_nodes.push_back(*it);
      }
      // prepare the indices of src_nodes & trg_nodes in all_up_equiv &
      // all_dn_equiv
      std::vector<size_t> fft_offset(
          src_nodes.size());  // displacement in all_up_equiv
      std::vector<size_t> ifft_offset(
          trg_nodes[l].size());  // displacement in all_dn_equiv
      for (size_t i = 0; i < src_nodes.size(); i++) {
        fft_offset[i] = src_nodes[i]->children[0]->idx * nsurf_;
      }
      for (size_t i = 0; i < trg_nodes[l].size(); i++) {
        ifft_offset[i] = trg_nodes[l][i]->children[0]->idx * nsurf_;
      }

      // calculate interaction_offset_f & interaction_count_offset
      std::vector<size_t> interaction_offset_f;
      std::vector<size_t> interaction_count_offset;
      for (size_t i = 0; i < src_nodes.size(); i++) {
        src_nodes[i]->idx_M2L = i;  // node_id: node's index in src_nodes list
      }
      size_t nblk_trg = trg_nodes[l].size() * sizeof(real_t) / CACHE_SIZE;
      if (nblk_trg == 0) nblk_trg = 1;
      size_t interaction_count_offset_ = 0;
      size_t fft_size = 2 * NCHILD * this->nfreq;
      for (size_t iblk_trg = 0; iblk_trg < nblk_trg; iblk_trg++) {
        size_t blk_start = (trg_nodes[l].size() * iblk_trg) / nblk_trg;
        size_t blk_end = (trg_nodes[l].size() * (iblk_trg + 1)) / nblk_trg;
        for (int k = 0; k < npos; k++) {
          for (size_t i = blk_start; i < blk_end; i++) {
            NodePtrs<T>& M2L_list = trg_nodes[l][i]->M2L_list;
            if (M2L_list[k]) {
              interaction_offset_f.push_back(
                  M2L_list[k]->idx_M2L *
                  fft_size);  // src_node's displacement in fft_in
              interaction_offset_f.push_back(
                  i * fft_size);  // trg_node's displacement in fft_out
              interaction_count_offset_++;
            }
          }
          interaction_count_offset.push_back(interaction_count_offset_);
        }
      }
      m2ldata[l].fft_offset = fft_offset;
      m2ldata[l].ifft_offset = ifft_offset;
      m2ldata[l].interaction_offset_f = interaction_offset_f;
      m2ldata[l].interaction_count_offset = interaction_count_offset;
    }
  }

  void hadamard_product(std::vector<size_t>& interaction_count_offset,
                        std::vector<size_t>& interaction_offset_f,
                        AlignedVec& fft_in, AlignedVec& fft_out,
                        std::vector<AlignedVec>& matrix_M2L) {
    size_t fft_size = 2 * NCHILD * this->nfreq;
    AlignedVec zero_vec0(fft_size, 0.);
    AlignedVec zero_vec1(fft_size, 0.);

    size_t npos = matrix_M2L.size();
    size_t nblk_inter =
        interaction_count_offset.size();  // num of blocks of interactions
    size_t nblk_trg = nblk_inter / npos;  // num of blocks based on trg_nodes
    int BLOCK_SIZE = CACHE_SIZE * 2 / sizeof(real_t);
    std::vector<real_t*> IN_(BLOCK_SIZE * nblk_inter);
    std::vector<real_t*> OUT_(BLOCK_SIZE * nblk_inter);

    // initialize fft_out with zero
#pragma omp parallel for
    for (long long i = 0; i < fft_out.capacity() / fft_size; ++i) {
      std::memset(fft_out.data() + i * fft_size, 0, fft_size * sizeof(real_t));
    }

#pragma omp parallel for
    for (long long iblk_inter = 0; iblk_inter < nblk_inter; iblk_inter++) {
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
#pragma omp parallel for
      for (int k = 0; k < this->nfreq; k++) {
        for (size_t ipos = 0; ipos < npos; ipos++) {
          size_t iblk_inter = iblk_trg * npos + ipos;
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
            real_t* M_ = M;
            real_t* IN0 = IN[j + 0] + k * NCHILD * 2;  // go to k-th freq chunk
            real_t* IN1 = IN[j + 1] + k * NCHILD * 2;
            real_t* OUT0 = OUT[j + 0] + k * NCHILD * 2;
            real_t* OUT1 = OUT[j + 1] + k * NCHILD * 2;
            matmult_8x8x2(M_, IN0, IN1, OUT0, OUT1);
          }
        }
      }
    }
  }

  void fft_up_equiv(std::vector<size_t>& fft_offset,
                    std::vector<T>& all_up_equiv, AlignedVec& fft_in) {}

  void ifft_dn_check(std::vector<size_t>& ifft_offset, AlignedVec& fft_out,
                     std::vector<T>& all_dn_equiv) {}

  void M2L(Nodes<T>& nodes) {
    int& nsurf_ = this->nsurf;
    int& nfreq_ = this->nfreq;
    int fft_size = 2 * NCHILD * nfreq_;
    int nnodes = nodes.size();
    int npos = REL_COORD[M2L_Type].size();  // number of relative positions

    // allocate memory
    std::vector<T> all_up_equiv, all_dn_equiv;
    all_up_equiv.resize(nnodes * nsurf_);
    all_dn_equiv.resize(nnodes * nsurf_);
    std::vector<AlignedVec> matrix_M2L(npos, AlignedVec(fft_size * NCHILD, 0));

    // setup ifstream of M2L precomputation matrix
    std::ifstream ifile(this->filename, std::ifstream::binary);
    ifile.seekg(0, ifile.end);
    size_t fsize = ifile.tellg();  // file size in bytes
    size_t msize = NCHILD * NCHILD * nfreq_ * 2 *
                   sizeof(real_t);  // size in bytes for each M2L matrix
    ifile.seekg(fsize - this->depth * npos * msize,
                ifile.beg);  // go to the start of M2L section

    // collect all upward equivalent charges
#pragma omp parallel for collapse(2)
    for (int i = 0; i < nnodes; ++i) {
      for (int j = 0; j < nsurf_; ++j) {
        all_up_equiv[i * nsurf_ + j] = nodes[i].up_equiv[j];
        all_dn_equiv[i * nsurf_ + j] = nodes[i].dn_equiv[j];
      }
    }
    // FFT-accelerate M2L
    for (int l = 0; l < this->depth; ++l) {
      // load M2L matrix for current level
      for (int i = 0; i < npos; ++i) {
        ifile.read(reinterpret_cast<char*>(matrix_M2L[i].data()), msize);
      }
      AlignedVec fft_in, fft_out;
      fft_in.resize(m2ldata[l].fft_offset.size() * fft_size);
      fft_out.resize(m2ldata[l].ifft_offset.size() * fft_size);
      fft_up_equiv(m2ldata[l].fft_offset, all_up_equiv, fft_in);
      hadamard_product(m2ldata[l].interaction_count_offset,
                       m2ldata[l].interaction_offset_f, fft_in, fft_out,
                       matrix_M2L);
      ifft_dn_check(m2ldata[l].ifft_offset, fft_out, all_dn_equiv);
    }
    // update all downward check potentials
#pragma omp parallel for collapse(2)
    for (int i = 0; i < nnodes; ++i) {
      for (int j = 0; j < nsurf_; ++j) {
        nodes[i].dn_equiv[j] = all_dn_equiv[i * nsurf_ + j];
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
    real_t c[3] = {0, 0, 0};
    int nsurf_ = this->nsurf;
#pragma omp parallel for
    for (int level = 0; level <= this->depth; ++level) {
      // compute kernel matrix
      RealVec up_check_surf =
          box_surface_coordinates(this->p, this->r0, level, c, 2.95);
      RealVec up_equiv_surf =
          box_surface_coordinates(this->p, this->r0, level, c, 1.05);
      matrix_t S_inv = matrix_t::Zero(nsurf_, nsurf_),
               matrix_c2e = matrix_t::Zero(nsurf_, nsurf_);
      this->kernel_matrix(up_check_surf, up_equiv_surf, matrix_c2e);
      Eigen::JacobiSVD<matrix_t> svd(matrix_c2e,
                                     Eigen::ComputeThinU | Eigen::ComputeThinV);
      auto singularDiag = svd.singularValues();
      auto U = svd.matrixU();
      auto V = svd.matrixV();
      // pseudo-inverse, removing negligible terms.
      real_t max_S = std::reduce(singularDiag.data(),
                                 singularDiag.data() + singularDiag.size(), 0.,
                                 [](auto a1, auto a2) {
                                   return std::max(std::abs(a1), std::abs(a2));
                                 });
      for (int i = 0; i < nsurf_; i++) {
        S_inv(i, i) =
            singularDiag(i) > EPS * max_S * 4 ? 1.0 / singularDiag(i) : 0.0;
      }
      matrix_UC2E_U[level] = U.adjoint();
      matrix_UC2E_V[level] = V * S_inv;
      matrix_DC2E_U[level] = V.transpose();
      matrix_DC2E_V[level] = U.conjugate() * S_inv;
    }
  }
};

//! member function specialization for real type
template <>
void Fmm<real_t>::precompute_M2L(std::ofstream& file) {
  int n1 = this->p * 2;
  int& nconv_ = this->nconv;
  int& nfreq_ = this->nfreq;
  int fft_size = 2 * nfreq_ * NCHILD * NCHILD;
  std::vector<RealVec> matrix_M2L_Helper(REL_COORD[M2L_Helper_Type].size(),
                                         RealVec(2 * nfreq_));
  std::vector<AlignedVec> matrix_M2L(REL_COORD[M2L_Type].size(),
                                     AlignedVec(fft_size));
  // create fft plan
  RealVec fftw_in(nconv_);
  RealVec fftw_out(2 * nfreq_);
  int dim[3] = {n1, n1, n1};
  fft_plan plan = fft_plan_dft_r2c(
      3, dim, fftw_in.data(), reinterpret_cast<fft_complex*>(fftw_out.data()),
      FFTW_ESTIMATE);
  RealVec trg_coord(3, 0);
  for (int l = 1; l < this->depth + 1; ++l) {
    // compute M2L kernel matrix, perform DFT
#pragma omp parallel for
    for (long long int i = 0; i < REL_COORD[M2L_Helper_Type].size(); ++i) {
      real_t coord[3];
      for (int d = 0; d < 3; d++) {
        coord[d] = REL_COORD[M2L_Helper_Type][i][d] * this->r0 *
                   powf(0.5, l - 1);  // relative coords
      }
      RealVec conv_coord =
          convolution_grid(this->p, this->r0, l, coord);  // convolution grid
      RealVec conv_value(nconv_);  // potentials on convolution grid
      this->kernel_matrix(conv_coord, trg_coord, conv_value);
      fft_execute_dft_r2c(
          plan, conv_value.data(),
          reinterpret_cast<fft_complex*>(matrix_M2L_Helper[i].data()));
    }
    // convert M2L_Helper to M2L and reorder data layout to improve locality
#pragma omp parallel for
    for (long long int i = 0; i < REL_COORD[M2L_Type].size(); ++i) {
      for (int j = 0; j < NCHILD * NCHILD;
           j++) {  // loop over child's relative positions
        int child_rel_idx = M2L_INDEX_MAP[i][j];
        if (child_rel_idx != -1) {
          for (int k = 0; k < nfreq_; k++) {  // loop over frequencies
            int new_idx = k * (2 * NCHILD * NCHILD) + 2 * j;
            matrix_M2L[i][new_idx + 0] =
                matrix_M2L_Helper[child_rel_idx][k * 2 + 0] / nconv_;  // real
            matrix_M2L[i][new_idx + 1] =
                matrix_M2L_Helper[child_rel_idx][k * 2 + 1] / nconv_;  // imag
          }
        }
      }
    }
    // write to file
    for (auto& vec : matrix_M2L) {
      file.write(reinterpret_cast<char*>(vec.data()),
                 fft_size * sizeof(real_t));
    }
  }
  // destroy fftw plan
  fft_destroy_plan(plan);
}

//! member function specialization for complex type
template <>
void Fmm<complex_t>::precompute_M2L(std::ofstream& file) {
  int n1 = this->p * 2;
  int& nconv_ = this->nconv;
  int& nfreq_ = this->nfreq;
  int fft_size = 2 * nfreq_ * NCHILD * NCHILD;
  std::vector<RealVec> matrix_M2L_Helper(REL_COORD[M2L_Helper_Type].size(),
                                         RealVec(2 * nfreq_));
  std::vector<AlignedVec> matrix_M2L(REL_COORD[M2L_Type].size(),
                                     AlignedVec(fft_size));
  // create fft plan
  RealVec fftw_in(nconv_);
  RealVec fftw_out(2 * nfreq_);
  int dim[3] = {n1, n1, n1};
  fft_plan plan =
      fft_plan_dft(3, dim, reinterpret_cast<fft_complex*>(fftw_in.data()),
                   reinterpret_cast<fft_complex*>(fftw_out.data()),
                   FFTW_FORWARD, FFTW_ESTIMATE);
  RealVec trg_coord(3, 0);
  for (int l = 1; l < this->depth + 1; ++l) {
    // compute M2L kernel matrix, perform DFT
#pragma omp parallel for
    for (long long int i = 0; i < REL_COORD[M2L_Helper_Type].size(); ++i) {
      real_t coord[3];
      for (int d = 0; d < 3; d++) {
        coord[d] = REL_COORD[M2L_Helper_Type][i][d] * this->r0 *
                   powf(0.5, l - 1);  // relative coords
      }
      RealVec conv_coord =
          convolution_grid(this->p, this->r0, l, coord);  // convolution grid
      ComplexVec conv_value(nconv_);  // potentials on convolution grid
      this->kernel_matrix(conv_coord, trg_coord, conv_value);
      fft_execute_dft(
          plan, reinterpret_cast<fft_complex*>(conv_value.data()),
          reinterpret_cast<fft_complex*>(matrix_M2L_Helper[i].data()));
    }
    // convert M2L_Helper to M2L and reorder data layout to improve locality
#pragma omp parallel for
    for (long long int i = 0; i < REL_COORD[M2L_Type].size(); ++i) {
      for (int j = 0; j < NCHILD * NCHILD;
           j++) {  // loop over child's relative positions
        int child_rel_idx = M2L_INDEX_MAP[i][j];
        if (child_rel_idx != -1) {
          for (int k = 0; k < nfreq_; k++) {  // loop over frequencies
            int new_idx = k * (2 * NCHILD * NCHILD) + 2 * j;
            matrix_M2L[i][new_idx + 0] =
                matrix_M2L_Helper[child_rel_idx][k * 2 + 0] / nconv_;  // real
            matrix_M2L[i][new_idx + 1] =
                matrix_M2L_Helper[child_rel_idx][k * 2 + 1] / nconv_;  // imag
          }
        }
      }
    }
    // write to file
    for (auto& vec : matrix_M2L) {
      file.write(reinterpret_cast<char*>(vec.data()),
                 fft_size * sizeof(real_t));
    }
  }
  // destroy fftw plan
  fft_destroy_plan(plan);
}

template <>
void Fmm<real_t>::fft_up_equiv(std::vector<size_t>& fft_offset,
                               RealVec& all_up_equiv, AlignedVec& fft_in) {
  int& nsurf_ = this->nsurf;
  int& nconv_ = this->nconv;
  int& nfreq_ = this->nfreq;
  int n1 = this->p * 2;
  auto map = generate_surf2conv_up(p);

  size_t fft_size = 2 * NCHILD * nfreq_;
  AlignedVec fftw_in(nconv_ * NCHILD);
  AlignedVec fftw_out(fft_size);
  int dim[3] = {n1, n1, n1};
  fft_plan plan = fft_plan_many_dft_r2c(
      3, dim, NCHILD, (real_t*)&fftw_in[0], nullptr, 1, nconv_,
      (fft_complex*)(&fftw_out[0]), nullptr, 1, nfreq_, FFTW_ESTIMATE);

#pragma omp parallel for
  for (long long int node_idx = 0; node_idx < fft_offset.size(); node_idx++) {
    RealVec buffer(fft_size, 0);
    RealVec equiv_t(NCHILD * nconv_, 0.);

    real_t* up_equiv =
        &all_up_equiv[fft_offset[node_idx]];  // offset ptr of node's 8 child's
                                              // up_equiv in all_up_equiv,
                                              // size=8*nsurf_
    real_t* up_equiv_f =
        &fft_in[fft_size * node_idx];  // offset ptr of node_idx in fft_in
                                       // vector, size=fftsize

    for (int k = 0; k < nsurf_; k++) {
      size_t idx = map[k];
      for (int j = 0; j < NCHILD; j++)
        equiv_t[idx + j * nconv_] = up_equiv[j * nsurf_ + k];
    }
    fft_execute_dft_r2c(plan, &equiv_t[0], (fft_complex*)&buffer[0]);
    for (int k = 0; k < nfreq_; k++) {
      for (int j = 0; j < NCHILD; j++) {
        up_equiv_f[2 * (NCHILD * k + j) + 0] = buffer[2 * (nfreq_ * j + k) + 0];
        up_equiv_f[2 * (NCHILD * k + j) + 1] = buffer[2 * (nfreq_ * j + k) + 1];
      }
    }
  }
  fft_destroy_plan(plan);
}

template <>
void Fmm<complex_t>::fft_up_equiv(std::vector<size_t>& fft_offset,
                                  ComplexVec& all_up_equiv,
                                  AlignedVec& fft_in) {
  int& nsurf_ = this->nsurf;
  int& nconv_ = this->nconv;
  int& nfreq_ = this->nfreq;
  int n1 = this->p * 2;
  auto map = generate_surf2conv_up(p);

  size_t fft_size = 2 * NCHILD * nfreq_;
  ComplexVec fftw_in(nconv_ * NCHILD);
  AlignedVec fftw_out(fft_size);
  int dim[3] = {n1, n1, n1};

  fft_plan plan = fft_plan_many_dft(
      3, dim, NCHILD, reinterpret_cast<fft_complex*>(&fftw_in[0]), nullptr, 1,
      nconv_, (fft_complex*)(&fftw_out[0]), nullptr, 1, nfreq_, FFTW_FORWARD,
      FFTW_ESTIMATE);

#pragma omp parallel for
  for (long long int node_idx = 0; node_idx < fft_offset.size(); node_idx++) {
    RealVec buffer(fft_size, 0);
    ComplexVec equiv_t(NCHILD * nconv_, complex_t(0., 0.));

    complex_t* up_equiv =
        &all_up_equiv[fft_offset[node_idx]];  // offset ptr of node's 8 child's
                                              // up_equiv in all_up_equiv,
                                              // size=8*nsurf_
    real_t* up_equiv_f =
        &fft_in[fft_size * node_idx];  // offset ptr of node_idx in fft_in
                                       // vector, size=fftsize

    for (int k = 0; k < nsurf_; k++) {
      size_t idx = map[k];
      for (int j = 0; j < NCHILD; j++)
        equiv_t[idx + j * nconv_] = up_equiv[j * nsurf_ + k];
    }
    fft_execute_dft(plan, reinterpret_cast<fft_complex*>(&equiv_t[0]),
                    (fft_complex*)&buffer[0]);
    for (int k = 0; k < nfreq_; k++) {
      for (int j = 0; j < NCHILD; j++) {
        up_equiv_f[2 * (NCHILD * k + j) + 0] = buffer[2 * (nfreq_ * j + k) + 0];
        up_equiv_f[2 * (NCHILD * k + j) + 1] = buffer[2 * (nfreq_ * j + k) + 1];
      }
    }
  }
  fft_destroy_plan(plan);
}

template <>
void Fmm<real_t>::ifft_dn_check(std::vector<size_t>& ifft_offset,
                                AlignedVec& fft_out, RealVec& all_dn_equiv) {
  int& nsurf_ = this->nsurf;
  int& nconv_ = this->nconv;
  int& nfreq_ = this->nfreq;
  int n1 = this->p * 2;
  auto map = generate_surf2conv_dn(p);

  size_t fft_size = 2 * NCHILD * nfreq_;
  AlignedVec fftw_in(fft_size);
  AlignedVec fftw_out(nconv_ * NCHILD);
  int dim[3] = {n1, n1, n1};

  fft_plan plan = fft_plan_many_dft_c2r(
      3, dim, NCHILD, (fft_complex*)(&fftw_in[0]), nullptr, 1, nfreq_,
      (real_t*)(&fftw_out[0]), nullptr, 1, nconv_, FFTW_ESTIMATE);

#pragma omp parallel for
  for (long long int node_idx = 0; node_idx < ifft_offset.size(); node_idx++) {
    RealVec buffer0(fft_size, 0);
    RealVec buffer1(fft_size, 0);
    real_t* dn_check_f =
        &fft_out[fft_size * node_idx];  // offset ptr for node_idx in fft_out
                                        // vector, size=fftsize
    real_t* dn_equiv =
        &all_dn_equiv[ifft_offset[node_idx]];  // offset ptr for node_idx's
                                               // child's dn_equiv in
                                               // all_dn_equiv, size=numChilds *
                                               // nsurf_
    for (int k = 0; k < nfreq_; k++)
      for (int j = 0; j < NCHILD; j++) {
        buffer0[2 * (nfreq_ * j + k) + 0] =
            dn_check_f[2 * (NCHILD * k + j) + 0];
        buffer0[2 * (nfreq_ * j + k) + 1] =
            dn_check_f[2 * (NCHILD * k + j) + 1];
      }
    fft_execute_dft_c2r(plan, (fft_complex*)&buffer0[0],
                        (real_t*)(&buffer1[0]));
    for (int k = 0; k < nsurf_; k++) {
      size_t idx = map[k];
      for (int j = 0; j < NCHILD; j++)
        dn_equiv[nsurf_ * j + k] += buffer1[idx + j * nconv_];
    }
  }
  fft_destroy_plan(plan);
}

template <>
void Fmm<complex_t>::ifft_dn_check(std::vector<size_t>& ifft_offset,
                                   AlignedVec& fft_out,
                                   ComplexVec& all_dn_equiv) {
  int& nsurf_ = this->nsurf;
  int& nconv_ = this->nconv;
  int& nfreq_ = this->nfreq;
  assert(fft_out.size() >= ifft_offset.size() * nfreq_ * NCHILD);
  int n1 = this->p * 2;
  auto map = generate_surf2conv_dn(p);

  size_t fft_size = 2 * NCHILD * nfreq_;
  AlignedVec fftw_in(fft_size);
  ComplexVec fftw_out(nconv_ * NCHILD);
  int dim[3] = {n1, n1, n1};

  fft_plan plan =
      fft_plan_many_dft(3, dim, NCHILD, (fft_complex*)(&fftw_in[0]), nullptr, 1,
                        nfreq_, reinterpret_cast<fft_complex*>(&fftw_out[0]),
                        nullptr, 1, nconv_, FFTW_BACKWARD, FFTW_ESTIMATE);

#pragma omp parallel for
  for (long long int node_idx = 0; node_idx < ifft_offset.size(); node_idx++) {
    RealVec buffer0(fft_size, 0);
    ComplexVec buffer1(NCHILD * nconv_, 0);
    real_t* dn_check_f = &fft_out[fft_size * node_idx];
    complex_t* dn_equiv = &all_dn_equiv[ifft_offset[node_idx]];
    for (int k = 0; k < nfreq_; k++)
      for (int j = 0; j < NCHILD; j++) {
        buffer0[2 * (nfreq_ * j + k) + 0] =
            dn_check_f[2 * (NCHILD * k + j) + 0];
        buffer0[2 * (nfreq_ * j + k) + 1] =
            dn_check_f[2 * (NCHILD * k + j) + 1];
      }
    fft_execute_dft(plan, (fft_complex*)&buffer0[0],
                    reinterpret_cast<fft_complex*>(&buffer1[0]));
    for (int k = 0; k < nsurf_; k++) {
      size_t idx = map[k];
      for (int j = 0; j < NCHILD; j++)
        dn_equiv[nsurf_ * j + k] += buffer1[idx + j * nconv_];
    }
  }
  fft_destroy_plan(plan);
}
}  // namespace ExaFMM
#endif  // INCLUDE_EXAFMM_FMM_H_
