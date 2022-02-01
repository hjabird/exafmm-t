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
#ifndef INCLUDE_EXAFMM_EXAFMM_H_
#define INCLUDE_EXAFMM_EXAFMM_H_

#include <fftw3.h>
#include <omp.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <set>
#include <type_traits>
#include <vector>

#include "align.h"
#include "args.h"
#include "vec.h"

namespace ExaFMM {

/// Use row-major memory order. Like C.
constexpr int row_major = Eigen::RowMajor;
/// Use column-major memory order. Like fortran.
constexpr int column_major = Eigen::ColMajor;
/// Describes resizable matrices.
constexpr int dynamic = Eigen::Dynamic;

/** Helper class to find whether a type is complex.
 *
 * @tparam T The type to test is std::complex.
 **/
template <typename T>
class is_complex : public std::false_type {};

template <typename T>
class is_complex<std::complex<T>> : public std::true_type {};

/** The value_type of a real or complex type.
 *
 * @tparam T The type to get the value real type of.
 **/
template <typename T>
class real_type_of {
 public:
  using type = T;
};

template <typename T>
class real_type_of<std::complex<T>> {
 public:
  using type = T;
};

/** Traits needed for an FMM with given potential type.
 *
 * @tparam PotentialT The type of the potential.
 **/
template <typename PotentialT>
struct potential_traits {
  /// The type of the potential.
  using potential_t = PotentialT;
  /// A real type with the same precision as the potential_t.
  using real_t = typename real_type_of<potential_t>::type;
  /// A complex type with the same precision as the potential_t.
  using complex_t = std::complex<real_t>;

  /// True if the potential type is complex.
  static constexpr bool isComplexPotential = is_complex<potential_t>::value;
  /// The epsilon of the real_t.
  static constexpr real_t epsilon = std::numeric_limits<real_t>::epsilon();

  /// A matrix of the potential_t.
  template <int Rows = dynamic, int Cols = dynamic, int RowOrder = row_major>
  using potential_matrix_t = Eigen::Matrix<potential_t, Rows, Cols, RowOrder>;
  /// A vector of the potential_t.
  template <int Rows = dynamic>
  using potential_vector_t = Eigen::Matrix<potential_t, Rows, 1>;
  /// A type to represent the grad of the potential
  using potential_grad_t = Eigen::Matrix<potential_t, 1, 3>;
  /// Represents a vector of potential_grad_t.
  template <int Rows = dynamic, int RowOrder = column_major>
  using potential_grad_vector_t = Eigen::Matrix<potential_t, Rows, 3, RowOrder>;
  /// A matrix of the real_t.
  template <int Rows = dynamic, int Cols = dynamic, int RowOrder = row_major>
  using real_matrix_t = Eigen::Matrix<real_t, Rows, Cols, RowOrder>;
  /// A vector of the real_t.
  template <int Rows = dynamic>
  using real_vector_t = Eigen::Matrix<real_t, Rows, 1>;
  /// A matrix of the complex_t.
  template <int Rows = dynamic, int Cols = dynamic, int RowOrder = row_major>
  using complex_matrix_t =
      typename Eigen::Matrix<complex_t, Rows, Cols, RowOrder>;
  /// A vector of the complex_t.
  template <int Rows = dynamic>
  using complex_vector_t = Eigen::Matrix<complex_t, Rows, 1>;

  /// Coordinate vector.
  using coord_t = Eigen::Matrix<real_t, 1, 3>;
  /// A vector (as in multiple) coordinates
  template <int Rows = dynamic, int RowOrder = column_major>
  using coord_matrix_t = Eigen::Matrix<real_t, Rows, 3, RowOrder>;
};

const int MEM_ALIGN = 64;
const int CACHE_SIZE = 512;
const int NCHILD = 8;

#if FLOAT
typedef float real_t;  //!< Real number type
const real_t EPS = 1e-8f;
typedef fftwf_complex fft_complex;
typedef fftwf_plan fft_plan;
#define fft_plan_dft fftwf_plan_dft
#define fft_plan_many_dft fftwf_plan_many_dft
#define fft_execute_dft fftwf_execute_dft
#define fft_plan_dft_r2c fftwf_plan_dft_r2c
#define fft_plan_many_dft_r2c fftwf_plan_many_dft_r2c
#define fft_plan_many_dft_c2r fftwf_plan_many_dft_c2r
#define fft_execute_dft_r2c fftwf_execute_dft_r2c
#define fft_execute_dft_c2r fftwf_execute_dft_c2r
#define fft_destroy_plan fftwf_destroy_plan
#define fft_flops fftwf_flops
#else
using real_t = double;  //!< Real number type
const real_t EPS = 1e-16;
using fft_complex = fftw_complex;
using fft_plan = fftw_plan;
#define fft_plan_dft fftw_plan_dft
#define fft_plan_many_dft fftw_plan_many_dft
#define fft_execute_dft fftw_execute_dft
#define fft_plan_dft_r2c fftw_plan_dft_r2c
#define fft_plan_many_dft_r2c fftw_plan_many_dft_r2c
#define fft_plan_many_dft_c2r fftw_plan_many_dft_c2r
#define fft_execute_dft_r2c fftw_execute_dft_r2c
#define fft_execute_dft_c2r fftw_execute_dft_c2r
#define fft_destroy_plan fftw_destroy_plan
#define fft_flops fftw_flops
#endif

const real_t PI = atan(1) * 4;           // M_PI;
using complex_t = std::complex<real_t>;  //!< Complex number type
typedef vec<3, int> ivec3;               //!< Vector of 3 int types
typedef vec<3, real_t> vec3;             //!< Vector of 3 real_t types
typedef vec<3, complex_t> cvec3;         //!< Vector of 3 complex_t types

//! SIMD vector types for AVX512, AVX, and SSE
const int NSIMD =
    SIMD_BYTES /
    int(sizeof(real_t));  //!< SIMD vector length (SIMD_BYTES defined in vec.h)
typedef vec<NSIMD, real_t> simdvec;  //!< SIMD vector of NSIMD real_t types

typedef std::vector<real_t> RealVec;        //!< Vector of real_t types
typedef std::vector<complex_t> ComplexVec;  //!< Vector of complex_t types
typedef AlignedAllocator<real_t, MEM_ALIGN>
    AlignAllocator;  //!< Allocator for memory alignment
typedef std::vector<real_t, AlignAllocator>
    AlignedVec;  //!< Aligned vector of real_t types

//! Interaction types that need to be pre-computed.
typedef enum {
  M2M_Type = 0,
  L2L_Type = 1,
  M2L_Helper_Type = 2,
  M2L_Type = 3,
  Type_Count = 4
} Precompute_Type;

/**
 * @brief Structure of bodies.
 *
 * @tparam T Value type of sources and targets (real or complex).
 */
template <typename T>
struct Body {
  int ibody;  //!< Initial body numbering for sorting back
  typename potential_traits<T>::coord_t X;  //!< Coordinates
  T q;                                      //!< Charge
  T p;                                      //!< Potential
  vec<3, T> F;                              //!< Gradient
};
template <typename T>
using Bodies = std::vector<Body<T>>;  //!< Vector of nodes

/**
 * @brief Structure of nodes.
 *
 * @tparam Value type of sources and targets (real or complex).
 */
template <typename PotentialT>
class Node {
 private:
  using pt = potential_traits<PotentialT>;

 public:
  using potential_t = PotentialT;
  using real_t = typename pt::real_t;
  using potential_vector_t = typename pt::potential_vector_t<>;
  using potential_grad_vector_t =
      typename pt::template potential_grad_vector_t<dynamic>;
  using coord_t = typename pt::coord_t;
  using coord_matrix_t = typename pt::coord_matrix_t<>;
  using node_t = typename Node<potential_t>;
  using nodeptrvec_t = std::vector<node_t*>;

  Node()
      : m_idx{0},
        m_idxM2L{0},
        m_isLeaf{false},
        m_numTargets{0},
        m_numSource{0},
        m_x{0, 0, 0},
        m_r{0},
        m_key{0},
        m_level{0},
        m_octant{0},
        m_parent{nullptr},
        m_sourceCoords(0, 0),
        m_targetCoords(0, 0),
        m_sourceStrengths(0, 0),
        m_targetPotentials(0, 0),
        m_targetGradients(0, 0),
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
      child.m_x(d) += child.m_r * (((octant & 1 << d) >> d) * 2 - 1);
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

// alias template
template <typename T>
using Nodes = std::vector<Node<T>>;  //!< Vector of nodes
template <typename T>
using NodePtrs = std::vector<Node<T>*>;  //!< Vector of Node pointers
using Keys =
    std::vector<std::set<uint64_t>>;  //!< Vector of Morton keys of each level

//! M2L setup data
struct M2LData {
  std::vector<size_t>
      fft_offset;  // source's first child's upward_equiv's displacement
  std::vector<size_t>
      ifft_offset;  // target's first child's dnward_equiv's displacement
  RealVec ifft_scale;
  std::vector<size_t> interaction_offset_f;
  std::vector<size_t> interaction_count_offset;
};

// Relative coordinates and interaction lists
std::vector<std::vector<ivec3>>
    REL_COORD;  //!< Vector of possible relative coordinates (inner) of each
                //!< interaction type (outer)
std::vector<std::vector<int>>
    HASH_LUT;  //!< Vector of hash Lookup tables (inner) of relative positions
               //!< for each interaction type (outer)
std::vector<std::vector<int>>
    M2L_INDEX_MAP;  //!< [M2L_relpos_idx][octant] -> M2L_Helper_relpos_idx
}  // namespace ExaFMM
#endif  // INCLUDE_EXAFMM_EXAFMM_H_
