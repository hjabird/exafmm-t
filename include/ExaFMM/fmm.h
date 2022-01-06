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
#include "timer.h"

namespace ExaFMM {

	//! Base FMM class
	template <class FmmKernel>
	class Fmm : public FmmKernel {
	public:
		using potential_t = typename FmmKernel::potential_t;

	protected:
		using pt = potential_traits<potential_t>;

	public:
		using real_t = typename pt::real_t;
		using complex_t = typename pt::complex_t;
		using fmm_kernel_funcs_arg_t = typename FmmKernel::kernel_args_t;


		template <int Rows = dynamic, int Cols = dynamic, int RowOrder = row_major>
		using potential_matrix_t = typename
			pt::template potential_matrix_t<Rows, Cols, RowOrder>;
		template <int Rows = dynamic>
		using potential_vector_t = typename pt::template potential_vector_t<Rows>;
		template <int Rows = dynamic, int Cols = dynamic, int RowOrder = row_major>
		using real_matrix_t = typename pt::template real_matrix_t<Rows, Cols, RowOrder>;
		template <int Rows = dynamic>
		using real_vector_t = typename pt::template real_vector_t<Rows>;
		template <int Rows = dynamic, int Cols = dynamic, int RowOrder = row_major>
		using complex_matrix_t = typename pt::template complex_matrix_t<Rows, Cols, RowOrder>;
		template <int Rows = dynamic>
		using complex_vector_t = typename pt::template complex_vector_t<Rows>;
		using coord_t = typename pt::coord_t;
		template <int Rows = dynamic>
		using coord_matrix_t = typename pt::template coord_matrix_t<Rows>;
		using node_t = Node<potential_t>;
		using nodevec_t = std::vector<node_t>;
		using nodeptrvec_t = std::vector<node_t*>;

		int p;      //!< Order of expansion
		int nsurf;  //!< Number of points on equivalent / check surface
		int nconv;  //!< Number of points on convolution grid
		int nfreq;  //!< Number of coefficients in DFT (depending on whether T is
					//!< real_t)
		int ncrit;  //!< Max number of bodies per leaf
		int depth;  //!< Depth of the tree
		real_t r0;  //!< Half of the side length of the bounding box
		coord_t x0;    //!< Coordinates of the center of root box
		bool is_precomputed;   //!< Whether the matrix file is found
		bool is_real;          //!< Whether template parameter T is real_t
		std::string filename;  //!< File name of the precomputation matrices

		Fmm() = delete;

		Fmm(int p_, int ncrit_, fmm_kernel_funcs_arg_t kernelArguments,
			std::string filename_ = std::string())
			: FmmKernel{ kernelArguments },
			p{ p_ },
			ncrit{ ncrit_ },
			filename{ filename_ },
			nsurf{ 6 * (p_ - 1) * (p_ - 1) + 2 },
			nconv{ 8 * p_ * p_ * p_ },
			is_real{ !potential_traits<potential_t>::isComplexPotential },
			nfreq{ is_real ? 4 * p_ * p_ * (p_ + 1) : nconv },
			is_precomputed{ false } {}

		~Fmm() = default;

	protected:
		std::vector<potential_matrix_t<dynamic, dynamic>> matrix_UC2E_U;
		std::vector<potential_matrix_t<dynamic, dynamic>> matrix_UC2E_V;
		std::vector<potential_matrix_t<dynamic, dynamic>> matrix_DC2E_U;
		std::vector<potential_matrix_t<dynamic, dynamic>> matrix_DC2E_V;

		std::vector<std::vector<potential_matrix_t<dynamic, dynamic>>> matrix_M2M;
		std::vector<std::vector<potential_matrix_t<dynamic, dynamic>>> matrix_L2L;

		std::vector<M2LData> m2ldata;


	public:
		/** Compute the kernel matrix of a given kernel.
		 *
		 * The kernel matrix defines the interaction between the sources and the
		 * targets: targetVal = kernelMatrix * sourceStrength.
		 * This function evaluates the interaction kernel using unit source strength
		 * to obtain each value in the matrix.
		 *
		 * @param sourceCoord Vector of source coordinates.
		 * @param targetCoord Vector of target coordinates.
		 * @return matrix Kernel matrix.
		 */
		auto kernel_matrix(RealVec& sourceCoord, RealVec& targetCoord) {
			auto sourceValue = potential_vector_t<1>::Ones();
			int numSources = sourceCoord.size() / 3;
			int numTargets = targetCoord.size() / 3;
			using return_t = Eigen::Matrix<potential_t, Eigen::Dynamic, Eigen::Dynamic,
				Eigen::RowMajor>;
			return_t kernelMatrix = return_t::Zero(numSources, numTargets);
			// Evaluate matrix one row at a time.
#pragma omp parallel for
			for (int i = 0; i < numSources; i++) {
				RealVec sourceCoordinates(sourceCoord.data() + 3 * i,
					sourceCoord.data() + 3 * (i + 1));
				potential_vector_t<> targetValue(numTargets);
				targetValue.setZero();
				potential_P2P<1, dynamic>(sourceCoordinates, sourceValue, targetCoord, targetValue);
				kernelMatrix.row(i) =
					Eigen::Map<Eigen::Matrix<potential_t, 1, Eigen::Dynamic>>(
						targetValue.data(), 1, numTargets);
			}
			return kernelMatrix;
		}

		/* the following kernels do not use precomputation matrices
		 * thus can be defined in the base class */

		 //! P2P operator.
		void P2P(nodeptrvec_t& leafs) {
			nodeptrvec_t& targets = leafs;
#pragma omp parallel for
			for (long long i = 0; i < targets.size(); i++) {
				node_t* target = targets[i];
				nodeptrvec_t& sources = target->P2P_list;
				for (size_t j = 0; j < sources.size(); j++) {
					node_t* source = sources[j];
					gradient_P2P(source->src_coord, source->src_value, target->trg_coord,
						target->trg_value);
				}
			}
		}

		//! M2P operator.
		void M2P(nodeptrvec_t& leafs) {
			nodeptrvec_t& targets = leafs;
			real_t c[3] = { 0.0 };
			std::vector<RealVec> up_equiv_surf;
			up_equiv_surf.resize(depth + 1);
			for (int level = 0; level <= depth; level++) {
				up_equiv_surf[level].resize(nsurf * 3);
				up_equiv_surf[level] = box_surface_coordinates(p, r0, level, c, 1.05);
			}
#pragma omp parallel for
			for (long long i = 0; i < targets.size(); i++) {
				node_t* target = targets[i];
				nodeptrvec_t& sources = target->M2P_list;
				for (size_t j = 0; j < sources.size(); j++) {
					node_t* source = sources[j];
					RealVec src_equiv_coord(nsurf * 3);
					int level = source->level;
					// source node's equiv coord = relative equiv coord + node's center
					for (int k = 0; k < nsurf; k++) {
						src_equiv_coord[3 * k + 0] =
							up_equiv_surf[level][3 * k + 0] + source->x[0];
						src_equiv_coord[3 * k + 1] =
							up_equiv_surf[level][3 * k + 1] + source->x[1];
						src_equiv_coord[3 * k + 2] =
							up_equiv_surf[level][3 * k + 2] + source->x[2];
					}
					gradient_P2P(src_equiv_coord, source->up_equiv, target->trg_coord,
						target->trg_value);
				}
			}
		}

		//! P2L operator.
		void P2L(nodevec_t& nodes) {
			nodevec_t& targets = nodes;
			real_t c[3] = { 0.0 };
			std::vector<RealVec> dn_check_surf;
			dn_check_surf.resize(depth + 1);
			for (int level = 0; level <= depth; level++) {
				dn_check_surf[level].resize(nsurf * 3);
				dn_check_surf[level] = box_surface_coordinates(p, r0, level, c, 1.05);
			}
#pragma omp parallel for
			for (long long i = 0; i < targets.size(); i++) {
				node_t* target = &targets[i];
				nodeptrvec_t& sources = target->P2L_list;
				for (size_t j = 0; j < sources.size(); j++) {
					node_t* source = sources[j];
					RealVec trg_check_coord(nsurf * 3);
					int level = target->level;
					// target node's check coord = relative check coord + node's center
					for (int k = 0; k < nsurf; k++) {
						trg_check_coord[3 * k + 0] =
							dn_check_surf[level][3 * k + 0] + target->x[0];
						trg_check_coord[3 * k + 1] =
							dn_check_surf[level][3 * k + 1] + target->x[1];
						trg_check_coord[3 * k + 2] =
							dn_check_surf[level][3 * k + 2] + target->x[2];
					}
					potential_P2P(source->src_coord, source->src_value, trg_check_coord,
						target->dn_equiv);
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
#pragma omp parallel
#pragma omp single nowait
			L2L(&nodes[0]);
			stop("L2L", verbose);
			start("L2P");
			L2P(leafs);
			stop("L2P", verbose);
		}

		/** Check FMM accuracy.
		 *
		 * @param leafs Vector of leaves.
		 * @return The relative error of potential and gradient in L2 norm.
		 */
		RealVec verify(nodeptrvec_t& leafs, bool sample = false) {
			nodevec_t targets;  // vector of target nodes
			if (sample) {
				int nsamples = 10;
				int stride = leafs.size() / nsamples;
				for (int i = 0; i < nsamples; i++) {
					targets.push_back(*(leafs[i * stride]));
				}
			}
			else {  // compute all values directly without sampling
				for (size_t i = 0; i < leafs.size(); i++) {
					targets.push_back(*leafs[i]);
				}
			}

			nodevec_t targets2 = targets;  // target2 is used for direct summation
#pragma omp parallel for
			for (long long int i = 0; i < targets2.size(); i++) {
				node_t* target = &targets2[i];
				target->trg_value.setZero();
				for (size_t j = 0; j < leafs.size(); j++) {
					gradient_P2P(leafs[j]->src_coord, leafs[j]->src_value,
						target->trg_coord, target->trg_value);
				}
			}

			// relative error in L2 norm
			double p_diff = 0, p_norm = 0, g_diff = 0, g_norm = 0;
			for (size_t i = 0; i < targets.size(); i++) {
				for (int j = 0; j < targets[i].ntrgs; j++) {
					p_norm += std::norm(targets2[i].trg_value[4 * j + 0]);
					p_diff += std::norm(targets2[i].trg_value[4 * j + 0] -
						targets[i].trg_value[4 * j + 0]);
					for (int d = 1; d < 4; d++) {
						g_diff += std::norm(targets2[i].trg_value[4 * j + d] -
							targets[i].trg_value[4 * j + d]);
						g_norm += std::norm(targets2[i].trg_value[4 * j + d]);
					}
				}
			}
			RealVec err(2);
			err[0] = sqrt(p_diff / p_norm);  // potential error in L2 norm
			err[1] = sqrt(g_diff / g_norm);  // gradient error in L2 norm

			return err;
		}

		/* precomputation */
		//! Setup the sizes of precomputation matrices
		void initialize_matrix() {
			int& nsurf_ = this->nsurf;
			int& depth_ = this->depth;
			matrix_UC2E_V.resize(depth_ + 1, potential_matrix_t<>(nsurf_, nsurf_));
			matrix_UC2E_U.resize(depth_ + 1, potential_matrix_t<>(nsurf_, nsurf_));
			matrix_DC2E_V.resize(depth_ + 1, potential_matrix_t<>(nsurf_, nsurf_));
			matrix_DC2E_U.resize(depth_ + 1, potential_matrix_t<>(nsurf_, nsurf_));
			matrix_M2M.resize(depth_ + 1);
			matrix_L2L.resize(depth_ + 1);
			for (int level = 0; level <= depth_; ++level) {
				matrix_M2M[level].resize(REL_COORD[M2M_Type].size(),
					potential_matrix_t<>(nsurf_, nsurf_));
				matrix_L2L[level].resize(REL_COORD[L2L_Type].size(),
					potential_matrix_t<>(nsurf_, nsurf_));
			}
		}

		//! Precompute M2M and L2L
		void precompute_M2M() {
			int& nsurf_ = this->nsurf;
			real_t parent_coord[3] = { 0, 0, 0 };
			for (int level = 0; level <= this->depth; level++) {
				RealVec parent_up_check_surf =
					box_surface_coordinates(this->p, this->r0, level, parent_coord, 2.95);
				real_t s = this->r0 * std::pow(0.5, level + 1);
				int npos = REL_COORD[M2M_Type].size();  // number of relative positions
#pragma omp parallel for
				for (int i = 0; i < npos; i++) {
					// compute kernel matrix
					ivec3& coord = REL_COORD[M2M_Type][i];
					real_t child_coord[3] = { parent_coord[0] + coord[0] * s,
											 parent_coord[1] + coord[1] * s,
											 parent_coord[2] + coord[2] * s };
					RealVec child_up_equiv_surf = box_surface_coordinates(
						this->p, this->r0, level + 1, child_coord, 1.05);
					potential_matrix_t<> matrix_pc2ce =
						this->kernel_matrix(parent_up_check_surf, child_up_equiv_surf);
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
			file.write(reinterpret_cast<char*>(&this->r0), sizeof(real_t));  // r0
			size_t size = this->nsurf * this->nsurf;
			for (int l = 0; l <= this->depth; l++) {
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
				std::ofstream file(this->filename, std::ios_base::binary);
				save_matrix(file);
				precompute_M2L(file);
				file.close();
			}
		}

		//! P2M operator
		void P2M(nodeptrvec_t& leafs) {
			int& nsurf_ = this->nsurf;
			real_t c[3] = { 0, 0, 0 };
			std::vector<RealVec> up_check_surf;
			up_check_surf.resize(this->depth + 1);
			for (int level = 0; level <= this->depth; level++) {
				up_check_surf[level].resize(nsurf_ * 3);
				up_check_surf[level] =
					box_surface_coordinates(this->p, this->r0, level, c, 2.95);
			}
#pragma omp parallel for
			for (long long int i = 0; i < leafs.size(); i++) {
				node_t* leaf = leafs[i];
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
				Eigen::Matrix<potential_t, Eigen::Dynamic, 1> equiv =
					matrix_UC2E_V[level] * matrix_UC2E_U[level] * leaf->up_equiv;
				for (int k = 0; k < nsurf_; k++) {
					leaf->up_equiv[k] = equiv[k];
				}
			}
		}

		//! L2P operator
		void L2P(nodeptrvec_t& leafs) {
			int& nsurf_ = this->nsurf;
			real_t c[3] = { 0, 0, 0 };
			std::vector<RealVec> dn_equiv_surf;
			dn_equiv_surf.resize(this->depth + 1);
			for (int level = 0; level <= this->depth; level++) {
				dn_equiv_surf[level].resize(nsurf_ * 3);
				dn_equiv_surf[level] =
					box_surface_coordinates(this->p, this->r0, level, c, 2.95);
			}
#pragma omp parallel for
			for (long long i = 0; i < leafs.size(); i++) {
				node_t* leaf = leafs[i];
				int level = leaf->level;
				// down check surface potential -> equivalent surface charge
				potential_vector_t<> equiv =
					matrix_DC2E_V[level] * matrix_DC2E_U[level] * leaf->dn_equiv;
				leaf->dn_equiv = equiv;
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
		void M2M(node_t* node) {
			const int nsurf_ = this->nsurf;
			if (node->is_leaf) return;
#pragma omp parallel for schedule(dynamic)
			for (int octant = 0; octant < 8; octant++) {
				if (node->children[octant]) M2M(node->children[octant]);
			}
			for (int octant = 0; octant < 8; octant++) {
				if (node->children[octant]) {
					node_t* child = node->children[octant];
					int level = node->level;
					potential_vector_t<> buffer = matrix_M2M[level][octant] * node->dn_equiv;
					node->up_equiv += buffer;
				}
			}
		}

		//! L2L operator
		void L2L(node_t* node) {
			const int nsurf_ = this->nsurf;
			if (node->is_leaf) return;
			for (int octant = 0; octant < 8; octant++) {
				if (node->children[octant]) {
					node_t* child = node->children[octant];
					int level = node->level;
					potential_vector_t<> buffer = matrix_L2L[level][octant] * node->dn_equiv;
					child->dn_equiv += buffer;
				}
			}
#pragma omp parallel for schedule(dynamic)
			for (int octant = 0; octant < 8; octant++) {
				if (node->children[octant]) {
					L2L(node->children[octant]);
				}
			}
		}

		void M2L_setup(nodeptrvec_t& nonleafs) {
			const int nsurf_ = this->nsurf;
			const int depth_ = this->depth;
			int npos = REL_COORD[M2L_Type].size();  // number of M2L relative positions
			m2ldata.resize(depth_);                 // initialize m2ldata

			// construct lists of target nodes for M2L operator at each level
			std::vector<nodeptrvec_t> trg_nodes(depth_);
			for (size_t i = 0; i < nonleafs.size(); i++) {
				trg_nodes[nonleafs[i]->level].push_back(nonleafs[i]);
			}

			// prepare for m2ldata for each level
			for (int l = 0; l < depth_; l++) {
				// construct M2L source nodes for current level
				std::set<node_t*> src_nodes_;
				for (size_t i = 0; i < trg_nodes[l].size(); i++) {
					nodeptrvec_t& M2L_list = trg_nodes[l][i]->M2L_list;
					for (int k = 0; k < npos; k++) {
						if (M2L_list[k]) src_nodes_.insert(M2L_list[k]);
					}
				}
				nodeptrvec_t src_nodes;
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
							nodeptrvec_t& M2L_list = trg_nodes[l][i]->M2L_list;
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

		void M2L(nodevec_t& nodes) {
			const int nsurf_ = this->nsurf;
			const int nfreq_ = this->nfreq;
			EXAFMM_ASSERT(nsurf_ > 0);
			EXAFMM_ASSERT(nfreq_ > 0);
			int fft_size = 2 * NCHILD * nfreq_;
			int nnodes = nodes.size();
			int npos = REL_COORD[M2L_Type].size();  // number of relative positions

			// allocate memory
			std::vector<potential_t> all_up_equiv, all_dn_equiv;
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
			real_t c[3] = { 0, 0, 0 };
			int nsurf_ = this->nsurf;
#pragma omp parallel for
			for (int level = 0; level <= this->depth; ++level) {
				// compute kernel matrix
				RealVec up_check_surf =
					box_surface_coordinates(this->p, this->r0, level, c, 2.95);
				RealVec up_equiv_surf =
					box_surface_coordinates(this->p, this->r0, level, c, 1.05);
				potential_matrix_t<> matrix_c2e =
					this->kernel_matrix(up_check_surf, up_equiv_surf);
				Eigen::BDCSVD<potential_matrix_t<>> svd(
					matrix_c2e, Eigen::ComputeFullU | Eigen::ComputeFullV);
				auto singularDiag = svd.singularValues();
				auto U = svd.matrixU();
				auto V = svd.matrixV();
				// pseudo-inverse, removing negligible terms.
				real_t max_S = std::reduce(
					singularDiag.data(), singularDiag.data() + singularDiag.size(), 0.,
					[](auto a1, auto a2) { return std::max(a1, a2); });
				for (int i = 0; i < nsurf_; i++) {
					singularDiag(i) =
						singularDiag(i) > EPS * max_S * 4 ? 1.0 / singularDiag(i) : 0.0;
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
			int n1 = this->p * 2;
			int nconv_ = this->nconv;
			int nfreq_ = this->nfreq;
			EXAFMM_ASSERT(n1 > 0);
			EXAFMM_ASSERT(nconv_ > 0);
			EXAFMM_ASSERT(nfreq_ > 0);
			int fft_size = 2 * nfreq_ * NCHILD * NCHILD;
			std::vector<RealVec> matrix_M2L_Helper(REL_COORD[M2L_Helper_Type].size(),
				RealVec(2 * nfreq_));
			std::vector<AlignedVec> matrix_M2L(REL_COORD[M2L_Type].size(),
				AlignedVec(fft_size));
			// create fft plan
			RealVec fftw_in(nconv_);
			RealVec fftw_out(2 * nfreq_);
			int dim[3] = { n1, n1, n1 };

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
							std::pow(0.5, l - 1);  // relative coords
					}
					RealVec conv_coord =
						convolution_grid(this->p, this->r0, l, coord);  // convolution grid
					// potentials on convolution grid
					auto convValue =
						this->kernel_matrix(conv_coord, trg_coord);
					fft_execute_dft(
						plan, reinterpret_cast<fft_complex*>(convValue.data()),
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
			fft_destroy_plan(plan);
		}

		void fft_up_equiv(std::vector<size_t>& fft_offset, ComplexVec& all_up_equiv,
			AlignedVec& fft_in) {
			int& nsurf_ = this->nsurf;
			int& nconv_ = this->nconv;
			int& nfreq_ = this->nfreq;
			int n1 = this->p * 2;
			auto map = generate_surf2conv_up(p);

			size_t fft_size = 2 * NCHILD * nfreq_;
			ComplexVec fftw_in(nconv_ * NCHILD);
			AlignedVec fftw_out(fft_size);
			int dim[3] = { n1, n1, n1 };
			fft_plan plan = fft_plan_many_dft(
				3, dim, NCHILD, reinterpret_cast<fft_complex*>(&fftw_in[0]), nullptr, 1,
				nconv_, (fft_complex*)(&fftw_out[0]), nullptr, 1, nfreq_, FFTW_FORWARD,
				FFTW_ESTIMATE);

#pragma omp parallel for
			for (long long int node_idx = 0; node_idx < fft_offset.size(); node_idx++) {
				RealVec buffer(fft_size, 0);
				ComplexVec equiv_t(NCHILD * nconv_, complex_t(0., 0.));

				complex_t* up_equiv =
					&all_up_equiv[fft_offset[node_idx]];  // offset ptr of node's 8
														  // child's up_equiv in
														  // all_up_equiv, size=8*nsurf_
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
						up_equiv_f[2 * (NCHILD * k + j) + 0] =
							buffer[2 * (nfreq_ * j + k) + 0];
						up_equiv_f[2 * (NCHILD * k + j) + 1] =
							buffer[2 * (nfreq_ * j + k) + 1];
					}
				}
			}
			fft_destroy_plan(plan);
		}

		void ifft_dn_check(std::vector<size_t>& ifft_offset, AlignedVec& fft_out,
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
			int dim[3] = { n1, n1, n1 };

			fft_plan plan = fft_plan_many_dft(
				3, dim, NCHILD, (fft_complex*)(&fftw_in[0]), nullptr, 1, nfreq_,
				reinterpret_cast<fft_complex*>(&fftw_out[0]), nullptr, 1, nconv_,
				FFTW_BACKWARD, FFTW_ESTIMATE);

#pragma omp parallel for
			for (long long int node_idx = 0; node_idx < ifft_offset.size();
				node_idx++) {
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
	};

	//! member function specialization for real type
	// template <class FmmKernel>
	// void Fmm<std::enable_if_t<!is_complex<typename
	// FmmKernel::potential_t>::value,
	//                          FmmKernel>>::precompute_M2L(std::ofstream& file) {
	//  int n1 = this->p * 2;
	//  int nconv_ = this->nconv;
	//  int nfreq_ = this->nfreq;
	//  EXAFMM_ASSERT(n1 > 0);
	//  EXAFMM_ASSERT(nconv_ > 0);
	//  EXAFMM_ASSERT(nfreq_ > 0);
	//  int fft_size = 2 * nfreq_ * NCHILD * NCHILD;
	//  std::vector<RealVec> matrix_M2L_Helper(REL_COORD[M2L_Helper_Type].size(),
	//                                         RealVec(2 * nfreq_));
	//  std::vector<AlignedVec> matrix_M2L(REL_COORD[M2L_Type].size(),
	//                                     AlignedVec(fft_size));
	//  // create fft plan
	//  RealVec fftw_in(nconv_);
	//  RealVec fftw_out(2 * nfreq_);
	//  int dim[3] = {n1, n1, n1};
	//  fft_plan plan = fft_plan_dft_r2c(
	//      3, dim, fftw_in.data(), reinterpret_cast<fft_complex*>(fftw_out.data()),
	//      FFTW_ESTIMATE);
	//  RealVec trg_coord(3, 0);
	//  for (int l = 1; l < this->depth + 1; ++l) {
	//    // compute M2L kernel matrix, perform DFT
	//#pragma omp parallel for
	//    for (long long int i = 0; i < REL_COORD[M2L_Helper_Type].size(); ++i) {
	//      real_t coord[3];
	//      for (int d = 0; d < 3; d++) {
	//        coord[d] = REL_COORD[M2L_Helper_Type][i][d] * this->r0 *
	//                   std::pow(0.5, l - 1);  // relative coords
	//      }
	//      RealVec conv_coord =
	//          convolution_grid(this->p, this->r0, l, coord);  // convolution grid
	//      // potentials on convolution grid
	//      auto convValue = this->kernel_matrix(conv_coord, trg_coord);
	//      fft_execute_dft_r2c(
	//          plan, convValue.data(),
	//          reinterpret_cast<fft_complex*>(matrix_M2L_Helper[i].data()));
	//    }
	//    // convert M2L_Helper to M2L and reorder data layout to improve locality
	//#pragma omp parallel for
	//    for (long long int i = 0; i < REL_COORD[M2L_Type].size(); ++i) {
	//      for (int j = 0; j < NCHILD * NCHILD;
	//           j++) {  // loop over child's relative positions
	//        int child_rel_idx = M2L_INDEX_MAP[i][j];
	//        if (child_rel_idx != -1) {
	//          for (int k = 0; k < nfreq_; k++) {  // loop over frequencies
	//            int new_idx = k * (2 * NCHILD * NCHILD) + 2 * j;
	//            matrix_M2L[i][new_idx + 0] =
	//                matrix_M2L_Helper[child_rel_idx][k * 2 + 0] / nconv_;  // real
	//            matrix_M2L[i][new_idx + 1] =
	//                matrix_M2L_Helper[child_rel_idx][k * 2 + 1] / nconv_;  // imag
	//          }
	//        }
	//      }
	//    }
	//    // write to file
	//    for (auto& vec : matrix_M2L) {
	//      file.write(reinterpret_cast<char*>(vec.data()),
	//                 fft_size * sizeof(real_t));
	//    }
	//  }
	//  fft_destroy_plan(plan);
	//}

	//! member function specialization for complex type
	// template <class FmmKernel>
	// void Fmm<std::enable_if_t<is_complex<typename FmmKernel::potential_t>::value,
	//                          FmmKernel>>::precompute_M2L(std::ofstream& file) {
	// template <class FmmKernel>
	// void Fmm<FmmKernel>::precompute_M2L(std::ofstream& file) {
	//  int n1 = this->p * 2;
	//  int nconv_ = this->nconv;
	//  int nfreq_ = this->nfreq;
	//  EXAFMM_ASSERT(n1 > 0);
	//  EXAFMM_ASSERT(nconv_ > 0);
	//  EXAFMM_ASSERT(nfreq_ > 0);
	//  int fft_size = 2 * nfreq_ * NCHILD * NCHILD;
	//  std::vector<RealVec> matrix_M2L_Helper(REL_COORD[M2L_Helper_Type].size(),
	//                                         RealVec(2 * nfreq_));
	//  std::vector<AlignedVec> matrix_M2L(REL_COORD[M2L_Type].size(),
	//                                     AlignedVec(fft_size));
	//  // create fft plan
	//  RealVec fftw_in(nconv_);
	//  RealVec fftw_out(2 * nfreq_);
	//  int dim[3] = {n1, n1, n1};
	//
	//  fft_plan plan =
	//      fft_plan_dft(3, dim, reinterpret_cast<fft_complex*>(fftw_in.data()),
	//                   reinterpret_cast<fft_complex*>(fftw_out.data()),
	//                   FFTW_FORWARD, FFTW_ESTIMATE);
	//  RealVec trg_coord(3, 0);
	//  for (int l = 1; l < this->depth + 1; ++l) {
	//// compute M2L kernel matrix, perform DFT
	//#pragma omp parallel for
	//    for (long long int i = 0; i < REL_COORD[M2L_Helper_Type].size(); ++i) {
	//      real_t coord[3];
	//      for (int d = 0; d < 3; d++) {
	//        coord[d] = REL_COORD[M2L_Helper_Type][i][d] * this->r0 *
	//                   std::pow(0.5, l - 1);  // relative coords
	//      }
	//      RealVec conv_coord =
	//          convolution_grid(this->p, this->r0, l, coord);  // convolution grid
	//      // potentials on convolution grid
	//      potential_vector_t convValue = this->kernel_matrix(conv_coord,
	//      trg_coord);
	//
	//      fft_execute_dft(
	//          plan, reinterpret_cast<fft_complex*>(convValue.data()),
	//          reinterpret_cast<fft_complex*>(matrix_M2L_Helper[i].data()));
	//    }
	//    // convert M2L_Helper to M2L and reorder data layout to improve locality
	//#pragma omp parallel for
	//    for (long long int i = 0; i < REL_COORD[M2L_Type].size(); ++i) {
	//      for (int j = 0; j < NCHILD * NCHILD;
	//           j++) {  // loop over child's relative positions
	//        int child_rel_idx = M2L_INDEX_MAP[i][j];
	//        if (child_rel_idx != -1) {
	//          for (int k = 0; k < nfreq_; k++) {  // loop over frequencies
	//            int new_idx = k * (2 * NCHILD * NCHILD) + 2 * j;
	//            matrix_M2L[i][new_idx + 0] =
	//                matrix_M2L_Helper[child_rel_idx][k * 2 + 0] / nconv_;  // real
	//            matrix_M2L[i][new_idx + 1] =
	//                matrix_M2L_Helper[child_rel_idx][k * 2 + 1] / nconv_;  // imag
	//          }
	//        }
	//      }
	//    }
	//    // write to file
	//    for (auto& vec : matrix_M2L) {
	//      file.write(reinterpret_cast<char*>(vec.data()),
	//                 fft_size * sizeof(real_t));
	//    }
	//  }
	//  fft_destroy_plan(plan);
	//}

	// template <class FmmKernel>
	// void Fmm<std::enable_if_t<!is_complex<typename
	// FmmKernel::potential_t>::value,
	//                          FmmKernel>>::fft_up_equiv(std::vector<size_t>&
	//                                                        fft_offset,
	//                                                    RealVec& all_up_equiv,
	//                                                    AlignedVec& fft_in) {
	//  int& nsurf_ = this->nsurf;
	//  int& nconv_ = this->nconv;
	//  int& nfreq_ = this->nfreq;
	//  int n1 = this->p * 2;
	//  auto map = generate_surf2conv_up(p);
	//
	//  size_t fft_size = 2 * NCHILD * nfreq_;
	//  AlignedVec fftw_in(nconv_ * NCHILD);
	//  AlignedVec fftw_out(fft_size);
	//  int dim[3] = {n1, n1, n1};
	//  fft_plan plan = fft_plan_many_dft_r2c(
	//      3, dim, NCHILD, (real_t*)&fftw_in[0], nullptr, 1, nconv_,
	//      (fft_complex*)(&fftw_out[0]), nullptr, 1, nfreq_, FFTW_ESTIMATE);
	//
	//#pragma omp parallel for
	//  for (long long int node_idx = 0; node_idx < fft_offset.size(); node_idx++)
	//  {
	//    RealVec buffer(fft_size, 0);
	//    RealVec equiv_t(NCHILD * nconv_, 0.);
	//
	//    real_t* up_equiv =
	//        &all_up_equiv[fft_offset[node_idx]];  // offset ptr of node's 8
	//        child's
	//                                              // up_equiv in all_up_equiv,
	//                                              // size=8*nsurf_
	//    real_t* up_equiv_f =
	//        &fft_in[fft_size * node_idx];  // offset ptr of node_idx in fft_in
	//                                       // vector, size=fftsize
	//
	//    for (int k = 0; k < nsurf_; k++) {
	//      size_t idx = map[k];
	//      for (int j = 0; j < NCHILD; j++)
	//        equiv_t[idx + j * nconv_] = up_equiv[j * nsurf_ + k];
	//    }
	//    fft_execute_dft_r2c(plan, &equiv_t[0], (fft_complex*)&buffer[0]);
	//    for (int k = 0; k < nfreq_; k++) {
	//      for (int j = 0; j < NCHILD; j++) {
	//        up_equiv_f[2 * (NCHILD * k + j) + 0] = buffer[2 * (nfreq_ * j + k) +
	//        0]; up_equiv_f[2 * (NCHILD * k + j) + 1] = buffer[2 * (nfreq_ * j +
	//        k)
	//        + 1];
	//      }
	//    }
	//  }
	//  fft_destroy_plan(plan);
	//}

	// template <class FmmKernel>
	// void Fmm<std::enable_if_t<is_complex<typename FmmKernel::potential_t>::value,
	//                          FmmKernel>>::fft_up_equiv(std::vector<size_t>&
	//                                                        fft_offset,
	//                                                    ComplexVec& all_up_equiv,
	//                                                    AlignedVec& fft_in) {
	// template <typename T>
	// inline void Fmm<FmmKernel>::fft_up_equiv(std::vector<size_t>& fft_offset,
	//                                         ComplexVec& all_up_equiv,
	//                                         AlignedVec& fft_in) {
	//  int& nsurf_ = this->nsurf;
	//  int& nconv_ = this->nconv;
	//  int& nfreq_ = this->nfreq;
	//  int n1 = this->p * 2;
	//  auto map = generate_surf2conv_up(p);
	//
	//  size_t fft_size = 2 * NCHILD * nfreq_;
	//  ComplexVec fftw_in(nconv_ * NCHILD);
	//  AlignedVec fftw_out(fft_size);
	//  int dim[3] = {n1, n1, n1};
	//  fft_plan plan = fft_plan_many_dft(
	//      3, dim, NCHILD, reinterpret_cast<fft_complex*>(&fftw_in[0]), nullptr, 1,
	//      nconv_, (fft_complex*)(&fftw_out[0]), nullptr, 1, nfreq_, FFTW_FORWARD,
	//      FFTW_ESTIMATE);
	//
	//#pragma omp parallel for
	//  for (long long int node_idx = 0; node_idx < fft_offset.size(); node_idx++) {
	//    RealVec buffer(fft_size, 0);
	//    ComplexVec equiv_t(NCHILD * nconv_, complex_t(0., 0.));
	//
	//    complex_t* up_equiv =
	//        &all_up_equiv[fft_offset[node_idx]];  // offset ptr of node's 8
	//                                              // child's up_equiv in
	//                                              // all_up_equiv, size=8*nsurf_
	//    real_t* up_equiv_f =
	//        &fft_in[fft_size * node_idx];  // offset ptr of node_idx in fft_in
	//                                       // vector, size=fftsize
	//
	//    for (int k = 0; k < nsurf_; k++) {
	//      size_t idx = map[k];
	//      for (int j = 0; j < NCHILD; j++)
	//        equiv_t[idx + j * nconv_] = up_equiv[j * nsurf_ + k];
	//    }
	//
	//    fft_execute_dft(plan, reinterpret_cast<fft_complex*>(&equiv_t[0]),
	//                    (fft_complex*)&buffer[0]);
	//    for (int k = 0; k < nfreq_; k++) {
	//      for (int j = 0; j < NCHILD; j++) {
	//        up_equiv_f[2 * (NCHILD * k + j) + 0] = buffer[2 * (nfreq_ * j + k) +
	//        0]; up_equiv_f[2 * (NCHILD * k + j) + 1] = buffer[2 * (nfreq_ * j + k)
	//        + 1];
	//      }
	//    }
	//  }
	//  fft_destroy_plan(plan);
	//}
	//
	// template <class FmmKernel>
	// void Fmm<std::enable_if_t<!is_complex<typename
	// FmmKernel::potential_t>::value,
	//                          FmmKernel>>::ifft_dn_check(std::vector<size_t>&
	//                                                         ifft_offset,
	//                                                     AlignedVec& fft_out,
	//                                                     RealVec& all_dn_equiv)
	//                                                     {
	//  int& nsurf_ = this->nsurf;
	//  int& nconv_ = this->nconv;
	//  int& nfreq_ = this->nfreq;
	//  int n1 = this->p * 2;
	//  auto map = generate_surf2conv_dn(p);
	//
	//  size_t fft_size = 2 * NCHILD * nfreq_;
	//  AlignedVec fftw_in(fft_size);
	//  AlignedVec fftw_out(nconv_ * NCHILD);
	//  int dim[3] = {n1, n1, n1};
	//
	//  fft_plan plan = fft_plan_many_dft_c2r(
	//      3, dim, NCHILD, (fft_complex*)(&fftw_in[0]), nullptr, 1, nfreq_,
	//      (real_t*)(&fftw_out[0]), nullptr, 1, nconv_, FFTW_ESTIMATE);
	//
	//#pragma omp parallel for
	//  for (long long int node_idx = 0; node_idx < ifft_offset.size();
	//  node_idx++)
	//  {
	//    RealVec buffer0(fft_size, 0);
	//    RealVec buffer1(fft_size, 0);
	//    real_t* dn_check_f =
	//        &fft_out[fft_size * node_idx];  // offset ptr for node_idx in
	//        fft_out
	//                                        // vector, size=fftsize
	//    real_t* dn_equiv =
	//        &all_dn_equiv[ifft_offset[node_idx]];  // offset ptr for node_idx's
	//                                               // child's dn_equiv in
	//                                               // all_dn_equiv,
	//                                               size=numChilds
	//                                               *
	//                                               // nsurf_
	//    for (int k = 0; k < nfreq_; k++)
	//      for (int j = 0; j < NCHILD; j++) {
	//        buffer0[2 * (nfreq_ * j + k) + 0] =
	//            dn_check_f[2 * (NCHILD * k + j) + 0];
	//        buffer0[2 * (nfreq_ * j + k) + 1] =
	//            dn_check_f[2 * (NCHILD * k + j) + 1];
	//      }
	//    fft_execute_dft_c2r(plan, (fft_complex*)&buffer0[0],
	//                        (real_t*)(&buffer1[0]));
	//    for (int k = 0; k < nsurf_; k++) {
	//      size_t idx = map[k];
	//      for (int j = 0; j < NCHILD; j++)
	//        dn_equiv[nsurf_ * j + k] += buffer1[idx + j * nconv_];
	//    }
	//  }
	//  fft_destroy_plan(plan);
	//}

	// template <class FmmKernel>
	// void Fmm<std::enable_if_t<is_complex<typename FmmKernel::potential_t>::value,
	//                          FmmKernel>>::ifft_dn_check(std::vector<size_t>&
	//                                                         ifft_offset,
	//                                                     AlignedVec& fft_out,
	//                                                     ComplexVec& all_dn_equiv)
	//                                                     {
	// template <class FmmKernel>
	// inline void Fmm<FmmKernel>::ifft_dn_check(std::vector<size_t>& ifft_offset,
	//                                          AlignedVec& fft_out,
	//                                          ComplexVec& all_dn_equiv) {
	//  int& nsurf_ = this->nsurf;
	//  int& nconv_ = this->nconv;
	//  int& nfreq_ = this->nfreq;
	//  assert(fft_out.size() >= ifft_offset.size() * nfreq_ * NCHILD);
	//  int n1 = this->p * 2;
	//  auto map = generate_surf2conv_dn(p);
	//
	//  size_t fft_size = 2 * NCHILD * nfreq_;
	//  AlignedVec fftw_in(fft_size);
	//  ComplexVec fftw_out(nconv_ * NCHILD);
	//  int dim[3] = {n1, n1, n1};
	//
	//  fft_plan plan =
	//      fft_plan_many_dft(3, dim, NCHILD, (fft_complex*)(&fftw_in[0]), nullptr,
	//      1,
	//                        nfreq_, reinterpret_cast<fft_complex*>(&fftw_out[0]),
	//                        nullptr, 1, nconv_, FFTW_BACKWARD, FFTW_ESTIMATE);
	//
	//#pragma omp parallel for
	//  for (long long int node_idx = 0; node_idx < ifft_offset.size(); node_idx++)
	//  {
	//    RealVec buffer0(fft_size, 0);
	//    ComplexVec buffer1(NCHILD * nconv_, 0);
	//    real_t* dn_check_f = &fft_out[fft_size * node_idx];
	//    complex_t* dn_equiv = &all_dn_equiv[ifft_offset[node_idx]];
	//    for (int k = 0; k < nfreq_; k++)
	//      for (int j = 0; j < NCHILD; j++) {
	//        buffer0[2 * (nfreq_ * j + k) + 0] =
	//            dn_check_f[2 * (NCHILD * k + j) + 0];
	//        buffer0[2 * (nfreq_ * j + k) + 1] =
	//            dn_check_f[2 * (NCHILD * k + j) + 1];
	//      }
	//    fft_execute_dft(plan, (fft_complex*)&buffer0[0],
	//                    reinterpret_cast<fft_complex*>(&buffer1[0]));
	//    for (int k = 0; k < nsurf_; k++) {
	//      size_t idx = map[k];
	//      for (int j = 0; j < NCHILD; j++)
	//        dn_equiv[nsurf_ * j + k] += buffer1[idx + j * nconv_];
	//    }
	//  }
	//  fft_destroy_plan(plan);
	//}

}  // namespace ExaFMM

#endif  // INCLUDE_EXAFMM_FMM_H_
